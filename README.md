# nss_kernel —— NSS GPU 内核源与内核级验证台

在真实 GPU 上执行 Arm NSS 神经网络的 compute 内核（GLSL→SPIR-V）与配套验证工具，
输出与 numpy 黄金参考**位精确一致**。

**不依赖 C++ 编译器、不依赖第三方 Python 包**：用 `ctypes` 直连 `vulkan-1.dll`，
着色器由 glslangValidator 编译成 SPIR-V。

与 [nss](https://github.com/eastear23333/nss)（nss_dp4a.dll，推理后端）的关系：
本仓库的 `shaders/*.comp` 编译出的 SPIR-V 被 nss 仓库烘焙进
`generated/nss_spirv_embed.h`；本仓库同时提供内核级验证台（单算子 / 整网 / 官方对拍）。

---

## 快速开始

依赖：Python 3.12+（仅 `numpy`）、glslangValidator（Vulkan SDK 的 `Bin/` 下，
或 Arm neural-graphics SDK 的 `sdk/tools/binary_store/` 下）。

```bash
# 0) 约定：nss 仓库 clone 到同级目录（参考脚本在 ../nss/tools/）
git clone https://github.com/eastear23333/nss ../nss

# 1) 编译着色器（改过 .comp 后需要重跑）
cd shaders
GLSLANG=/path/to/glslangValidator.exe
for f in conv_rq conv_tc resize2x concat_copy; do $GLSLANG -V --target-env vulkan1.2 $f.comp -o $f.spv; done

# 2) 跑网络（544×960 ≈ 540p 渲染分辨率；模型 .vgf 需按 nss 仓库 README 的许可节获取）
cd ..
PY=python
$PY -u nss_vk.py "<MODELS>/nss_v1_0_1_high_int8.vgf" --h 544 --w 960 -o gpu.npz --repeat 30

# 3) 与 numpy 黄金参考对拍（判据：位精确相等）
$PY -u ../nss/tools/nss_ref.py "<MODELS>/nss_v1_0_1_high_int8.vgf" --input gpu.npz -o ref.npz
$PY -u ../nss/tools/nss_compare.py ref.npz gpu.npz
```

常用参数：`--h/--w`（必须 8 的倍数）、`--seed`、`--repeat`（测平均耗时）、
`--dump-all DIR`（导出全部中间张量）、`--debug-op N`（导出第 N 个卷积的累加器/量化中间量）。

**内存模式**：默认 `DEVICE_LOCAL`（显存，经 staging 往返），这是生产配置。
`--host-memory` 可切回系统内存，**仅供对照**——慢一个数量级以上。

---

## 已验证结果

### 与 Arm 官方实现位精确一致（第 4 层门禁）

用官方 `scenario-runner` + `ai-ml-emulation-layer-for-vulkan` 仿真层跑真实帧，
拿数据图边界处的原始张量做三方对拍（官方 / numpy 参考 / GPU 实现）：

| 档位 | 输入 | KPN | temporal |
|---|---|---|---|
| high | 544×960×12 | 0/1,175,040 ✓ | 0/2,088,960 ✓ |
| mid | 272×480×12 | 0/130,560 ✓ | 0/522,240 ✓ |

### 四层验证一键回归

```bash
python verify_all.py            # 完整（含 540p）
python verify_all.py --quick    # 跳过 540p
```

| 层 | 内容 | 结果 |
|---|---|---|
| 1 | DP4A 算式等价性（两档） | 2/2 ✓ |
| 2 | `OpSDot` 4x8 打包语义（真机） | 1/1 ✓ |
| 3 | GPU 实现 vs numpy 参考（两档 × 多尺寸） | 4/4 ✓ |
| 4 | GPU 实现 vs **Arm 官方**（真实帧） | 2/2 ✓ |

另有 64/128/256/544×960 各尺寸的位精确回归，输出形状与资源表吻合：
KPN `(H/4, W/4, 36)`（mid_low 为 16 通道）、时序 `(H, W, 4)`。

---

## 性能要点

具体基准数字随驱动版本与机型漂移，此处只留可复现的结构性结论
（复测方法：`nss_vk.py --repeat N` 测帧均，`../nss/tools/nss_compare.py` 对拍）：

- **中间张量放显存**（`DEVICE_LOCAL` + staging 往返）是决定性优化；放系统内存
  慢一个数量级以上，仅作对照。
- 相对 Arm 仿真层（`VK_LAYER_ML_Graph_Emulation`，37 算子逐算子调度）
  有**数量级**提升。
- **权重放 workgroup shared memory 无收益**：权重总量仅 146 KiB，L2 已完全容纳，
  瓶颈不在权重带宽（有 A/B 实测数据支撑）。
- DP4A 内核耗时随输入面积线性增长 → 已是**算力受限**，进一步提速依赖
  Tensor Core 路径（`conv_tc.comp`）或每线程多像素。

---

## 硬件前提（实测）

| 能力 | GTX 1660 Ti (TU116) | RTX 4060 (AD107) |
|---|---|---|
| `integerDotProduct4x8BitPackedSignedAccelerated` | 是 | 是 |
| `shaderIntegerDotProduct` 特性 | 可启用 | 可启用 |
| `shaderInt64`（requantize 需要） | 是 | 是 |
| `VK_KHR_cooperative_matrix` | **无**（无 Tensor Core）→ 走 DP4A | **有**，int8 MMA 16×16×32（sint8→sint32，subgroup）→ `conv_tc.comp` 生效 |

用 `../nss/tools/vk_probe.py` 可以在别的卡上复测。

---

## 算子与融合

35 个 TOSA 算子 → **19 次 dispatch**：

| 内核 | 次数 | 说明 |
|---|---|---|
| `conv_rq.comp` | 14 | 融合 `CONV2D + bias + DP4A修正 + RESCALE(+LUT)`；OpSDot/DP4A 路径 |
| `conv_tc.comp` | —（按层替换 conv_rq） | cooperative matrix Tensor Core 变体（见下节） |
| `resize2x.comp` | 3 | 2× 最近邻 |
| `concat_copy.comp` | 2 | 通道轴拼接 |

**尚未做的优化**：CONCAT/RESIZE 融进消费者卷积可降到 16 次。

### conv_tc.comp（Tensor Core 变体）要点

- 形状取自驱动上报的 sint8 形状 **M=16 N=16 K=32**（一次 MMA 出 16 个输出通道）。
  coop-mat 的 M/N/K 是 **SPIR-V 编译期常量**，运行时不可变 —— 跨厂商泛化必须
  按形状出多份编译产物。
- 矩阵切分：A（16×32）= 16 个同一行的连续输出像素 × 32 个输入通道；
  B（32×16）= 32 输入通道 × 16 输出通道；C/累加器（16×16）int32。
- K 轴按 `(ky, kx, cdx)` 分块，每块 32 个通道；Cin 不是 32 倍数时激活用 `z_a(−128)`
  填充、权重补 0。
- int8 矩阵放在 shared `uint[]` 时行跨距要传 **8 word**（=32 字节），不是 32。
- 提供单累加器版本（同一时刻只有 1 个 16×16 累加器存活，省寄存器）与
  acc0..acc3 恒常版本，按占用率取舍。

---

## 关键实现要点

### 1. DP4A 打包是零成本的

`.vgf` 权重是 OHWI int8 且 `Cin` 全是 4 的倍数，K 方向天然连续——直接
`np.ascontiguousarray(w).view(np.uint32)` 就是 DP4A 需要的格式，**无需重排**。
激活 NHWC 同理。**因此激活缓冲直接用 `uint32` 视图，一次读写就是 4 个连续通道。**

### 2. 边框预填激活零点 = 严格等价于「跳过越界 tap」

每个激活缓冲四周留 1 像素，**预填 `0x80`（= int8 的 −128 = NSS 的激活零点）**，
内核只写内部区域、保留边框。于是 `(q_a − z_a)` 在越界处恰好为 0，
与 TOSA 的「跳过 tap」严格等价——**这样逐输出通道的修正项才可以是常量**。

> 若按惯例填 0 而不是填 z_a，修正项会变成边界相关量，结果错误。

### 3. 修正项

权重对称（`z_w = 0`）时 `Σ(q_a − z_a)q_w = Σ q_a q_w − z_a·Σ_all q_w`，
后一项逐输出通道是常量，构建期算好并与 bias 合并上传（host 侧的 `bc` 缓冲）。

### 4. ReLU 是隐含的，不要再插 max

NSS 的 12 个 ReLU 层 `output_zero_point = −128`，配上限幅到 `[−128,127]`
自动恒非负。**内核里额外插 `max(x,0)` 会二次钳位、破坏语义。**

### 5. requantize 必须用 64 位

`acc (~25 位) × multiplier (最大 31 位)` 乘积可达 **56 位**，32 位会溢出。
直接 `int64_t`（需 `shaderInt64`）。

### 6. glslang 15.4 不支持 `GL_EXT_shader_integer_dot_product`

官方扩展名不认、内建函数 `dot4AddI8Packed` 也不认。解法是用
**`GL_EXT_spirv_intrinsics` 直接声明 `OpSDot`**：

```glsl
#extension GL_EXT_spirv_intrinsics : require
// 省略最后一个 Packed Vector Format 操作数（实测等价于显式指定 4x8Bit）
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"],
                  capabilities = [6018], id = 4450)
int spv_sdot(uint v1, uint v2);
...
acc += spv_sdot(a, w);      // OpSDot 不含累加操作数，累加在外面做
```

用 `dot4_test.py` 可在任意卡上验证 `OpSDot` 的打包语义与符号解释。

---

## 踩过的坑（都靠位精确对拍抓出来）

| 现象 | 根因 |
|---|---|
| **与官方差 10–50%，差异集中在 8–16 像素边框** | **TOSA RESIZE 的 `offset` 操作数是 −1，不是 0** —— 即索引为 `oy/2` 而非 `(oy+1)/2`。两者只在边界不同，内部完全等价 |
| 输出整体沿对角线偏移一格 | 内核读输入时加了边框偏移 `+1`，**写输出时忘了加** |
| Resize 后所有层错 | 最近邻索引**没钳到 `H-1/W-1`**，最后一行读到边框 |
| 逐层对拍「从 op11 起全错」 | **harness 的 bug**：IR 里图输入张量 id 恰好是 11，我把它登记进 `result_buf` 时覆盖了 conv5 rescale 的融合映射 |
| 逐元素对拍差 255 左右 | 对拍脚本把 GPU 的 uint8 当 int8 读了（同一批值的两种表示） |
| 全零点输入下内部仍有差异 | 均匀输入经最近邻上采样**仍是均匀的**，所以「内部一致」**不能**验证 resize 映射 —— 必须用有空间结构的输入 |

> **结论：位精确判据不是形式主义。** 上面每一个问题都让输出「看起来正常」，
> 只有逐位比较才能发现。而 `--debug-op` 导出累加器/量化中间量是把范围从「整网」
> 缩到「单个算子」的关键手段。
>
> **定位 `resize offset` 的方法值得复用**：先用「全零点输入」把差异压到边框
> （差异与输入内容无关 ⇒ 只取决于位置 ⇒ 边界/分块效应），再逐段裁剪确认
> 差异范围恰为 `1→2→4→8` 像素（三次 2× 上采样的边界传播），
> 最后从 SPIR-V 里解出 `offset` 操作数 `= -1` 验证。

---

## 剩余优化空间

| 项 | 状态 |
|---|---|
| 中间张量放显存 | **已做** —— 数量级收益，决定性的一步 |
| 权重放 shared memory | **已试，无收益** —— 权重共 146 KiB，L2 已完全容纳 |
| cooperative matrix 路径 | **已做**（`conv_tc.comp`，NVIDIA 16×16×32；AMD/Intel 形状泛化见 nss 仓库 README） |
| CONCAT/RESIZE 融进消费者卷积 | 未做，19 → 16 dispatch |
| 每线程多像素（提高 DP4A/加载 比） | 未做，DP4A 路径距峰值利用率仍有空间 |

---

## 官方 oracle（第 4 层验证的基础设施）

`oracle/` 下是一个最小数据图场景，**只跑 NSS 图、不跑前后处理**，
因此可以把任意输入喂给 Arm 官方实现并拿到输出，作为可任意查询的 oracle：

```bash
# 生成场景（位置参数：输入 .npy，输出目录）
python oracle/make_oracle.py 输入.npy oracle/run

# 用仿真层跑（需要 nss-verify venv，VK_LAYER_PATH 指向仿真层部署目录）
export VK_LAYER_PATH="$VENV/Lib/site-packages/emulation_layer/deploy/bin"
export VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
cd oracle/run && scenario-runner --scenario mini_graph.json --output out --log-level warning
```

产物 `out/oracle_out0.npy`（KPN）、`out/oracle_out1.npy`（temporal）即官方输出。
关键点是配置里 `tensor` 支持 `src` 键（从文件导入），官方自带场景只用了 `dst`。

**官方输出的确定性已验证**：连跑 3 次逐位相同。

---

## 文件

| 文件 | 说明 |
|---|---|
| `nss_vk.py` | 运行器（Vulkan 全部用 ctypes，无第三方依赖） |
| `verify_all.py` | **四层验证一键回归** |
| `nss_verify_official.py` | 第 4 层：与 Arm 官方真实帧对拍 |
| `dot4_test.py` | `OpSDot` 语义最小测试 |
| `msvc_env.sh` | MSVC 环境变量速查（Git Bash 下编译 nss 仓库用） |
| `oracle/make_oracle.py` | 生成最小数据图场景（官方 oracle） |
| `oracle/query.py` | 用受控输入查询 oracle（脉冲/零点等模式） |
| `shaders/conv_rq.comp` | 融合卷积内核（OpSDot/DP4A） |
| `shaders/conv_tc.comp` | 融合卷积内核（cooperative matrix Tensor Core） |
| `shaders/resize2x.comp` | 2× 最近邻 |
| `shaders/concat_copy.comp` | 通道拼接 |
| `shaders/dp4a_test.comp`、`t2.comp`、`t3.comp` | 编译探针（可删） |

---

## 许可

内核源码与验证脚本为自研实现（与 [nss](https://github.com/eastear23333/nss)
仓库的 nss_dp4a.dll 配套），MIT 许可（© 2026 eastear23333）。
`shaders/*.comp` 编译产物（.spv / nss 仓库的 `generated/nss_spirv_embed.h`）随 nss 仓库分发；
oracle / ref_official / reg 下的参考快照不入库，由 `make_oracle.py` 与 verify 脚本重新生成。
