#!/usr/bin/env python3
"""四层验证一键回归 —— 与 Arm 官方实现位精确对拍。

四层门禁（逐层收紧）：
  1. DP4A 算式等价性          numpy 内部：DP4A 路径 vs TOSA 标准路径
  2. OpSDot 打包语义          真机最小测试（构造 0xFF/0x80 等边界）
  3. GPU 实现 vs numpy 参考   两档模型 × 多尺寸，随机输入
  4. GPU 实现 vs Arm 官方     官方 scenario 真实帧 + 仿真层 oracle

用法:
    python verify_all.py [--quick]        # --quick 跳过 540p 全分辨率项

前提:
    - 已在 E:/.luna/nss-vk/oracle 生成 oracle（见 README.md）
    - nss-verify venv 已装 ai-ml-emulation-layer-for-vulkan + ai-ml-sdk-scenario-runner
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "nss-tools")
MODELS = os.path.join(ROOT, "nss-model")
PY = sys.executable

REF = os.path.join(TOOLS, "nss_ref.py")
CMP = os.path.join(TOOLS, "nss_compare.py")
VKRUN = os.path.join(HERE, "nss_vk.py")

HIGH = os.path.join(MODELS, "nss_v1_0_1_high_int8.vgf")
MIDLOW = os.path.join(MODELS, "nss_v1_0_1_mid_low_int8.vgf")

results = []


def record(layer, name, ok, detail=""):
    results.append((layer, name, ok, detail))
    mark = "通过" if ok else "失败"
    print(f"  [{mark}] {name:<48} {detail}")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def layer1():
    print("\n=== 第 1 层：DP4A 算式等价性 ===")
    for name, vgf in (("high", HIGH), ("mid_low", MIDLOW)):
        r = run([PY, REF, vgf, "--size", "64", "--dp4a-verify"])
        out = r.stdout + r.stderr
        ok = "成立" in out or "一致" in out
        record(1, f"{name} DP4A 路径 == TOSA 标准路径", ok,
               "修正项推导成立" if ok else out.strip()[-120:])


def layer2():
    print("\n=== 第 2 层：OpSDot 打包语义（真机） ===")
    r = run([PY, os.path.join(HERE, "dot4_test.py")])
    out = r.stdout + r.stderr
    ok = "被解释为 4x8 打包" in out
    record(2, "OpSDot 4x8 打包语义", ok,
           "含 0xFF/0x80/0x7F 边界" if ok else out.strip()[-160:])


def _cmp(a_path, b_path):
    """返回 (不一致数, 总数) 或 None。"""
    a, b = np.load(a_path), np.load(b_path)
    ka = [k for k in a.files if k.startswith("out")]
    kb = [k for k in b.files if k.startswith("out")]
    nz = tot = 0
    for k in sorted(set(ka) & set(kb)):
        x = np.squeeze(a[k]).astype(np.int32)
        y = np.squeeze(b[k]).astype(np.int32)
        if x.shape != y.shape:
            return None
        nz += int((x != y).sum())
        tot += x.size
    return (nz, tot)


def layer3(sizes):
    print("\n=== 第 3 层：GPU 实现 vs numpy 黄金参考（随机输入） ===")
    work = os.path.join(HERE, "reg")
    os.makedirs(work, exist_ok=True)
    for name, vgf in (("high", HIGH), ("mid_low", MIDLOW)):
        for sz in sizes:
            g = os.path.join(work, f"g_{name}_{sz}.npz")
            rr = os.path.join(work, f"r_{name}_{sz}.npz")
            run([PY, VKRUN, vgf, "--h", str(sz), "--w", str(sz), "-o", g])
            run([PY, REF, vgf, "--input", g, "-o", rr])
            if not (os.path.exists(g) and os.path.exists(rr)):
                record(3, f"{name} {sz}x{sz}", False, "产物缺失")
                continue
            res = _cmp(rr, g)
            if res is None:
                record(3, f"{name} {sz}x{sz}", False, "形状不一致")
            else:
                nz, tot = res
                record(3, f"{name} {sz}x{sz}", nz == 0, f"{nz}/{tot}")


def layer4(quick):
    print("\n=== 第 4 层：GPU 实现 vs Arm 官方（真实帧） ===")
    cases = [("high 544x960 (540p)", HIGH, "ref_official/out_high", "out_input_tensor.npy",
              "out_graph_0.npy", "out_graph_1.npy"),
             ("mid  272x480", MIDLOW, "ref_official/out_mid", "out_input_tensor.npy",
              "out_graph_0.npy", "out_graph_1.npy")]
    if quick:
        cases = cases[1:]
    for idx, (label, vgf, reldir, fin, fo0, fo1) in enumerate(cases):
        d = os.path.join(HERE, reldir)
        if not os.path.isdir(d):
            record(4, label, False, f"缺官方基准 {reldir}（先跑 scenario-runner）")
            continue
        inp = os.path.join(d, fin)
        if not os.path.exists(inp):
            record(4, label, False, f"缺 {fin}")
            continue
        g = os.path.join(HERE, f"v_gpu_{idx}.npz")
        r = run([PY, VKRUN, vgf, "--input", inp, "-o", g])
        if not os.path.exists(g):
            record(4, label, False, (r.stdout + r.stderr).strip()[-140:])
            continue
        z = np.load(g)
        gk = [k for k in z.files if k.startswith("out")]
        nz = tot = 0
        detail = []
        for fo, k in zip((fo0, fo1), sorted(gk)):
            o = np.squeeze(np.load(os.path.join(d, fo))).astype(np.int32)
            m = np.squeeze(z[k]).astype(np.int32)
            if o.shape != m.shape:
                detail.append(f"{fo} 形状 {m.shape} vs {o.shape}")
                nz += 1
                continue
            n = int((o != m).sum())
            nz += n
            tot += o.size
            detail.append(f"{fo.split('_')[-1][:-4]}:{n}")
        record(4, f"{label}  vs Arm 官方", nz == 0,
               f"{nz}/{tot}  ({' '.join(detail)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过 540p 全分辨率项")
    args = ap.parse_args()

    print("=" * 84)
    print("NSS DP4A 后端 —— 四层验证回归")
    print("=" * 84)
    layer1()
    layer2()
    layer3([64, 128] if args.quick else [128, 256])
    layer4(args.quick)

    print("\n" + "=" * 84)
    failed = [r for r in results if not r[2]]
    for lyr in (1, 2, 3, 4):
        rs = [r for r in results if r[0] == lyr]
        ok = sum(1 for r in rs if r[2])
        print(f"  第 {lyr} 层: {ok}/{len(rs)} 通过")
    print("=" * 84)
    if failed:
        print(f"\n共 {len(failed)} 项失败:")
        for lyr, name, _, detail in failed:
            print(f"  [L{lyr}] {name}  {detail}")
        return 1
    print("\n四层验证全部通过 —— 与 Arm 官方实现位精确一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
