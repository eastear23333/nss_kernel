#!/usr/bin/env python3
"""构造最小场景：把任意输入喂给 Arm 官方数据图，导出官方输出。

得到的是一个可任意查询的 **oracle** —— 不再依赖官方 scenario 自带的那一帧输入，
可以用受控输入做二分定位。

用法:
    python make_oracle.py <输入.npy 或 --random> <输出目录>

产物: <输出目录>/mini_graph.json  （用 scenario-runner 运行它）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VGF_REL = "../../../nss-model/scenario/assets/960x540_1920x1080/2_nss-960x540-v1_0_1.vgf"


def make(vgf_D=544, vgf_W=960):
    cfg = {
        "resources": [
            {"graph": {"uid": "nss", "src": VGF_REL}},
            {"tensor": {"uid": "in_t", "src": "./in_t.npy",
                        "dims": [1, vgf_D, vgf_W, 12], "format": "VK_FORMAT_R8_SINT",
                        "shader_access": "readwrite", "tiling": "LINEAR"}},
            {"tensor": {"uid": "out0", "dst": "./oracle_out0.npy",
                        "dims": [1, vgf_D // 4, vgf_W // 4, 36], "format": "VK_FORMAT_R8_SINT",
                        "shader_access": "readwrite", "tiling": "LINEAR"}},
            {"tensor": {"uid": "out1", "dst": "./oracle_out1.npy",
                        "dims": [1, vgf_D, vgf_W, 4], "format": "VK_FORMAT_R8_SINT",
                        "shader_access": "readwrite", "tiling": "LINEAR"}},
        ],
        "commands": [
            {"dispatch_graph": {"graph_ref": "nss", "implicit_barrier": False,
                                "bindings": [
                                    {"set": 0, "id": 0, "resource_ref": "in_t"},
                                    {"set": 0, "id": 1, "resource_ref": "out0"},
                                    {"set": 0, "id": 2, "resource_ref": "out1"},
                                ]}},
        ],
    }
    return cfg


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "sess")
    os.makedirs(outdir, exist_ok=True)

    if src == "--random":
        rng = np.random.default_rng(1234)
        inp = rng.integers(-128, 128, size=(1, 544, 960, 12), dtype=np.int8)
    else:
        inp = np.load(src).astype(np.int8)
        if inp.ndim == 3:
            inp = inp[None]
    assert inp.shape[-1] == 12, f"通道数应为 12，实际 {inp.shape[-1]}"

    np.save(os.path.join(outdir, "in_t.npy"), inp)
    cfg = make(inp.shape[1], inp.shape[2])

    path = os.path.join(outdir, "mini_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"输入  shape={inp.shape} 范围=[{inp.min()},{inp.max()}]")
    print(f"配置  {path}")
    print(f"数据图 {VGF_REL}")
    print("\n运行:")
    print(f"  scenario-runner --scenario {path} --output {outdir}/out "
          f"--session-memory-dump-dir {outdir}/sess --emulation-layer-profiling-dump-dir {outdir}/prof")


if __name__ == "__main__":
    main()
