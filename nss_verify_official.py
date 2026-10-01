#!/usr/bin/env python3
"""第 4 层验证：与 Arm 官方实现（ML 仿真层 + scenario-runner）对拍。

官方 side 由 scenario-runner 在启用 `ai-ml-emulation-layer-for-vulkan` 时产出：
    out_high/out_input_tensor.npy   (1,H,W,12)    数据图输入
    out_high/out_graph_0.npy        (1,H/4,W/4,36) KPN 系数
    out_high/out_graph_1.npy        (1,H,W,4)     时序反馈

用法:
    python nss_verify_official.py <official_out_dir> <mine.npz>
"""
import os
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    odir, mine_path = sys.argv[1], sys.argv[2]

    official = {
        "out0": np.squeeze(np.load(os.path.join(odir, "out_graph_0.npy"))),
        "out1": np.squeeze(np.load(os.path.join(odir, "out_graph_1.npy"))),
    }
    z = np.load(mine_path)
    mine = {k: np.squeeze(z[k]) for k in z.files if k.startswith("out")}

    print(f"官方: {odir}")
    print(f"本方: {mine_path}\n")
    rc = 0
    for k in sorted(official):
        o, m = official[k], mine.get(k)
        if m is None:
            print(f"{k}: 本方缺失")
            rc = 1
            continue
        if o.shape != m.shape:
            print(f"{k}: 形状不一致  官方={o.shape} 本方={m.shape}")
            rc = 1
            continue
        oi, mi = o.astype(np.int32), m.astype(np.int32)
        d = np.abs(oi - mi)
        nz = int(np.count_nonzero(d))
        print(f"{k}: shape={o.shape}  ({o.size:,} 元素)")
        print(f"    官方范围=[{o.min()},{o.max()}]   本方范围=[{m.min()},{m.max()}]")
        print(f"    逐元素不一致: {nz:,}/{o.size:,}  ({100.0*nz/o.size:.6f}%)")
        print(f"    最大绝对差: {int(d.max())}   平均绝对差: {d.mean():.8f}")
        if nz:
            idx = np.unravel_index(int(np.argmax(d)), d.shape)
            print(f"    最大差位置 {idx}: 官方={int(o[idx])} 本方={int(m[idx])}")
            rc = 1
        else:
            print("    **位精确一致 ✓**")
        print()
    print("结论:", "与 Arm 官方实现不一致 ✗" if rc else "与 Arm 官方实现位精确一致 ✓")
    return rc


if __name__ == "__main__":
    sys.exit(main())
