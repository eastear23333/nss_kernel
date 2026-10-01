#!/usr/bin/env python3
"""最小测试：确认 OpSDot 的打包解释是否正确。

期望（PackedVectorFormat4x8Bit，有符号）：
  a = 0x01020304 → int8 元素 [4,3,2,1]（小端，低字节是元素 0）
  b = 0x01010101 → [1,1,1,1]
  点积 = 4+3+2+1 = 10

若驱动把操作数当成「非打包的单个 32 位整数」，结果会完全不同。
"""
import ctypes
import os
import sys
from ctypes import byref, c_uint64

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nss_vk as V  # noqa: E402
from nss_vk import ST, Vk  # noqa: E402

DS_TYPE_STORAGE_BUFFER = V.DS_TYPE_STORAGE_BUFFER

HERE = os.path.dirname(os.path.abspath(__file__))

CASES_A = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x0000007F, 0x7F7F7F7F]
CASES_B = [0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101]


def expect_packed_signed(a, b):
    ea = [(a >> (8 * i)) & 0xFF for i in range(4)]
    eb = [(b >> (8 * i)) & 0xFF for i in range(4)]
    ea = [x - 256 if x > 127 else x for x in ea]
    eb = [x - 256 if x > 127 else x for x in eb]
    return sum(x * y for x, y in zip(ea, eb))


def main():
    vk = Vk()
    print(f"设备: {vk.phys_name}\n")

    N = len(CASES_A)
    a = np.array(CASES_A, dtype=np.uint32)
    b = np.array(CASES_B, dtype=np.uint32)

    ra = vk.create_buffer(a.nbytes, "a")
    rb = vk.create_buffer(b.nbytes, "b")
    rc = vk.create_buffer(N * 4, "c")
    vk.upload(ra, a)
    vk.upload(rb, b)

    dsl = vk.make_dsl(7)
    pll = vk.make_pll(dsl, 128)
    pipe = vk.make_pipeline(os.path.join(HERE, "shaders", "t3.spv"), dsl, pll)

    # 描述符池
    class Pool(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_int32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                    ("maxSets", ctypes.c_uint32), ("poolSizeCount", ctypes.c_uint32),
                    ("pPoolSizes", ctypes.c_void_p)]

    class PSize(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int32), ("descriptorCount", ctypes.c_uint32)]

    ps = PSize(DS_TYPE_STORAGE_BUFFER, 7)
    pci = Pool(ST["DS_POOL"], None, 0, 1, 1, ctypes.cast(ctypes.pointer(ps), ctypes.c_void_p))
    pool = c_uint64()
    vk.vkCreateDescriptorPool(vk.device, byref(pci), None, byref(pool))

    dsa = V.VkDescriptorSetAllocateInfo(ST["DS_ALLOC"], None, pool, 1, ctypes.pointer(dsl))
    ds = c_uint64()
    vk.vkAllocateDescriptorSets(vk.device, byref(dsa), byref(ds))

    infos, writes = [], []
    for i, res in enumerate([ra, rb, rc] + [ra, ra, ra, ra]):
        info = V.VkDescriptorBufferInfo(res["buffer"], 0, res["req_size"])
        infos.append(info)
        writes.append(V.VkWriteDescriptorSet(
            ST["WRITE_DS"], None, ds.value, i, 0, 1, DS_TYPE_STORAGE_BUFFER,
            None, ctypes.pointer(infos[-1]), None))
    arr = (V.VkWriteDescriptorSet * 7)(*writes)
    vk.vkUpdateDescriptorSets(vk.device, 7, arr, 0, None)

    # 命令缓冲
    cpci = V.VkCommandPoolCreateInfo(ST["CMD_POOL"], None, 0, vk.qfam)
    cpool = c_uint64()
    vk.vkCreateCommandPool(vk.device, byref(cpci), None, byref(cpool))
    cba = V.VkCommandBufferAllocateInfo(ST["CMD_ALLOC"], None, cpool, 0, 1)
    cb = c_uint64()
    vk.vkAllocateCommandBuffers(vk.device, byref(cba), byref(cb))
    vk.vkBeginCommandBuffer(cb, byref(V.VkCommandBufferBeginInfo(ST["CMD_BEGIN"], None, 0, None)))
    vk.vkCmdBindPipeline(cb, 1, pipe)
    dsv = c_uint64(ds.value)
    vk.vkCmdBindDescriptorSets(cb, 1, pll, 0, 1,
                               ctypes.cast(ctypes.pointer(dsv), ctypes.c_void_p), 0, None)
    vk.vkCmdDispatch(cb, 1, 1, 1)
    vk.vkEndCommandBuffer(cb)

    fci = V.VkFenceCreateInfo(ST["FENCE"], None, 0)
    fence = c_uint64()
    vk.vkCreateFence(vk.device, byref(fci), None, byref(fence))
    cbs = (c_uint64 * 1)(cb.value)
    si = V.VkSubmitInfo(ST["SUBMIT"], None, 0, None, None, 1, cbs, 0, None)
    vk.vkQueueSubmit(vk.queue, 1, byref(si), fence)
    vk.vkWaitForFences(vk.device, 1, byref(fence), 1, 0xFFFFFFFFFFFFFFFF)

    got = vk.download(rc, N * 4).view(np.int32)
    print(f"{'a':>10} {'b':>10} {'GPU':>12} {'打包期望':>12} {'非打包期望':>14}")
    ok = True
    for i in range(N):
        exp = expect_packed_signed(CASES_A[i], CASES_B[i])
        ai = CASES_A[i] - (1 << 32) if CASES_A[i] >= (1 << 31) else CASES_A[i]
        bi = CASES_B[i] - (1 << 32) if CASES_B[i] >= (1 << 31) else CASES_B[i]
        raw = (ai * bi) & 0xFFFFFFFF
        raw = raw - (1 << 32) if raw >= (1 << 31) else raw
        ok &= (int(got[i]) == exp)
        print(f"0x{CASES_A[i]:08x} 0x{CASES_B[i]:08x} {int(got[i]):>12} {exp:>12} {raw:>14}")
    print(f"\n结论: {'OpSDot 被解释为 4x8 打包 ✓' if ok else '不是打包解释 ✗'}")


if __name__ == "__main__":
    main()
