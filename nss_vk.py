#!/usr/bin/env python3
"""NSS 的独立 Vulkan 计算运行器（DP4A 后端）。

不依赖 C++ 编译器、不依赖第三方包：用 ctypes 直连 vulkan-1.dll，
着色器由 SDK 自带的 glslangValidator 编译成 SPIR-V。

流程：解析 .vgf → 构建图 IR → 分配缓冲（四周预填 0x80 = 激活零点）
      → 逐算子录制 dispatch → 提交 → 回读 KPN / 时序两个输出。

用法:
    python nss_vk.py <model>.vgf [--h 128] [--w 128] [--seed 0] [-o out.npz]
    python nss_vk.py <model>.vgf --list-devices
"""
import argparse
import ctypes
import os
import struct
import subprocess
import sys
from ctypes import POINTER, byref, c_char_p, c_int32, c_uint32, c_uint64, c_void_p

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "nss-tools"))
from vgf_graph import analyze                                   # noqa: E402
from nss_ref import as_int8, as_int32, const_array, load_constants  # noqa: E402

GLSLANG = os.path.join(
    os.path.dirname(HERE), "neural-graphics-sdk-for-game-engines",
    "sdk", "tools", "binary_store", "glslangValidator.exe")
SHADER_DIR = os.path.join(HERE, "shaders")

# ---------------------------------------------------------------- Vulkan 常量
ST = {"DEVICE_QUEUE": 2, "DEVICE": 3, "SUBMIT": 4, "MEM_ALLOC": 5, "FENCE": 8,
      "BUFFER": 12, "SHADER_MODULE": 16, "PIPE_SHADER_STAGE": 18,
      "COMPUTE_PIPE": 29, "PIPE_LAYOUT": 30, "DSL": 32, "DS_POOL": 33,
      "DS_ALLOC": 34, "WRITE_DS": 35, "CMD_POOL": 39, "CMD_ALLOC": 40,
      "CMD_BEGIN": 42, "MEM_BARRIER": 46}
ST_DOT_FEATURES = 1000280000
VK_SUCCESS = 0
BUFFER_USAGE_STORAGE = 0x20
MEM_HOST_VISIBLE = 0x02
MEM_HOST_COHERENT = 0x04
MEM_DEVICE_LOCAL = 0x01
BUFFER_USAGE_TRANSFER_SRC = 0x01
BUFFER_USAGE_TRANSFER_DST = 0x02
DS_TYPE_STORAGE_BUFFER = 7
PIPE_BIND_COMPUTE = 1
STAGE_COMPUTE = 0x20
QUEUE_COMPUTE = 0x02
CB_LEVEL_PRIMARY = 0
ACCESS_SHADER_READ = 0x20
ACCESS_SHADER_WRITE = 0x40
STAGE_COMPUTE_BIT = 0x800
QUEUE_FAMILY_IGNORED = 0xFFFFFFFF


# ------------------------------------------------------------------ 结构定义
def _S(name, fields):
    return type(name, (ctypes.Structure,), {"_fields_": fields})


VkDeviceQueueCreateInfo = _S("VkDeviceQueueCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("queueFamilyIndex", c_uint32), ("queueCount", c_uint32), ("pQueuePriorities", POINTER(ctypes.c_float))])
VkDeviceCreateInfo = _S("VkDeviceCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("queueCreateInfoCount", c_uint32), ("pQueueCreateInfos", POINTER(VkDeviceQueueCreateInfo)),
    ("enabledLayerCount", c_uint32), ("ppEnabledLayerNames", c_void_p),
    ("enabledExtensionCount", c_uint32), ("ppEnabledExtensionNames", c_void_p),
    ("pEnabledFeatures", c_void_p)])
VkDotFeatures = _S("VkDotFeatures", [
    ("sType", c_int32), ("pNext", c_void_p), ("shaderIntegerDotProduct", c_uint32)])
VkMemoryAllocateInfo = _S("VkMemoryAllocateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("allocationSize", c_uint64), ("memoryTypeIndex", c_uint32)])
VkBufferCreateInfo = _S("VkBufferCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("size", c_uint64),
    ("usage", c_uint32), ("sharingMode", c_int32),
    ("queueFamilyIndexCount", c_uint32), ("pQueueFamilyIndices", c_void_p)])
VkMemoryRequirements = _S("VkMemoryRequirements", [
    ("size", c_uint64), ("alignment", c_uint64), ("memoryTypeBits", c_uint32)])
VkShaderModuleCreateInfo = _S("VkShaderModuleCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("codeSize", ctypes.c_size_t), ("pCode", POINTER(c_uint32))])
VkPipelineShaderStageCreateInfo = _S("VkPipelineShaderStageCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("stage", c_int32),
    ("module", c_uint64), ("pName", c_char_p), ("pSpecializationInfo", c_void_p)])
VkComputePipelineCreateInfo = _S("VkComputePipelineCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("stage", VkPipelineShaderStageCreateInfo), ("layout", c_uint64),
    ("basePipelineHandle", c_uint64), ("basePipelineIndex", c_int32)])
VkPushConstantRange = _S("VkPushConstantRange", [
    ("stageFlags", c_uint32), ("offset", c_uint32), ("size", c_uint32)])
VkPipelineLayoutCreateInfo = _S("VkPipelineLayoutCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("setLayoutCount", c_uint32), ("pSetLayouts", POINTER(c_uint64)),
    ("pushConstantRangeCount", c_uint32), ("pPushConstantRanges", POINTER(VkPushConstantRange))])
VkDescriptorSetLayoutBinding = _S("VkDescriptorSetLayoutBinding", [
    ("binding", c_uint32), ("descriptorType", c_int32), ("descriptorCount", c_uint32),
    ("stageFlags", c_uint32), ("pImmutableSamplers", c_void_p)])
VkDescriptorSetLayoutCreateInfo = _S("VkDescriptorSetLayoutCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
    ("bindingCount", c_uint32), ("pBindings", POINTER(VkDescriptorSetLayoutBinding))])
VkDescriptorPoolSize = _S("VkDescriptorPoolSize", [("type", c_int32), ("descriptorCount", c_uint32)])
VkDescriptorPoolCreateInfo = _S("VkDescriptorPoolCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("maxSets", c_uint32),
    ("poolSizeCount", c_uint32), ("pPoolSizes", POINTER(VkDescriptorPoolSize))])
VkDescriptorSetAllocateInfo = _S("VkDescriptorSetAllocateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("descriptorPool", c_uint64),
    ("descriptorSetCount", c_uint32), ("pSetLayouts", POINTER(c_uint64))])
VkDescriptorBufferInfo = _S("VkDescriptorBufferInfo", [
    ("buffer", c_uint64), ("offset", c_uint64), ("range", c_uint64)])
VkWriteDescriptorSet = _S("VkWriteDescriptorSet", [
    ("sType", c_int32), ("pNext", c_void_p), ("dstSet", c_uint64),
    ("dstBinding", c_uint32), ("dstArrayElement", c_uint32), ("descriptorCount", c_uint32),
    ("descriptorType", c_int32), ("pImageInfo", c_void_p),
    ("pBufferInfo", POINTER(VkDescriptorBufferInfo)), ("pTexelBufferView", c_void_p)])
VkCommandPoolCreateInfo = _S("VkCommandPoolCreateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("queueFamilyIndex", c_uint32)])
VkCommandBufferAllocateInfo = _S("VkCommandBufferAllocateInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("commandPool", c_uint64),
    ("level", c_int32), ("commandBufferCount", c_uint32)])
VkCommandBufferBeginInfo = _S("VkCommandBufferBeginInfo", [
    ("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("pInheritanceInfo", c_void_p)])
VkSubmitInfo = _S("VkSubmitInfo", [
    ("sType", c_int32), ("pNext", c_void_p),
    ("waitSemaphoreCount", c_uint32), ("pWaitSemaphores", c_void_p), ("pWaitDstStageMask", c_void_p),
    ("commandBufferCount", c_uint32), ("pCommandBuffers", POINTER(c_uint64)),
    ("signalSemaphoreCount", c_uint32), ("pSignalSemaphores", c_void_p)])
VkFenceCreateInfo = _S("VkFenceCreateInfo", [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32)])
VkMemoryBarrier = _S("VkMemoryBarrier", [
    ("sType", c_int32), ("pNext", c_void_p), ("srcAccessMask", c_uint32), ("dstAccessMask", c_uint32)])


class Vk:
    def __init__(self, device_index=None):
        self.lib = ctypes.WinDLL("vulkan-1.dll")
        self._inst_procs()
        self.instance = self._create_instance()
        self.phys, self.phys_name = self._pick_device(device_index)
        self.device, self.queue, self.qfam = self._create_device()

    def _inst_procs(self):
        L = self.lib
        for n, a, r in [
            ("vkCreateInstance", [c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkEnumeratePhysicalDevices", [c_uint64, POINTER(c_uint32), POINTER(c_uint64)], c_int32),
            ("vkGetPhysicalDeviceQueueFamilyProperties", [c_uint64, POINTER(c_uint32), c_void_p], None),
            ("vkGetPhysicalDeviceProperties", [c_uint64, c_void_p], None),
            ("vkGetPhysicalDeviceMemoryProperties", [c_uint64, c_void_p], None),
            ("vkCreateDevice", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkGetDeviceQueue", [c_uint64, c_uint32, c_uint32, POINTER(c_uint64)], None),
            ("vkGetDeviceProcAddr", [c_uint64, c_char_p], c_void_p),
        ]:
            f = getattr(L, n)
            f.argtypes, f.restype = a, r

    def _create_instance(self):
        class AppInfo(ctypes.Structure):
            _fields_ = [("sType", c_int32), ("pNext", c_void_p), ("app", c_char_p), ("av", c_uint32),
                        ("eng", c_char_p), ("ev", c_uint32), ("api", c_uint32)]

        class InstCI(ctypes.Structure):
            _fields_ = [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("pAppInfo", POINTER(AppInfo)), ("lc", c_uint32), ("pl", c_void_p),
                        ("ec", c_uint32), ("pe", c_void_p)]

        app = AppInfo(0, None, b"nss-dp4a", 1, b"ng", 1, 0x00400000)
        ci = InstCI(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
        inst = c_uint64()
        r = self.lib.vkCreateInstance(byref(ci), None, byref(inst))
        if r != VK_SUCCESS:
            raise RuntimeError(f"vkCreateInstance 失败 {r}")
        return inst

    def _pick_device(self, index):
        n = c_uint32()
        self.lib.vkEnumeratePhysicalDevices(self.instance, byref(n), None)
        devs = (c_uint64 * n.value)()
        self.lib.vkEnumeratePhysicalDevices(self.instance, byref(n), devs)
        if index is None:
            # 默认挑 NVIDIA（本机 DP4A 可用的那块）
            for d in devs:
                buf = (ctypes.c_byte * 1024)()
                self.lib.vkGetPhysicalDeviceProperties(d, buf)
                name = ctypes.string_at(ctypes.addressof(buf) + 20).decode("utf-8", "replace")
                if "NVIDIA" in name:
                    return d, name
            index = 0
        buf = (ctypes.c_byte * 1024)()
        self.lib.vkGetPhysicalDeviceProperties(devs[index], buf)
        return devs[index], ctypes.string_at(ctypes.addressof(buf) + 20).decode("utf-8", "replace")

    def _create_device(self):
        cnt = c_uint32()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self.phys, byref(cnt), None)
        props = (ctypes.c_uint32 * (cnt.value * 12))()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self.phys, byref(cnt), props)
        qfam = next((i for i in range(cnt.value) if props[i * 12] & QUEUE_COMPUTE), 0)

        dot = VkDotFeatures(ST_DOT_FEATURES, None, 1)

        # 若设备暴露 VK_KHR_cooperative_matrix，就一并启用扩展 + 特性（Tensor Core 路径需要）。
        # 不支持时保持原样，不影响其它 kernel。
        class _ExtProps(ctypes.Structure):
            _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", c_uint32)]

        self.lib.vkEnumerateDeviceExtensionProperties.argtypes = [
            c_uint64, c_char_p, POINTER(c_uint32), c_void_p]
        self.lib.vkEnumerateDeviceExtensionProperties.restype = c_int32
        en = c_uint32()
        self.lib.vkEnumerateDeviceExtensionProperties(self.phys, None, byref(en), None)
        earr = (_ExtProps * en.value)()
        self.lib.vkEnumerateDeviceExtensionProperties(self.phys, None, byref(en), earr)
        have_cm = any(earr[i].extensionName == b"VK_KHR_cooperative_matrix"
                      for i in range(en.value))

        pnext = ctypes.cast(byref(dot), c_void_p)
        ext_names = None
        if have_cm:
            class _CmFeatures(ctypes.Structure):
                _fields_ = [("sType", c_int32), ("pNext", c_void_p),
                            ("cooperativeMatrix", c_uint32),
                            ("robustBufferAccess", c_uint32)]
            # VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
            # 官方 vk 头定义 = 1000506000（曾误用 1000507000）
            cmf = _CmFeatures(1000506000, pnext, 1, 0)   # VK_TRUE
            pnext = ctypes.cast(byref(cmf), c_void_p)
            ext_names = (ctypes.c_char_p * 1)(b"VK_KHR_cooperative_matrix")

        prio = (ctypes.c_float * 1)(1.0)
        qci = VkDeviceQueueCreateInfo(ST["DEVICE_QUEUE"], None, 0, qfam, 1, prio)
        dci = VkDeviceCreateInfo(ST["DEVICE"], pnext, 0, 1,
                                 ctypes.pointer(qci), 0, None,
                                 1 if have_cm else 0,
                                 ctypes.cast(ext_names, c_void_p) if have_cm else None,
                                 None)
        if have_cm:
            print("[nss_vk] 已启用 VK_KHR_cooperative_matrix")
        dev = c_uint64()
        r = self.lib.vkCreateDevice(self.phys, byref(dci), None, byref(dev))
        if r != VK_SUCCESS:
            raise RuntimeError(f"vkCreateDevice 失败 {r}")
        q = c_uint64()
        self.lib.vkGetDeviceQueue(dev, qfam, 0, byref(q))
        self.device = dev
        self._dev_procs()
        return dev, q, qfam

    def _dev_procs(self):
        p = self.lib.vkGetDeviceProcAddr
        for n, a, r in [
            ("vkCreateBuffer", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkGetBufferMemoryRequirements", [c_uint64, c_uint64, c_void_p], None),
            ("vkAllocateMemory", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkBindBufferMemory", [c_uint64, c_uint64, c_uint64, c_uint64], c_int32),
            ("vkMapMemory", [c_uint64, c_uint64, c_uint64, c_uint64, c_uint32, c_void_p], c_int32),
            ("vkUnmapMemory", [c_uint64, c_uint64], None),
            ("vkCreateShaderModule", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkCreateDescriptorSetLayout", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkCreatePipelineLayout", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkCreateComputePipelines", [c_uint64, c_uint64, c_uint32, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkCreateDescriptorPool", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkAllocateDescriptorSets", [c_uint64, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkUpdateDescriptorSets", [c_uint64, c_uint32, c_void_p, c_uint32, c_void_p], None),
            ("vkCreateCommandPool", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkAllocateCommandBuffers", [c_uint64, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkResetCommandPool", [c_uint64, c_uint64, c_uint32], c_int32),
            ("vkBeginCommandBuffer", [c_uint64, c_void_p], c_int32),
            ("vkEndCommandBuffer", [c_uint64], c_int32),
            ("vkCmdBindPipeline", [c_uint64, c_int32, c_uint64], None),
            ("vkCmdBindDescriptorSets", [c_uint64, c_int32, c_uint64, c_uint32, c_uint32, c_void_p, c_uint32, c_void_p], None),
            ("vkCmdPushConstants", [c_uint64, c_uint64, c_uint32, c_uint32, c_uint32, c_void_p], None),
            ("vkCmdDispatch", [c_uint64, c_uint32, c_uint32, c_uint32], None),
            ("vkCmdCopyBuffer", [c_uint64, c_uint64, c_uint64, c_uint32, c_void_p], None),
            ("vkCmdPipelineBarrier", [c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_void_p, c_uint32, c_void_p, c_uint32, c_void_p], None),
            ("vkCreateFence", [c_uint64, c_void_p, c_void_p, POINTER(c_uint64)], c_int32),
            ("vkQueueSubmit", [c_uint64, c_uint32, c_void_p, c_uint64], c_int32),
            ("vkWaitForFences", [c_uint64, c_uint32, c_void_p, c_uint32, c_uint64], c_int32),
            ("vkQueueWaitIdle", [c_uint64], c_int32),
            ("vkDeviceWaitIdle", [c_uint64], c_int32),
        ]:
            fp = p(self.device, n.encode())
            if not fp:
                raise RuntimeError(f"取不到设备函数 {n}")
            f = ctypes.CFUNCTYPE(r, *a)(fp)
            setattr(self, n, f)

    # ------------------------------------------------------------ 资源
    def create_buffer(self, size, name="", device_local=False):
        size = max(size, 4)
        usage = BUFFER_USAGE_STORAGE
        if device_local:
            usage |= BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST
        bci = VkBufferCreateInfo(ST["BUFFER"], None, 0, size,
                                 usage, 0, 0, None)
        buf = c_uint64()
        if self.vkCreateBuffer(self.device, byref(bci), None, byref(buf)) != VK_SUCCESS:
            raise RuntimeError(f"创建缓冲失败 {name}")
        req = VkMemoryRequirements()
        self.vkGetBufferMemoryRequirements(self.device, buf, byref(req))

        mp = self._mem_props()
        if device_local:
            mt = next(i for i in range(mp.memoryTypeCount)
                      if (mp.memoryTypes[i * 2] & 0x1) == MEM_DEVICE_LOCAL)
        else:
            mt = next(i for i in range(mp.memoryTypeCount)
                      if (mp.memoryTypes[i * 2] & 0x7) == (MEM_HOST_VISIBLE | MEM_HOST_COHERENT))
        mai = VkMemoryAllocateInfo(ST["MEM_ALLOC"], None, req.size, mt)
        mem = c_uint64()
        if self.vkAllocateMemory(self.device, byref(mai), None, byref(mem)) != VK_SUCCESS:
            raise RuntimeError(f"分配内存失败 {name}")
        self.vkBindBufferMemory(self.device, buf, mem, 0)
        ptr = c_void_p()
        if not device_local:
            self.vkMapMemory(self.device, mem, 0, req.size, 0, byref(ptr))
        return {"buffer": buf, "memory": mem, "size": req.size, "ptr": ptr,
                "req_size": size, "device_local": device_local}

    def _mem_props(self):
        if getattr(self, "_mp_cache", None) is None:
            class MemProps(ctypes.Structure):
                # VkMemoryType = {propertyFlags(u32), heapIndex(u32)}，故每个类型占 2 个 u32
                _fields_ = [("memoryTypeCount", c_uint32), ("memoryTypes", c_uint32 * 64),
                            ("memoryHeapCount", c_uint32), ("memoryHeaps", c_uint64 * 32)]

            mp = MemProps()
            self.lib.vkGetPhysicalDeviceMemoryProperties(self.phys, byref(mp))
            self._mp_cache = mp
        return self._mp_cache

    def upload(self, res, data):
        """上传到映射内存；若目标是设备本地缓冲，则经 staging 拷贝。"""
        if res.get("device_local"):
            staging = self.create_buffer(data.nbytes, "staging")
            ctypes.memmove(staging["ptr"], data.ctypes.data, data.nbytes)
            self.copy_buffer(staging, res, data.nbytes)
            return
        ctypes.memmove(res["ptr"], data.ctypes.data, data.nbytes)

    def download(self, res, nbytes):
        if res.get("device_local"):
            staging = self.create_buffer(nbytes, "staging-dl")
            self.copy_buffer(res, staging, nbytes)
            out = np.empty(nbytes, dtype=np.uint8)
            ctypes.memmove(out.ctypes.data, staging["ptr"], nbytes)
            return out
        out = np.empty(nbytes, dtype=np.uint8)
        ctypes.memmove(out.ctypes.data, res["ptr"], nbytes)
        return out

    def copy_buffer(self, src, dst, nbytes):
        """一次性命令缓冲做 vkCmdCopyBuffer（staging 往返）。"""
        if getattr(self, "_copy_pool", None) is None:
            pool_ci = VkCommandPoolCreateInfo(ST["CMD_POOL"], None, 0x2, self.qfam)  # RESET
            pool = c_uint64()
            if self.vkCreateCommandPool(self.device, byref(pool_ci), None, byref(pool)) != VK_SUCCESS:
                raise RuntimeError("创建 staging 命令池失败")
            self._copy_pool = pool
        cb = c_uint64()
        ai = VkCommandBufferAllocateInfo(ST["CMD_ALLOC"], None, self._copy_pool,
                                         CB_LEVEL_PRIMARY, 1)
        if self.vkAllocateCommandBuffers(self.device, byref(ai), byref(cb)) != VK_SUCCESS:
            raise RuntimeError("分配 staging 命令缓冲失败")

        class VkCommandBufferBeginInfo(ctypes.Structure):
            _fields_ = [("sType", c_int32), ("pNext", c_void_p),
                        ("flags", c_uint32), ("pInheritanceInfo", c_void_p)]

        class VkBufferCopy(ctypes.Structure):
            _fields_ = [("srcOffset", c_uint64), ("dstOffset", c_uint64), ("size", c_uint64)]

        bi = VkCommandBufferBeginInfo(ST["CMD_BEGIN"], None, 0x1, None)  # ONETIME
        self.vkBeginCommandBuffer(cb, byref(bi))
        region = VkBufferCopy(0, 0, nbytes)
        self.vkCmdCopyBuffer(cb, src["buffer"], dst["buffer"], 1, byref(region))
        self.vkEndCommandBuffer(cb)

        fence_ci = VkFenceCreateInfo(ST["FENCE"], None, 0)
        fence = c_uint64()
        self.vkCreateFence(self.device, byref(fence_ci), None, byref(fence))
        cbs = (c_uint64 * 1)(cb.value)
        si = VkSubmitInfo(ST["SUBMIT"], None, 0, None, None, 1, cbs, 0, None)
        self.vkQueueSubmit(self.queue, 1, byref(si), fence)
        self.vkWaitForFences(self.device, 1, byref(fence), 1, 0xFFFFFFFFFFFFFFFF)
        self.vkResetCommandPool(self.device, self._copy_pool, 0)

    # ------------------------------------------------------------ 管线
    def make_pipeline(self, spv_path, dsl, pll):
        code = open(spv_path, "rb").read()
        arr = (c_uint32 * (len(code) // 4)).from_buffer_copy(code)
        smci = VkShaderModuleCreateInfo(ST["SHADER_MODULE"], None, 0, len(code), arr)
        mod = c_uint64()
        if self.vkCreateShaderModule(self.device, byref(smci), None, byref(mod)) != VK_SUCCESS:
            raise RuntimeError(f"创建 shader module 失败 {spv_path}")
        stage = VkPipelineShaderStageCreateInfo(ST["PIPE_SHADER_STAGE"], None, 0,
                                                STAGE_COMPUTE, mod, b"main", None)
        cpci = VkComputePipelineCreateInfo(ST["COMPUTE_PIPE"], None, 0, stage, pll, 0, -1)
        pipe = c_uint64()
        r = self.vkCreateComputePipelines(self.device, 0, 1, byref(cpci), None, byref(pipe))
        if r != VK_SUCCESS:
            raise RuntimeError(f"创建计算管线失败 {spv_path} (VkResult={r})")
        return pipe

    def make_dsl(self, nbindings):
        bindings = (VkDescriptorSetLayoutBinding * nbindings)()
        for i in range(nbindings):
            bindings[i] = VkDescriptorSetLayoutBinding(i, DS_TYPE_STORAGE_BUFFER, 1, STAGE_COMPUTE, None)
        ci = VkDescriptorSetLayoutCreateInfo(ST["DSL"], None, 0, nbindings, bindings)
        dsl = c_uint64()
        self.vkCreateDescriptorSetLayout(self.device, byref(ci), None, byref(dsl))
        return dsl

    def make_pll(self, dsl, push_size):
        pcr = VkPushConstantRange(STAGE_COMPUTE, 0, push_size)
        ci = VkPipelineLayoutCreateInfo(ST["PIPE_LAYOUT"], None, 0, 1,
                                        ctypes.pointer(dsl), 1, ctypes.pointer(pcr))
        pll = c_uint64()
        self.vkCreatePipelineLayout(self.device, byref(ci), None, byref(pll))
        return pll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", help="从 .npy 读图输入（(1,H,W,12) int8），替代随机输入")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("-o", "--output")
    ap.add_argument("--infer-only", action="store_true", help="只打印规划好的 dispatch 列表")
    ap.add_argument("--dbg-handles", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="重复提交以测平均耗时")
    ap.add_argument("--device-local", action="store_true", default=True,
                    help="中间张量与权重放 DEVICE_LOCAL（显存），经 staging 往返（默认开）")
    ap.add_argument("--host-memory", dest="device_local", action="store_false",
                    help="改用 HOST_VISIBLE|HOST_COHERENT 系统内存（仅用于对照）")
    ap.add_argument("--debug-op", type=int, default=-1,
                    help="对指定 conv 算子把原始 int32 累加器导出到 dump_dbg.npy")
    ap.add_argument("--dump-all", metavar="DIR", help="导出所有中间张量用于逐层对拍")
    args = ap.parse_args()

    vk = Vk()
    print(f"设备: {vk.phys_name}")

    ir = analyze(args.vgf)
    print(f"模型: {os.path.basename(args.vgf)}  算子 {len(ir['ops'])} 个")

    if args.infer_only:
        return

    consts = load_constants(args.vgf, ir)
    H, W = args.h, args.w
    inp = None
    if args.input:
        src = np.load(args.input)
        inp = np.ascontiguousarray(src[0] if src.ndim == 4 else src, dtype=np.int8)
        if inp.shape[2] != 12:
            raise SystemExit(f"输入通道数应为 12，实际 {inp.shape[2]}")
        H, W = inp.shape[0], inp.shape[1]          # 必须在使用形状之前确定
    print(f"输入: 1×{H}×{W}×12" + ("  (来自文件)" if inp is not None else "  (随机)"))
    if H % 8 or W % 8:
        print("  !! 警告: 输入尺寸必须是 8 的倍数（NSS 的约束）")

    # ---- 布局规划：每个中间张量一个缓冲，四周留 1 像素边框（预填 0x80） ----
    N_BIND = 8
    dsl = vk.make_dsl(N_BIND)
    pll = vk.make_pll(dsl, 128)
    pipes = {
        "conv": vk.make_pipeline(os.path.join(SHADER_DIR, "conv_rq.spv"), dsl, pll),
        "resize": vk.make_pipeline(os.path.join(SHADER_DIR, "resize2x.spv"), dsl, pll),
        "concat": vk.make_pipeline(os.path.join(SHADER_DIR, "concat_copy.spv"), dsl, pll),
    }

    class Pool(ctypes.Structure):
        _fields_ = [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
                    ("maxSets", c_uint32), ("poolSizeCount", c_uint32),
                    ("pPoolSizes", POINTER(VkDescriptorPoolSize))]

    max_sets = len(ir["ops"]) + 4
    ps = VkDescriptorPoolSize(DS_TYPE_STORAGE_BUFFER, max_sets * N_BIND)
    pci = Pool(ST["DS_POOL"], None, 0, max_sets, 1, ctypes.pointer(ps))
    pool = c_uint64()
    if vk.vkCreateDescriptorPool(vk.device, byref(pci), None, byref(pool)) != VK_SUCCESS:
        raise RuntimeError("创建描述符池失败")

    dummy = vk.create_buffer(256, "dummy")
    vk.upload(dummy, np.zeros(64, dtype=np.uint32))
    json_dummy = vk.create_buffer(16, "jsondummy")
    empty = vk.create_buffer(16, "empty")
    vk.upload(empty, np.zeros(4, dtype=np.uint32))

    print(f"描述符池 maxSets={max_sets} dsl 绑定数={N_BIND}")
    print(f"管线就绪: {list(pipes)}\n")

    # ---------------------------------------------------------------- 常量缓冲
    def up_i32(arr, name):
        a = np.ascontiguousarray(arr, dtype=np.int32)
        r = vk.create_buffer(max(a.nbytes, 4), name, device_local=args.device_local)
        vk.upload(r, a)
        return r

    def up_u32(arr, name):
        a = np.ascontiguousarray(arr, dtype=np.uint32)
        r = vk.create_buffer(max(a.nbytes, 4), name, device_local=args.device_local)
        vk.upload(r, a)
        return r

    # -------------------------------------------------- 融合：conv+rescale(+table)
    ops = ir["ops"]
    fused = {}          # conv idx -> (rescale idx, table idx or None)
    skip = set()
    for r in ops:
        if r["kind"] != "CONV2D":
            continue
        j = next((s["index"] for s in ops
                  if s["kind"] == "RESCALE" and s["input"]["kind"] == "node"
                  and s["input"]["index"] == r["index"]), None)
        k = None
        if j is not None:
            k = next((t["index"] for t in ops
                      if t["kind"] == "TABLE" and t["input"]["kind"] == "node"
                      and t["input"]["index"] == j), None)
        fused[r["index"]] = (j, k)
        if j is not None:
            skip.add(j)
        if k is not None:
            skip.add(k)
    n_disp = len(ops) - len(skip)
    print(f"融合: {len(ops)} 算子 -> {n_disp} 次 dispatch "
          f"(conv+rescale{' +table' if any(k for _, k in fused.values()) else ''})\n")

    # ------------------------------------------------------------ 形状与缓冲
    shapes = {}         # op index -> (h, w, c)
    bufs = {}
    for r in ops:
        i = r["index"]
        if r["kind"] == "CONV2D":
            ish = shapes[r["input"]["index"]] if r["input"]["kind"] == "node" else (H, W, 12)
            _, kh, kw, cin = r["weight"]["shape"]
            pt, pb, pl, pr = r["pad"]
            sh, sw = r["stride"]
            shapes[i] = ((ish[0] + pt + pb - kh) // sh + 1,
                         (ish[1] + pl + pr - kw) // sw + 1,
                         r["weight"]["shape"][0])
        elif r["kind"] in ("RESCALE", "TABLE"):
            shapes[i] = shapes[r["input"]["index"]]
        elif r["kind"] == "RESIZE":
            h, w, c = shapes[r["input"]["index"]]
            sc = r["scale"]
            shapes[i] = (h * sc[0] // sc[1], w * sc[2] // sc[3], c)
        elif r["kind"] == "CONCAT":
            parts = [shapes[t["index"]] for t in r["inputs"]]
            shapes[i] = (parts[0][0], parts[0][1], sum(p[2] for p in parts))

    def alloc_padded(h, w, c, name):
        words = (h + 2) * (w + 2) * (c // 4)
        r = vk.create_buffer(words * 4, name, device_local=args.device_local)
        vk.upload(r, np.full(words, 0x80808080, dtype=np.uint32))   # 边框 = 激活零点
        return r

    # 图输入
    # 图输入：直接构造好带边框的数组再整体上传（映射内存是不可写的 numpy 视图）
    if inp is None:
        inp = np.random.default_rng(args.seed).integers(-128, 128, size=(H, W, 12), dtype=np.int8)
    in_buf = alloc_padded(H, W, 12, "input")
    pad = np.full((H + 2, W + 2, 12), -128, dtype=np.int8)
    pad[1:H + 1, 1:W + 1, :] = inp
    vk.upload(in_buf, np.ascontiguousarray(pad.view(np.uint32).reshape(-1)))

    result_buf = {}
    for r in ops:                      # 所有算子都分配缓冲（被融合的虽不单独 dispatch 但作为写入目标）
        i = r["index"]
        h, w, c = shapes[i]
        bufs[i] = alloc_padded(h, w, c, f"op{i}")

    for i, (j, k) in fused.items():
        tgt = k if k is not None else (j if j is not None else i)
        result_buf[i] = bufs[tgt]
        if j is not None:
            result_buf[j] = bufs[tgt]
        if k is not None:
            result_buf[k] = bufs[tgt]
    for r in ops:
        i = r["index"]
        if i not in result_buf:
            result_buf[i] = bufs[i]
    # 注意：不要往 result_buf 里登记图输入张量 —— NSS 的图输入张量 id 恰好是 11，
    # 而 11 同时又是 conv5 的 rescale 下标，会被覆盖掉融合映射。
    # 卷积内核读输入时已经按 kind=="input" 直接取 in_buf，不需要这张表。

    # ------------------------------------------------------------ 录制命令
    cmd_pool_ci = VkCommandPoolCreateInfo(ST["CMD_POOL"], None, 0, vk.qfam)
    cmd_pool = c_uint64()
    vk.vkCreateCommandPool(vk.device, byref(cmd_pool_ci), None, byref(cmd_pool))
    cb_ai = VkCommandBufferAllocateInfo(ST["CMD_ALLOC"], None, cmd_pool, CB_LEVEL_PRIMARY, 1)
    cb = c_uint64()
    vk.vkAllocateCommandBuffers(vk.device, byref(cb_ai), byref(cb))
    cbi = VkCommandBufferBeginInfo(ST["CMD_BEGIN"], None, 0, None)
    vk.vkBeginCommandBuffer(cb, byref(cbi))

    def bind_and_dispatch(pipe, bindings, push, gx, gy, gz):
        ds = c_uint64()
        ds_ai = VkDescriptorSetAllocateInfo(ST["DS_ALLOC"], None, pool, 1, ctypes.pointer(dsl))
        if vk.vkAllocateDescriptorSets(vk.device, byref(ds_ai), byref(ds)) != VK_SUCCESS:
            raise RuntimeError("分配描述符集失败")
        infos, writes = [], []
        for b in range(N_BIND):
            res = bindings.get(b, dummy)
            info = VkDescriptorBufferInfo(res["buffer"], 0, res["req_size"])
            infos.append(info)
            w = VkWriteDescriptorSet(ST["WRITE_DS"], None, ds.value, b, 0, 1,
                                     DS_TYPE_STORAGE_BUFFER, None, ctypes.pointer(infos[-1]), None)
            writes.append(w)
        arr = (VkWriteDescriptorSet * len(writes))(*writes)
        vk.vkUpdateDescriptorSets(vk.device, len(writes), arr, 0, None)
        vk.vkCmdBindPipeline(cb, PIPE_BIND_COMPUTE, pipe)
        dsv = c_uint64(ds.value)
        vk.vkCmdBindDescriptorSets(cb, PIPE_BIND_COMPUTE, pll, 0, 1,
                                   ctypes.cast(ctypes.pointer(dsv), c_void_p), 0, None)
        if push:
            pv = (ctypes.c_uint32 * len(push))(*push)
            vk.vkCmdPushConstants(cb, pll, STAGE_COMPUTE, 0, len(push) * 4, pv)
        vk.vkCmdDispatch(cb, (gx + 7) // 8, (gy + 7) // 8, gz)

    def barrier():
        mb = VkMemoryBarrier(ST["MEM_BARRIER"], None, ACCESS_SHADER_WRITE,
                             ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        vk.vkCmdPipelineBarrier(cb, STAGE_COMPUTE_BIT, STAGE_COMPUTE_BIT, 0,
                                1, byref(mb), 0, None, 0, None)

    stats = {"conv": 0, "resize": 0, "concat": 0}
    dbg_res = None
    for r in ops:
        i = r["index"]
        if i in skip:
            continue
        if r["kind"] == "CONV2D":
            j, k = fused[i]
            if j is None:
                raise RuntimeError(f"conv {i} 没有可融合的 rescale")
            rs = ops[j]
            wgt = const_array(r["weight"], consts)
            bias = const_array(r["bias"], consts)
            izp = r["input_zero_point"]
            izp = as_int8(izp[0] if isinstance(izp, list) else izp)
            wzp = r["weight_zero_point"]
            wzp = as_int8(wzp[0] if isinstance(wzp, list) else wzp)
            corr = (-np.int64(izp) * wgt.astype(np.int64).sum(axis=(1, 2, 3))).astype(np.int64)
            bias_i64 = np.asarray(bias, dtype=np.int64).reshape(-1)
            if bias_i64.size == 1:
                bias_i64 = np.repeat(bias_i64, wgt.shape[0])
            bc = (bias_i64 + corr).astype(np.int32)

            wpack = np.ascontiguousarray(wgt).view(np.uint32).reshape(-1)
            mult = np.ascontiguousarray(const_array(rs["multiplier"], consts),
                                        dtype=np.int32).reshape(-1)
            shv = np.ascontiguousarray(const_array(rs["shift"], consts),
                                       dtype=np.int32).reshape(-1)
            ozp = rs["output_zero_point"]["value"]
            ozp = as_int8(ozp[0] if isinstance(ozp, list) else ozp)
            lut = (np.ascontiguousarray(const_array(ops[k]["table"], consts),
                                        dtype=np.int32).reshape(-1)
                   if k is not None else np.zeros(256, dtype=np.int32))

            h, w, c = shapes[i]
            pt, pb, pl, pr = r["pad"]
            sh, sw = r["stride"]
            push = [shapes[r["input"]["index"]][1] if r["input"]["kind"] == "node" else W,
                    wgt.shape[3] // 4, h, w, c // 4, pt, pl, sh, sw, ozp & 0xFFFFFFFF,
                    wgt.shape[1], wgt.shape[2], 1 if k is not None else 0,
                    1 if i == args.debug_op else 0]
            dbg = vk.create_buffer(max(h * w * c * 4, 4), f"dbg{i}")
            if i == args.debug_op:
                dbg_res = dbg
            bindings = {0: result_buf[r["input"]["index"]] if r["input"]["kind"] == "node" else in_buf,
                        1: up_u32(wpack, f"w{i}"), 2: up_i32(bc, f"bc{i}"),
                        3: up_i32(mult, f"m{i}"), 4: up_i32(shv, f"s{i}"),
                        5: up_i32(lut, f"l{i}"), 6: result_buf[i], 7: dbg}
            bind_and_dispatch(pipes["conv"], bindings, push, w, h, c // 4)
            stats["conv"] += 1
            barrier()

        elif r["kind"] == "RESIZE":
            h, w, c = shapes[r["input"]["index"]]
            oh, ow, _ = shapes[i]
            push = [h, w, c // 4, oh, ow]
            bindings = {0: result_buf[r["input"]["index"]], 6: result_buf[i]}
            bind_and_dispatch(pipes["resize"], bindings, push, ow, oh, c // 4)
            stats["resize"] += 1
            barrier()

        elif r["kind"] == "CONCAT":
            parts = r["inputs"]
            h, w, _ = shapes[parts[0]["index"]]
            _, _, dc = shapes[i]
            off = 0
            for t in parts:
                _, _, sc = shapes[t["index"]]
                push = [h, w, sc // 4, dc // 4, off // 4]
                bindings = {0: result_buf[t["index"]], 6: result_buf[i]}
                bind_and_dispatch(pipes["concat"], bindings, push, w, h, sc // 4)
                off += sc
            stats["concat"] += 1
            barrier()

    vk.vkEndCommandBuffer(cb)

    fence_ci = VkFenceCreateInfo(ST["FENCE"], None, 0)
    fence = c_uint64()
    vk.vkCreateFence(vk.device, byref(fence_ci), None, byref(fence))
    cbs = (c_uint64 * 1)(cb.value)
    si = VkSubmitInfo(ST["SUBMIT"], None, 0, None, None, 1, cbs, 0, None)

    vk.vkQueueSubmit(vk.queue, 1, byref(si), fence)
    r = vk.vkWaitForFences(vk.device, 1, byref(fence), 1, 0xFFFFFFFFFFFFFFFF)
    print(f"提交完成 (VkResult={r})  dispatch: {stats}")

    if args.repeat > 1:
        import time
        vk.vkQueueWaitIdle(vk.queue)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            vk.vkQueueSubmit(vk.queue, 1, byref(si), 0)
        vk.vkQueueWaitIdle(vk.queue)
        dt = (time.perf_counter() - t0) / args.repeat
        print(f"性能: {dt * 1000:.3f} ms/帧  ({1.0 / dt:.1f} FPS)  [{args.repeat} 次平均]")
    print()

    # ---------------------------------------------------------------- 回读
    outs = []
    for o in ir["outputs"]:
        tid = o["tensor_id"]
        idx = next(r2["index"] for r2 in ops if r2["result"] == tid)
        h, w, c = shapes[idx]
        raw = vk.download(result_buf[idx], (h + 2) * (w + 2) * c)
        arr = raw.reshape(h + 2, w + 2, c)[1:h + 1, 1:w + 1, :].astype(np.int8)
        outs.append(arr)
        print(f"  输出 tensor={tid} shape={arr.shape} range=[{arr.min()}, {arr.max()}]")

    if args.debug_op >= 0:
        h, w, c = shapes[args.debug_op]
        n = h * w * 24
        raw = vk.download(dbg_res, n * 4)
        arr = raw.view(np.int32).reshape(h, w, 24)
        np.save("dump_dbg.npy", arr)
        print(f"已导出 conv{args.debug_op} 的原始累加器到 dump_dbg.npy  "
              f"shape={arr.shape} range=[{arr.min()}, {arr.max()}]")

    if args.dbg_handles:
        print("\n缓冲句柄检查:")
        for i in sorted(set(list(range(len(ops))))):
            rb = result_buf.get(i)
            bb = bufs.get(i)
            print(f"  op{i:>2}  result_buf={hex(rb['buffer'].value) if rb else 'None':<20}"
                  f" bufs={hex(bb['buffer'].value) if bb else 'None':<20}"
                  f" {'同' if rb and bb and rb['buffer'].value == bb['buffer'].value else '**不同**'}")

    if args.dump_all:
        os.makedirs(args.dump_all, exist_ok=True)
        for r in ops:
            i = r["index"]
            h, w, c = shapes[i]
            raw = vk.download(result_buf[i], (h + 2) * (w + 2) * c)
            arr = raw.reshape(h + 2, w + 2, c)[1:h + 1, 1:w + 1, :]
            np.save(os.path.join(args.dump_all, f"op{i:02d}_{r['kind']}.npy"), arr)
        print(f"已导出 {len(ops)} 个中间张量到 {args.dump_all}/")

    if args.output:
        np.savez(args.output, in0=inp, **{f"out{i}": v for i, v in enumerate(outs)})
        print(f"\n已写出 {args.output}  （含输入 in0，供参考实现对拍同一份数据）")
    return outs


if __name__ == "__main__":
    main()
