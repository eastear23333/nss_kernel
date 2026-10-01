#!/bin/bash
# MSVC 编译环境（绕开 vcvars64.bat，直接设置 PATH/INCLUDE/LIB）
# 用于在 Git Bash 里调用 cl.exe。
# 注意：INCLUDE/LIB 用正斜杠 + 分号分隔（Git Bash 下反斜杠续行会失效）。

export MSVC_ROOT="/f/VS2022BuildTools/VC/Tools/MSVC/14.44.35207"
export SDK_ROOT="/c/Program Files (x86)/Windows Kits/10"
export SDK_VER="10.0.26100.0"

export PATH="$MSVC_ROOT/bin/Hostx64/x64:$PATH"

export INCLUDE="$MSVC_ROOT/include;$SDK_ROOT/Include/$SDK_VER/ucrt;$SDK_ROOT/Include/$SDK_VER/shared;$SDK_ROOT/Include/$SDK_VER/um;$SDK_ROOT/Include/$SDK_VER/winrt"

export LIB="$MSVC_ROOT/lib/x64;$SDK_ROOT/Lib/$SDK_VER/ucrt/x64;$SDK_ROOT/Lib/$SDK_VER/um/x64"
