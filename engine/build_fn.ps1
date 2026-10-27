# Build faie2fn.dll (FAIE2 engine for Qwen3.8-Flash-Next) with nvcc 13.1 + MSVC 2022 (sm_120 = RTX 5090).
# Builds to faie2fn_new.dll first, then swaps it in, so a failed build never removes a working DLL.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe'
$cmd = "call `"$vc`" >nul && `"$nvcc`" -O3 -std=c++17 -arch=sm_120 -lineinfo -shared -Xcompiler `"/MD /O2 /EHsc`" -o `"$here\faie2fn_new.dll`" `"$here\faie2fn.cu`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Move-Item -Force "$here\faie2fn_new.dll" "$here\faie2fn.dll"
Remove-Item -ErrorAction SilentlyContinue "$here\faie2fn_new.lib", "$here\faie2fn_new.exp"
"built $here\faie2fn.dll ($([math]::Round((Get-Item "$here\faie2fn.dll").Length / 1MB, 2)) MB)"
