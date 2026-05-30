<#
.SYNOPSIS
  Build hanzo-engine GPU backends FROM SOURCE (llama.cpp/ggml) with Vulkan and/or HIP/ROCm.
  Use this if you want to compile yourself (vs setup.ps1 which fetches prebuilt). Output -> dist\<backend>.

.PREREQUISITES (Windows)
  - CMake + Ninja               (winget install Kitware.CMake Ninja-build.Ninja)
  - MSVC build tools            (already present — used to build hanzod)
  - Vulkan backend:  Vulkan SDK (winget install KhronosGroup.VulkanSDK) -> sets VULKAN_SDK, provides glslc
  - ROCm backend:    HIP SDK for Windows 7.1.1+ (gfx1151 is officially Supported) -> sets HIP_PATH
                     https://rocm.docs.amd.com/projects/install-on-windows/

.EXAMPLE
  .\build-from-source.ps1 -Backend vulkan
  .\build-from-source.ps1 -Backend rocm     # requires HIP_PATH
#>
[CmdletBinding()]
param([ValidateSet('vulkan','rocm')] [string]$Backend = 'vulkan', [string]$Ref = 'master')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$src  = Join-Path $root 'llama.cpp'
$dist = Join-Path $root "dist\$Backend"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Write-Error "CMake not found. winget install Kitware.CMake"; exit 1 }
if ($Backend -eq 'vulkan' -and -not $env:VULKAN_SDK) { Write-Error "VULKAN_SDK not set. winget install KhronosGroup.VulkanSDK"; exit 1 }
if ($Backend -eq 'rocm'   -and -not $env:HIP_PATH)    { Write-Error "HIP_PATH not set. Install HIP SDK for Windows (gfx1151 supported in 7.1.1+)"; exit 1 }

if (-not (Test-Path $src)) { git clone https://github.com/ggml-org/llama.cpp $src }
git -C $src fetch --depth 1 origin $Ref; git -C $src checkout $Ref

$flags = @('-DGGML_NATIVE=ON','-DLLAMA_CURL=OFF','-DGGML_BACKEND_DL=ON')
if ($Backend -eq 'vulkan') { $flags += '-DGGML_VULKAN=ON' }
if ($Backend -eq 'rocm')   { $flags += @('-DGGML_HIP=ON','-DAMDGPU_TARGETS=gfx1151','-DGGML_HIP_ROCWMMA_FATTN=ON') } # rocWMMA flash-attn = long-ctx win

$build = Join-Path $src "build-$Backend"
cmake -S $src -B $build -G Ninja @flags
cmake --build $build --config Release -j
New-Item -ItemType Directory -Force $dist | Out-Null
Get-ChildItem "$build\bin" -Recurse -Include *.exe,*.dll | Copy-Item -Destination $dist -Force
Write-Host "built $Backend -> $dist" -ForegroundColor Green
