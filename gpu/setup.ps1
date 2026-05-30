<#
.SYNOPSIS
  Assemble the hanzo-engine GPU backends from upstream prebuilt, optimized binaries:
    vulkan -> ggml-org/llama.cpp official Vulkan release (cross-vendor, the default)
    rocm   -> lemonade-sdk/llamacpp-rocm native-Windows ROCm-7 gfx1151 build (AMD opt-in)
  These are the same code you'd compile from source, already tuned. For a true
  from-source build (needs Vulkan SDK / HIP SDK + CMake), see build-from-source.ps1.
.EXAMPLE
  .\setup.ps1 -Backend all
  .\setup.ps1 -Backend vulkan -PullModel
#>
[CmdletBinding()]
param(
  [ValidateSet('vulkan','rocm','all')] [string]$Backend = 'all',
  [switch]$PullModel
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dist = Join-Path $root 'dist'; New-Item -ItemType Directory -Force $dist | Out-Null
$ua = @{ 'User-Agent' = 'hanzo-engine-setup' }

function Get-Release($repo, $pattern) {
  $rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers $ua -TimeoutSec 30
  $asset = $rel.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
  if (-not $asset) { throw "no asset matching '$pattern' in $repo $($rel.tag_name)" }
  return @{ tag = $rel.tag_name; url = $asset.browser_download_url; name = $asset.name }
}
function Install-Backend($name, $repo, $pattern) {
  $target = Join-Path $dist $name
  $r = Get-Release $repo $pattern
  Write-Host "[$name] $repo $($r.tag): $($r.name)" -ForegroundColor Cyan
  $zip = Join-Path $dist "$name.zip"
  Invoke-WebRequest $r.url -OutFile $zip -UseBasicParsing
  if (Test-Path $target) { Remove-Item $target -Recurse -Force }
  New-Item -ItemType Directory -Force $target | Out-Null
  Expand-Archive $zip -DestinationPath $target -Force
  # flatten if the zip has a single top dir containing llama-server.exe
  $srv = Get-ChildItem $target -Recurse -Filter llama-server.exe | Select-Object -First 1
  if ($srv -and $srv.DirectoryName -ne $target) { Get-ChildItem $srv.DirectoryName | Move-Item -Destination $target -Force }
  Remove-Item $zip -Force
  Write-Host "[$name] -> $target ($(@(Get-ChildItem $target -Filter *.dll).Count) dlls)" -ForegroundColor Green
}

if ($Backend -in 'vulkan','all') {
  # reuse the already-downloaded Vulkan build if present, else fetch
  $existing = 'C:\Users\z\work\llama-vulkan\llama-server.exe'
  $vt = Join-Path $dist 'vulkan'
  if ((Test-Path $existing) -and -not (Test-Path (Join-Path $vt 'llama-server.exe'))) {
    New-Item -ItemType Directory -Force $vt | Out-Null
    Copy-Item 'C:\Users\z\work\llama-vulkan\*' $vt -Recurse -Force
    Write-Host "[vulkan] reused C:\Users\z\work\llama-vulkan" -ForegroundColor Green
  } else {
    Install-Backend 'vulkan' 'ggml-org/llama.cpp' 'win-vulkan-x64\.zip$'
  }
}
if ($Backend -in 'rocm','all') {
  Install-Backend 'rocm' 'lemonade-sdk/llamacpp-rocm' 'windows-rocm-gfx1151-x64\.zip$'
}
if ($PullModel) {
  $mdir = Join-Path $root 'models'; New-Item -ItemType Directory -Force $mdir | Out-Null
  $f = 'Qwen3-0.6B-Q4_0.gguf'
  if (-not (Test-Path (Join-Path $mdir $f))) {
    Write-Host "[model] $f" -ForegroundColor Cyan
    Invoke-WebRequest "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/$f" -OutFile (Join-Path $mdir $f) -UseBasicParsing
  }
}
Write-Host "done. run:  .\hanzo-engine.ps1 -Backend $($Backend -replace 'all','vulkan')" -ForegroundColor Yellow
