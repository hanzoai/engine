<#
.SYNOPSIS
  hanzo-engine (GPU) — native, bridge-free GPU inference for Windows on the
  llama.cpp/ggml runtime, with selectable Vulkan or ROCm backend. Serves the
  OpenAI-compatible API (/v1/chat/completions, /v1/embeddings) that the hanzo
  node already speaks — a drop-in for the WSL mistral.rs engine, ~250x faster.

.EXAMPLE
  .\hanzo-engine.ps1 -Backend vulkan -Model models\Qwen3-0.6B-Q4_0.gguf -Port 36920
  .\hanzo-engine.ps1 -Backend rocm   -Model models\Qwen3-0.6B-Q4_0.gguf -Port 36920 -Embedding
#>
[CmdletBinding()]
param(
  [ValidateSet('vulkan','rocm')] [string]$Backend = 'vulkan',
  [string]$Model = '',
  [int]$Port = 36920,
  [int]$Ctx = 8192,
  [int]$Ngl = 99,
  [switch]$Embedding,            # also expose /v1/embeddings (pooling mode)
  [string]$Alias = 'hanzo'
)
$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$dist   = Join-Path $root 'dist'
$bindir = Join-Path $dist $Backend
$server = Join-Path $bindir 'llama-server.exe'

if (-not (Test-Path $server)) {
  Write-Error "Backend '$Backend' not installed at $bindir. Run:  .\setup.ps1 -Backend $Backend"
  exit 1
}
if (-not $Model) {
  $Model = Get-ChildItem (Join-Path $root 'models') -Filter *.gguf -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName
  if (-not $Model) { Write-Error "No model. Pass -Model <file.gguf> or put a .gguf in .\models\"; exit 1 }
}
Write-Host "hanzo-engine | backend=$Backend | model=$(Split-Path $Model -Leaf) | http://127.0.0.1:$Port (OpenAI API)" -ForegroundColor Cyan

# ggml backend DLLs live in $bindir; ensure they resolve.
$env:PATH = "$bindir;$env:PATH"
$args = @('-m', $Model, '-ngl', $Ngl, '--port', $Port, '--host', '127.0.0.1', '-c', $Ctx, '--jinja', '--alias', $Alias)
if ($Embedding) { $args += @('--embedding','--pooling','mean') }
& $server @args
