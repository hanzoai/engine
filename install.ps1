#Requires -Version 5.1
<#
  Hanzo Engine installer for Windows — downloads a prebuilt, signed hanzoai.exe.

    irm https://raw.githubusercontent.com/hanzoai/engine/main/install.ps1 | iex

  Detects your CPU (amd64/arm64), fetches the matching bundle from the latest
  GitHub release, verifies it (cosign if present, else SHA256SUMS), and installs
  hanzoai.exe under %LOCALAPPDATA%\Hanzo\bin (user-level, no admin), on PATH.

  Env overrides: $env:HANZOAI_VERSION, $env:HANZOAI_INSTALL_DIR,
                 $env:HANZOAI_BASE_URL, $env:HANZOAI_NO_VERIFY=1
#>
$ErrorActionPreference = "Stop"
function Info($m)    { Write-Host "==> $m" -ForegroundColor Blue }
function Ok($m)      { Write-Host "OK  $m" -ForegroundColor Green }
function Warn($m)    { Write-Host "warning: $m" -ForegroundColor Yellow }
function Die($m)     { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

$Repo = "hanzoai/engine"

# --- platform ---
switch ($env:PROCESSOR_ARCHITECTURE) {
  "AMD64" { $arch = "amd64" }
  "ARM64" { $arch = "arm64" }
  "x86"   { if ($env:PROCESSOR_ARCHITEW6432 -eq "ARM64") { $arch = "arm64" } else { $arch = "amd64" } }
  default { Die "unsupported CPU architecture: $($env:PROCESSOR_ARCHITECTURE)" }
}
$asset = "hanzoai-windows-$arch.zip"

# --- release base ---
if ($env:HANZOAI_BASE_URL)      { $base = $env:HANZOAI_BASE_URL.TrimEnd('/'); $tag = if ($env:HANZOAI_VERSION) { $env:HANZOAI_VERSION } else { "mirror" } }
elseif ($env:HANZOAI_VERSION)   { $base = "https://github.com/$Repo/releases/download/$($env:HANZOAI_VERSION)"; $tag = $env:HANZOAI_VERSION }
else                            { $base = "https://github.com/$Repo/releases/latest/download"; $tag = "latest" }
$url = "$base/$asset"

Info "Hanzo Engine — installing $asset ($tag)"
$tmp = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("hanzoai-" + [guid]::NewGuid())) -Force
try {
  $zip = Join-Path $tmp $asset
  Info "Downloading $url"
  try { Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing } catch {
    Die "no prebuilt binary for windows/$arch in release $tag. See https://github.com/$Repo/releases"
  }

  # --- verify ---
  if ($env:HANZOAI_NO_VERIFY -ne "1") {
    $cosign = Get-Command cosign -ErrorAction SilentlyContinue
    $org = $Repo.Split('/')[0]
    if ($cosign) {
      try {
        Invoke-WebRequest -Uri "$url.sig" -OutFile "$zip.sig" -UseBasicParsing
        Invoke-WebRequest -Uri "$url.pem" -OutFile "$zip.pem" -UseBasicParsing
        & cosign verify-blob --certificate "$zip.pem" --signature "$zip.sig" `
          --certificate-identity-regexp "https://github.com/$org/.*" `
          --certificate-oidc-issuer "https://token.actions.githubusercontent.com" $zip 2>$null
        if ($LASTEXITCODE -ne 0) { Die "cosign verification FAILED for $asset" }
        Ok "cosign signature verified"
      } catch { Warn "cosign present but signature unavailable — skipping" }
    } else {
      try {
        $sums = Join-Path $tmp "SHA256SUMS"
        Invoke-WebRequest -Uri "$base/SHA256SUMS" -OutFile $sums -UseBasicParsing
        $want = (Select-String -Path $sums -Pattern ([regex]::Escape($asset)) | Select-Object -First 1).Line.Split(' ')[0]
        $got  = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
        if ($want -and ($want -eq $got)) { Ok "sha256 checksum verified" } else { Die "sha256 mismatch for $asset" }
      } catch { Warn "no cosign and no SHA256SUMS — skipping verification" }
    }
  }

  # --- extract + install ---
  Info "Extracting"
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
  $exe = Join-Path $tmp "hanzoai.exe"
  if (-not (Test-Path $exe)) { Die "archive did not contain hanzoai.exe" }

  $dir = if ($env:HANZOAI_INSTALL_DIR) { $env:HANZOAI_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Hanzo\bin" }
  New-Item -ItemType Directory -Path $dir -Force | Out-Null
  $dest = Join-Path $dir "hanzoai.exe"
  if (Test-Path $dest) { Info "Upgrading existing install at $dest" }
  Copy-Item -Path $exe -Destination $dest -Force
  Ok "installed hanzoai.exe -> $dest"

  # --- PATH (user) ---
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if (($userPath -split ';') -notcontains $dir) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$dir", "User")
    $env:Path = "$env:Path;$dir"
    Info "Added $dir to your user PATH (restart terminals to pick it up)"
  }

  try { Ok (& $dest --version) } catch { Warn "installed, but 'hanzoai --version' did not run cleanly" }
  Write-Host ""
  Write-Host "Next: serve an OpenAI + Anthropic compatible endpoint on :1234" -ForegroundColor White
  Write-Host "  hanzoai --port 1234 run -m Qwen/Qwen3-4B" -ForegroundColor Green
  Write-Host ""
}
finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
