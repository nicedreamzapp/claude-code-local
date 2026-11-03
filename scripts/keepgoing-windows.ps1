# Claude Code Local - installs or removes "keep going" on Windows. install.ps1 and uninstall.ps1 run it.
#
#   powershell -ExecutionPolicy Bypass -File scripts\keepgoing-windows.ps1 [-Uninstall]
#
# Install does two things, both undone by -Uninstall:
#   1. adds <repo>\bin (keepgoing.cmd, keep.cmd) to your user PATH, so `keepgoing` and `keep going`
#      work in cmd and PowerShell even where profile scripts are blocked
#   2. adds one line to your PowerShell profiles (Windows PowerShell 5 and PowerShell 7) that
#      dot-sources scripts\keepgoing-profile.ps1
# Overrides: -Profiles / CCL_PROFILES (semicolon-separated profile files to edit instead of the
# default two), -NoPath / CCL_NO_PATH=1 (leave PATH alone). The tests use both.

param(
  [switch]$Uninstall,
  [string]$RepoDir = (Split-Path -Parent $PSScriptRoot),
  [string]$Profiles = $env:CCL_PROFILES,
  [switch]$NoPath = ($env:CCL_NO_PATH -eq "1")
)

$ErrorActionPreference = "Stop"
$Marker = "# Claude Code Local: type `"keep going`" when Claude runs out of credits"
$hook = Join-Path $RepoDir "scripts\keepgoing-profile.ps1"
$line = ". `"$hook`""
$binDir = Join-Path $RepoDir "bin"

if ($Profiles) {
  $targets = @($Profiles -split ';' | Where-Object { $_ })
} else {
  $docs = [Environment]::GetFolderPath("MyDocuments")
  $targets = @((Join-Path $docs "WindowsPowerShell\profile.ps1"), (Join-Path $docs "PowerShell\profile.ps1"))
}

function Remove-HookLines([string]$file) {
  if (-not (Test-Path $file)) { return $false }
  $lines = @(Get-Content $file)
  $keep = @($lines | Where-Object { $_ -ne $Marker -and $_ -notmatch 'scripts[\\/]keepgoing-profile\.ps1' })
  if ($keep.Count -eq $lines.Count) { return $false }
  Set-Content -Path $file -Value $keep -Encoding UTF8
  return $true
}

foreach ($p in $targets) {
  if ($Uninstall) {
    if (Remove-HookLines $p) { Write-Host "-> Removed the keep going line from $p" }
    continue
  }
  $dir = Split-Path -Parent $p
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  $have = (Test-Path $p) -and ((Get-Content -Raw $p) -match [regex]::Escape($line))
  if (-not $have) {
    Remove-HookLines $p | Out-Null   # an older line pointing at another clone
    Add-Content -Path $p -Value @("", $Marker, $line) -Encoding UTF8
  }
}

if (-not $NoPath) {
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  $parts = @(($userPath -split ';') | Where-Object { $_ })
  $has = $parts | Where-Object { $_.TrimEnd('\') -ieq $binDir.TrimEnd('\') }
  if ($Uninstall) {
    if ($has) {
      $rest = @($parts | Where-Object { $_.TrimEnd('\') -ine $binDir.TrimEnd('\') })
      [Environment]::SetEnvironmentVariable("Path", ($rest -join ';'), "User")
      Write-Host "-> Removed $binDir from your PATH"
    }
  } elseif (-not $has) {
    [Environment]::SetEnvironmentVariable("Path", (($parts + $binDir) -join ';'), "User")
  }
}

if (-not $Uninstall) {
  Write-Host "keep going installed: type 'keep going' (or keepgoing) in a new PowerShell or cmd window"
  $policy = Get-ExecutionPolicy
  if ($policy -in @("Restricted", "AllSigned")) {
    Write-Host "  (PowerShell's execution policy is $policy, so profiles don't load; keep going still works"
    Write-Host "   through keep.cmd on your PATH)"
  }
}
