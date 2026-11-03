# Claude Code Local - Windows uninstaller.
#
#   irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/uninstall.ps1 | iex
#   powershell -ExecutionPolicy Bypass -File uninstall.ps1 [-With all] [-Purge]
#
# Removes the Desktop launcher, the keep going PATH entry and PowerShell profile line, and the
# claude-code-local checkout (CCL_DIR). The downloaded
# llama.cpp and models in %LOCALAPPDATA%\claude-code-local (CCL_HOME) are kept unless -Purge.
#
# Add-ons: -With / $env:CCL_WITH takes the same names as install.ps1 (moonstone, browser, remote,
# terminal, voice, all). Each is removed by its own repo's uninstaller, which keeps that add-on's user
# data unless -Purge (or CCL_PURGE=1) is passed along. In an interactive console you are asked;
# otherwise no add-ons are touched.

function Uninstall-ClaudeCodeLocal {
  param(
    [string]$With = $env:CCL_WITH,
    [switch]$Purge = ($env:CCL_PURGE -eq "1")
  )
  $ErrorActionPreference = "Continue"
  $InstallDir = if ($env:CCL_DIR) { $env:CCL_DIR } else { Join-Path $env:USERPROFILE "claude-code-local" }
  $Home2 = if ($env:CCL_HOME) { $env:CCL_HOME } else { Join-Path $env:LOCALAPPDATA "claude-code-local" }
  $Raw = "https://raw.githubusercontent.com/nicedreamzapp"
  $broker = if ($env:BROWSER_BROKER_DIR) { $env:BROWSER_BROKER_DIR } else { Join-Path $env:USERPROFILE "browser-agent" }
  $moon = if ($env:MOONSTONE_DIR) { $env:MOONSTONE_DIR } else { Join-Path $env:USERPROFILE "moonstone" }
  $narr = if ($env:NARRATE_DIR) { $env:NARRATE_DIR } else { Join-Path $env:USERPROFILE "NarrateClaude" }
  $fia = if ($env:FIAOS_DIR) { $env:FIAOS_DIR } else { Join-Path $env:USERPROFILE "FiaOS" }
  # Local uninstaller first (it matches what is installed), the repo's copy if that is gone.
  $AddOns = [ordered]@{
    moonstone = @{ Name = "Moonstone";      Local = (Join-Path $moon "uninstall.ps1");                     Url = "$Raw/moonstone/main/uninstall.ps1" }
    browser   = @{ Name = "browser-broker"; Local = (Join-Path $broker "broker\windows\uninstall.ps1");     Url = $null }
    remote    = @{ Name = "FiaOS";          Local = (Join-Path $fia "windows\uninstall.ps1");              Url = "$Raw/FiaOS/main/windows/uninstall.ps1" }
    terminal  = @{ Name = "Trinidad Head";  Local = $null;                                                 Url = "$Raw/trinidad-head/main/uninstall.ps1" }
    voice     = @{ Name = "NarrateClaude";  Local = (Join-Path $narr "uninstall.ps1");                     Url = "$Raw/NarrateClaude/main/uninstall.ps1" }
  }
  $Alias = @{ fiaos = "remote"; broker = "browser"; "browser-broker" = "browser"; trinidad = "terminal"; "trinidad-head" = "terminal"; narrate = "voice"; narrateclaude = "voice" }

  function Test-Interactive {
    if ($env:CCL_NONINTERACTIVE -eq "1") { return $false }
    if (-not [Environment]::UserInteractive) { return $false }
    if ([Environment]::GetCommandLineArgs() | Where-Object { $_ -like "-NonI*" }) { return $false }
    try { if ([Console]::IsInputRedirected) { return $false } } catch { }
    return $true
  }

  Write-Host "== Claude Code Local - uninstall =="
  $desk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Claude Local.cmd"
  # Only the launcher that runs THIS install.
  if ((Test-Path $desk) -and ((Get-Content -Raw $desk).IndexOf($InstallDir, [StringComparison]::OrdinalIgnoreCase) -ge 0)) {
    Remove-Item $desk -Force; Write-Host "-> Removed $desk"
  }
  # keep going: its PATH entry and PowerShell profile line (CCL_PROFILES / CCL_NO_PATH as in install.ps1)
  $kgHelper = Join-Path $InstallDir "scripts\keepgoing-windows.ps1"
  if (Test-Path $kgHelper) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $kgHelper -Uninstall -RepoDir $InstallDir
  }
  if (Test-Path $InstallDir) {
    Set-Location $env:USERPROFILE
    Remove-Item $InstallDir -Recurse -Force
    Write-Host "-> Deleted $InstallDir"
  } else { Write-Host "-> $InstallDir not found" }
  if ($Purge) {
    if (Test-Path $Home2) { Remove-Item $Home2 -Recurse -Force; Write-Host "-> Deleted llama.cpp and models in $Home2" }
  } elseif (Test-Path $Home2) {
    Write-Host "-> Kept llama.cpp and models in $Home2 (run with -Purge to delete them)"
  }

  if (-not $With -and (Test-Interactive)) {
    $With = Read-Host "Also remove add-ons? (moonstone, browser, remote, terminal, voice, all; Enter for none)"
  }
  $keys = @($AddOns.Keys); $picked = @()
  foreach ($w in ($With -split '[,\s]+' | Where-Object { $_ })) {
    $w = $w.Trim().ToLower()
    if ($w -eq "all") { $picked = $keys; break }
    if ($Alias.ContainsKey($w)) { $w = $Alias[$w] }
    if ($AddOns.Contains($w) -and $picked -notcontains $w) { $picked += $w }
  }
  foreach ($k in $picked) {
    $a = $AddOns[$k]
    Write-Host ""
    Write-Host "---- $($a.Name) ----"
    $file = $null; $tmp = $null
    if ($a.Local -and (Test-Path $a.Local)) { $file = $a.Local }
    elseif ($a.Url) {
      $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ccl-un-$k-" + [guid]::NewGuid().ToString("N") + ".ps1")
      try {
        $src = (Invoke-WebRequest -UseBasicParsing -Uri $a.Url).Content
        if ($src -is [byte[]]) { $src = [Text.Encoding]::UTF8.GetString($src) }
        [IO.File]::WriteAllText($tmp, $src); $file = $tmp
      } catch { Write-Host "   could not fetch its uninstaller: $($_.Exception.Message)" }
    }
    if (-not $file) { Write-Host "   not installed here (no uninstaller found), skipped"; continue }
    $argv = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $file)
    if ($Purge) { $argv += "-Purge" }
    & powershell.exe @argv
    if ($tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
  }
  Write-Host ""
  Write-Host "Done."
}

Uninstall-ClaudeCodeLocal @args
