# Claude Code Local - Doctor for Windows (the same checks as scripts/doctor.sh)
#
#   powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1 [-Fix] [-Report]
#
#   -Fix      safe repairs only: start your own scheduled task for a service that is down, and
#             set up keep going (scripts\keepgoing-windows.ps1) if it is missing. Never deletes anything.
#   -Report   print a privacy-safe block to paste into a bug report (no usernames, home
#             paths, hostnames, IPs, tokens or env values) instead of the full output
#
# Exit code: 0 when nothing failed, 1 when at least one check failed.
# Add-ons that were never installed show as "not installed" and are never errors.
# Overrides (also used by the tests): MLX_PORT 4000, BROKER_PORT 9223, DOCK_PORT 8899,
# FIAOS_PORT 9000, CCL_HOME, CCL_DIR, NARRATE_DIR, NARRATE_DATA, FIAOS_DIR, MOONSTONE_DIR,
# TH_DIR, MOONSTONE_TASK, FIAOS_TASK, CCL_PROFILES, CCL_NO_PATH.

param(
  [switch]$Fix,
  [switch]$Report
)

$ErrorActionPreference = "Continue"
$DoctorVersion = "2"
$MinClaude = [version]"2.0.0"
$script:Results = New-Object System.Collections.Generic.List[object]
$script:ClaudeVer = $null; $script:PyVer = $null; $script:LlamaTag = $null

function Say([string]$t) { if (-not $Report) { Write-Host $t } }

function Record([string]$status, [string]$id, [string]$msg) {
  $script:Results.Add([pscustomobject]@{ Status = $status; Id = $id; Msg = $msg })
  $mark = @{ pass = "[ok]  "; warn = "[warn]"; fail = "[FAIL]"; skip = "[ -- ]" }[$status]
  Say "$mark $msg"
}

function EnvOr([string]$name, [string]$default) {
  $v = [Environment]::GetEnvironmentVariable($name)
  if ($v) { return $v } else { return $default }
}

function Get-HomeDir { if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME } }

# ---- Privacy: Protect-Text returns the text with nothing personal in it ----
function Protect-Text([string]$text) {
  $homeDir = Get-HomeDir
  $user = EnvOr "CCL_DOCTOR_USER" $env:USERNAME
  $hostName = EnvOr "CCL_DOCTOR_HOST" $env:COMPUTERNAME
  if (-not $hostName) { try { $hostName = [System.Net.Dns]::GetHostName() } catch { } }
  $t = $text
  if ($homeDir) {
    $t = $t.Replace($homeDir, "~")
    $t = $t.Replace($homeDir.Replace("\", "/"), "~")
  }
  if ($hostName -and $hostName.Length -ge 4) {
    $short = ($hostName -split '\.')[0]
    $t = $t -ireplace [regex]::Escape($hostName), "<host>"
    if ($short.Length -ge 4) { $t = $t -ireplace [regex]::Escape($short), "<host>" }
  }
  $t = $t -replace '(?i)[A-Z]:\\Users\\[^\\\s]+', "~"
  $t = $t -replace '/Users/[^/\s]+', "~"
  $t = $t -replace '/home/[^/\s]+', "~"
  if ($user -and $user.Length -ge 2) {
    $u = [regex]::Escape($user)
    $t = $t -ireplace "(?<![A-Za-z0-9_])$u(?![A-Za-z0-9_])", "<user>"
  }
  $t = $t -replace '[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', "<email>"
  $t = $t -replace '\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "<ip>"
  $t = $t -replace '\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b', "<mac>"
  $t = $t -replace '([0-9A-Fa-f]{0,4}:){3,7}[0-9A-Fa-f]{0,4}', "<ip>"
  $t = $t -replace '[A-Za-z0-9-]+\.(local|lan|home|internal|corp)(?![A-Za-z0-9])', "<host>"
  $t = $t -replace '\b(sk|pk|rk)-[A-Za-z0-9_-]{8,}', "<token>"
  $t = $t -replace '\b(ghp|gho|ghu|ghs|ghr|github_pat|hf|xox[abprs])[_-][A-Za-z0-9_-]{8,}', "<token>"
  $t = $t -replace '\bAKIA[0-9A-Z]{12,}', "<token>"
  $t = $t -replace '(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+', '$1 <token>'
  $t = $t -replace '(?i)\b([A-Za-z_]*(key|token|secret|passw(or)?d)[A-Za-z_]*)\s*[:=]\s*\S+', '$1=<redacted>'
  $t = $t -replace '(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]{2,})=\S+', '$1=<redacted>'
  $t = $t -replace '[A-Za-z0-9+=_]{32,}', "<token>"
  return $t
}

function Tilde([string]$p) {
  $h = Get-HomeDir
  if ($h -and $p.StartsWith($h, [StringComparison]::OrdinalIgnoreCase)) { return "~" + $p.Substring($h.Length) }
  return $p
}

# ---- Helpers ----
function Test-Port([int]$port) {
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $c.BeginConnect("127.0.0.1", $port, $null, $null)
    if ($iar.AsyncWaitHandle.WaitOne(2000) -and $c.Connected) { return $true }
    return $false
  } catch { return $false } finally { $c.Close() }
}

function Wait-Port([int]$port) {
  for ($i = 0; $i -lt 10; $i++) { if (Test-Port $port) { return $true }; Start-Sleep -Seconds 1 }
  return $false
}

function Find-Task([string]$prefix) {
  try {
    Get-ScheduledTask -ErrorAction Stop |
      Where-Object { $_.TaskName -like "$prefix*" -and $_.TaskName -notmatch 'update|tunnel' } |
      Select-Object -First 1
  } catch { $null }
}

function Test-Service([string]$id, [string]$label, [int]$port, [string]$taskPrefix, [bool]$required = $true) {
  if (Test-Port $port) { Record "pass" $id "$label answering on port $port"; return }
  $hint = ""
  $task = Find-Task $taskPrefix
  if ($Fix) {
    if ($task) {
      try { Start-ScheduledTask -TaskName $task.TaskName -ErrorAction Stop } catch { }
      if (Wait-Port $port) { Record "pass" $id "$label was down; started your own scheduled task, now answering on port $port (fixed)"; return }
      $hint = " (started its scheduled task but it is still not answering)"
    } else { $hint = " (no scheduled task of yours to start; start it by hand)" }
  } elseif ($task) { $hint = " (run with -Fix to start it)" }
  if ($required) { Record "fail" $id "$label not answering on port $port$hint" }
  else { Record "warn" $id "$label not answering on port $port$hint" }
}

function Get-ClaudeBin {
  $c = Get-Command claude -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($p in @((Join-Path $env:LOCALAPPDATA "Programs\claude\claude.exe"), (Join-Path (Get-HomeDir) ".local\bin\claude.exe"))) {
    if ($p -and (Test-Path $p)) { return $p }
  }
  return $null
}

function Test-Core {
  Say ""; Say "Core"
  # claude CLI
  $bin = Get-ClaudeBin
  if (-not $bin) { Record "fail" "claude" "claude CLI not found (install Claude Code, then re-run)" }
  else {
    $out = (& $bin --version 2>$null | Out-String)
    if ($out -match '(\d+\.\d+\.\d+)') {
      $script:ClaudeVer = $Matches[1]
      if ([version]$Matches[1] -ge $MinClaude) { Record "pass" "claude" "claude CLI $($Matches[1]) (need $MinClaude or newer)" }
      else { Record "fail" "claude" "claude CLI $($Matches[1]) is older than $MinClaude (update Claude Code)" }
    } else { Record "warn" "claude" "claude CLI found but its version could not be read" }
  }

  # Python
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) { Record "fail" "python" "python not found (install Python 3 and tick Add to PATH)" }
  else {
    $v = (& $py.Source --version 2>&1 | Out-String)
    if ($v -match '(\d+\.\d+(\.\d+)?)') {
      $script:PyVer = $Matches[1]
      if ([version]$Matches[1] -lt [version]"3.9") { Record "fail" "python" "Python $($Matches[1]) is too old (need 3.9+)" }
      else {
        & $py.Source -c "import venv, ssl" 2>$null
        if ($LASTEXITCODE -eq 0) { Record "pass" "python" "Python $($Matches[1]) with venv and ssl" }
        else { Record "fail" "python" "Python $($Matches[1]) lacks venv or ssl (reinstall Python from python.org)" }
      }
    } else { Record "warn" "python" "python found but its version could not be read" }
  }

  # Model files + llama.cpp binary, from the config the installer wrote
  $base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Get-HomeDir }
  $cclHome = EnvOr "CCL_HOME" (Join-Path $base "claude-code-local")
  $cfgPath = Join-Path $cclHome "config.json"
  if (-not (Test-Path $cfgPath)) {
    Record "fail" "models" "no model installed (run the installer, or: python -I portable\ccl.py install)"
    Record "fail" "llama" "llama.cpp not installed yet"
  } else {
    try { $cfg = Get-Content -Raw $cfgPath | ConvertFrom-Json } catch { $cfg = $null }
    if (-not $cfg) { Record "fail" "models" "the install config could not be read (re-run the installer)" }
    else {
      $script:LlamaTag = $cfg.llama_tag
      $hasModel = $cfg.model_gguf -and (Test-Path $cfg.model_gguf)
      $hasMm = $cfg.mmproj -and (Test-Path $cfg.mmproj)
      if ($hasModel -and $hasMm) { Record "pass" "models" "model files present ($($cfg.model_name), $($cfg.quant))" }
      elseif ($hasModel) { Record "warn" "models" "model present but its vision projector file is missing (re-run the installer)" }
      else { Record "fail" "models" "model file missing (re-run the installer to download it again)" }
      if ($cfg.server_bin -and (Test-Path $cfg.server_bin)) { Record "pass" "llama" "llama-server present (build $($cfg.llama_tag), $($cfg.accel))" }
      else { Record "fail" "llama" "llama-server.exe missing (re-run the installer)" }
    }
  }

  # Disk
  try {
    $drive = (Get-Item (Get-HomeDir)).PSDrive
    $free = [int]($drive.Free / 1GB)
    if ($free -ge 10) { Record "pass" "disk" "$free GB free" } else { Record "warn" "disk" "only $free GB free (models need 10 GB or more)" }
  } catch { }

  # Server: starts on demand with the launcher, so down is a warning
  Test-Service "server" "model server" ([int](EnvOr "MLX_PORT" "4000")) "ClaudeCodeLocal" $false
}

# keep going: bin\keepgoing.cmd on PATH and the PowerShell profile line (scripts\keepgoing-windows.ps1
# installs both). An older install has neither, so missing is a warning; -Fix runs that installer.
function Test-KeepGoing {
  $repo = Split-Path -Parent $PSScriptRoot
  $bin = Join-Path $repo "bin"
  if (-not (Test-Path (Join-Path $bin "keepgoing.cmd"))) {
    Record "fail" "keepgoing" "bin\keepgoing.cmd missing from the install (re-run the installer)"; return
  }
  $docs = [Environment]::GetFolderPath("MyDocuments")
  $profiles = if ($env:CCL_PROFILES) { @($env:CCL_PROFILES -split ';' | Where-Object { $_ }) }
              else { @((Join-Path $docs "WindowsPowerShell\profile.ps1"), (Join-Path $docs "PowerShell\profile.ps1")) }
  $hasProfile = { @($profiles | Where-Object { (Test-Path $_) -and ((Get-Content -Raw $_) -match 'keepgoing-profile\.ps1') }).Count -gt 0 }
  $userPath = "$([Environment]::GetEnvironmentVariable('Path', 'User'));$env:Path"
  $onPath = { @(($userPath -split ';') | Where-Object { $_ -and $_.TrimEnd('\') -ieq $bin.TrimEnd('\') }).Count -gt 0 }
  $okProfile = & $hasProfile; $okPath = & $onPath
  if ($Fix -and -not ($okProfile -and $okPath)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\keepgoing-windows.ps1") -RepoDir $repo | Out-Null
    $userPath = "$([Environment]::GetEnvironmentVariable('Path', 'User'));$env:Path"
    $okProfile = & $hasProfile; $okPath = & $onPath
    if ($okProfile -and ($okPath -or $env:CCL_NO_PATH -eq "1")) { Record "pass" "keepgoing" "keep going was not set up; installed it (fixed)"; return }
  }
  if ($okProfile -and $okPath) { Record "pass" "keepgoing" "keep going on PATH and in your PowerShell profile" }
  elseif ($okProfile) { Record "warn" "keepgoing" "keep going is in your PowerShell profile but bin is not on PATH, so cmd can't find it (run with -Fix)" }
  elseif ($okPath) { Record "warn" "keepgoing" "keep going is on PATH but not in your PowerShell profile (run with -Fix)" }
  else { Record "warn" "keepgoing" "keep going not installed (run with -Fix to add it)" }
}

function Test-AddOns {
  Say ""; Say "Add-ons"
  $h = Get-HomeDir
  $base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $h }

  # voice
  $voiceDir = EnvOr "NARRATE_DIR" (Join-Path $h "NarrateClaude")
  $voiceData = EnvOr "NARRATE_DATA" (Join-Path $base "narrateclaude")
  $binDir = EnvOr "KIT_BIN_DIR" (Join-Path $h ".local\bin")
  $narrate = (Get-Command narrate-claude -ErrorAction SilentlyContinue) -or (Test-Path (Join-Path $binDir "narrate-claude.cmd"))
  if ((Test-Path $voiceDir) -or $narrate) {
    if ($narrate) { Record "pass" "voice-cmd" "narrate-claude command present" }
    else { Record "fail" "voice-cmd" "narrate-claude command missing (re-run the voice installer)" }
    if (Get-ChildItem (Join-Path $voiceData "models") -Filter "ggml-*.bin" -ErrorAction SilentlyContinue) { Record "pass" "voice-whisper" "whisper model present" }
    else { Record "fail" "voice-whisper" "no whisper model in the voice data folder (re-run the voice installer)" }
    $speak = (Get-Command speak -ErrorAction SilentlyContinue) -or (Test-Path (Join-Path $binDir "speak.cmd")) -or (Test-Path (Join-Path $voiceData "venv\Scripts\piper.exe"))
    $onnx = Get-ChildItem (Join-Path $voiceData "models") -Filter "*.onnx" -ErrorAction SilentlyContinue
    if ($speak -and $onnx) { Record "pass" "voice-piper" "Piper voice and speak command present" }
    elseif ($speak) { Record "warn" "voice-piper" "speak/piper present but no Piper voice file found" }
    else { Record "fail" "voice-piper" "Piper (speak) not found (re-run the voice installer)" }
  } else { Record "skip" "voice" "voice: not installed" }

  # browser broker
  $brokerPort = [int](EnvOr "BROKER_PORT" "9223")
  $browserDir = EnvOr "BROWSER_BROKER_DIR" (Join-Path $h ".browser-agent")
  if ((Test-Path $browserDir) -or (Test-Path (Join-Path $h ".browser-broker")) -or (Test-Port $brokerPort)) {
    Test-Service "browser" "browser broker" $brokerPort "BrowserBroker" $true
  } else { Record "skip" "browser" "browser: not installed" }

  # moonstone
  $dockPort = [int](EnvOr "DOCK_PORT" (EnvOr "MOONSTONE_PORT" "8899"))
  if ((Test-Path (EnvOr "MOONSTONE_DIR" (Join-Path $h "moonstone"))) -or (Test-Port $dockPort)) {
    Test-Service "moonstone" "Moonstone" $dockPort (EnvOr "MOONSTONE_TASK" "Moonstone") $true
  } else { Record "skip" "moonstone" "moonstone: not installed" }

  # remote (FiaOS)
  $fiaPort = [int](EnvOr "FIAOS_PORT" "9000")
  if (Test-Path (EnvOr "FIAOS_DIR" (Join-Path $h "FiaOS"))) {
    Test-Service "remote" "FiaOS" $fiaPort (EnvOr "FIAOS_TASK" "FiaOS") $true
  } else { Record "skip" "remote" "remote: not installed" }

  # terminal (Trinidad Head)
  $thDir = EnvOr "TH_DIR" (Join-Path $base "Programs\TrinidadHead")
  if (Test-Path $thDir) {
    if (Test-Path (Join-Path $thDir "trinidad-head.exe")) { Record "pass" "terminal" "Trinidad Head installed" }
    else { Record "fail" "terminal" "Trinidad Head folder exists but trinidad-head.exe is missing (re-run its installer)" }
  } else { Record "skip" "terminal" "terminal: not installed" }

  Record "skip" "phone" "phone: macOS only"
  Record "skip" "failover" "failover: macOS only"
}

# ---- Main ----
Say ""
Say "=============================================="
Say "    Claude Code Local - Doctor (Windows)"
Say "=============================================="
if ($Fix) { Say "-Fix: safe repairs only (start your own scheduled task). Nothing is deleted." }

$cpu = try { (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim() } catch { "unknown CPU" }
$gpu = try { (Get-CimInstance Win32_VideoController | Select-Object -First 1).Name.Trim() } catch { "unknown GPU" }
$ramGb = try { [int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { 0 }
$osCaption = try { (Get-CimInstance Win32_OperatingSystem).Caption.Trim() } catch { "Windows" }
$osVersion = try { [Environment]::OSVersion.Version.ToString() } catch { "" }
Say ""
Say "This machine"
Say "  OS:     $osCaption $osVersion"
Say "  CPU:    $cpu"
Say "  GPU:    $gpu"
Say "  Memory: $ramGb GB"

Test-Core
Test-KeepGoing
Test-AddOns

$nPass = @($Results | Where-Object Status -eq "pass").Count
$nWarn = @($Results | Where-Object Status -eq "warn").Count
$nFail = @($Results | Where-Object Status -eq "fail").Count
$nSkip = @($Results | Where-Object Status -eq "skip").Count

if ($Report) {
  $failed = ($Results | Where-Object Status -eq "fail" | ForEach-Object { "  - " + $_.Msg }) -join "`n"
  $warned = ($Results | Where-Object Status -eq "warn" | ForEach-Object { "  - " + $_.Msg }) -join "`n"
  $passed = ($Results | Where-Object Status -eq "pass" | ForEach-Object { $_.Id }) -join " "
  $skipped = ($Results | Where-Object Status -eq "skip" | ForEach-Object { $_.Id }) -join " "
  if (-not $failed) { $failed = "  (none)" }
  if (-not $warned) { $warned = "  (none)" }
  if (-not $passed) { $passed = "(none)" }
  if (-not $skipped) { $skipped = "(none)" }
  $ver = "claude $(if ($ClaudeVer) { $ClaudeVer } else { 'none' }), python $(if ($PyVer) { $PyVer } else { 'none' })"
  if ($LlamaTag) { $ver += ", llama.cpp $LlamaTag" }
  $cclVer = "unknown"
  if ($PSScriptRoot) {
    $vf = Join-Path (Split-Path -Parent $PSScriptRoot) "VERSION"
    if (Test-Path $vf) { $cclVer = (Get-Content $vf -TotalCount 1).Trim() }
  }
  $block = @(
    "claude-code-local doctor report (doctor v$DoctorVersion, claude-code-local $cclVer, Windows)",
    "OS:        $osCaption $osVersion",
    "Chip/GPU:  $cpu / $gpu",
    "RAM:       $ramGb GB",
    "Versions:  $ver",
    "Summary:   $nPass ok, $nWarn warnings, $nFail failed, $nSkip not installed",
    "Failed:", $failed,
    "Warnings:", $warned,
    "Passed:        $passed",
    "Not installed: $skipped"
  ) -join "`n"
  Write-Output (Protect-Text $block)
} else {
  Say ""
  Say "Result: $nPass ok, $nWarn warnings, $nFail failed, $nSkip not installed"
  if ($nFail -gt 0) { Say "Filing a bug? Run: powershell -File scripts\doctor.ps1 -Report   and paste the block." }
  Say ""
}
if ($nFail -gt 0) { exit 1 } else { exit 0 }
