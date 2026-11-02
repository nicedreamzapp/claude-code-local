# Claude Code Local - Windows installer (llama.cpp backend, vision model), plus optional add-ons.
#
#   irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.ps1 | iex
#
# Clones the repo to %USERPROFILE%\claude-code-local, then detects your hardware,
# downloads a pinned llama.cpp build plus the best vision model that fits, and
# writes a launcher. Needs git and Python 3 on PATH. No Ollama, no LM Studio.
#
# Then it offers the add-ons that run on Windows. Each one is its own repo, installed by that
# repo's own Windows installer, so every piece also works on its own:
#   moonstone  Moonstone, every Claude Code chat in one browser panel   (github.com/nicedreamzapp/moonstone)
#   browser    browser-broker, agents share your real browser safely    (github.com/nicedreamzapp/browser-agent)
#   remote     FiaOS, your desktop and a terminal from any browser tab  (github.com/nicedreamzapp/FiaOS)
#   terminal   Trinidad Head, the terminal for local models             (github.com/nicedreamzapp/trinidad-head)
#   voice      NarrateClaude, talk to Claude Code and hear it answer    (github.com/nicedreamzapp/NarrateClaude)
#
# Pick add-ons without the menu (irm | iex can't take parameters, so use an env var):
#   $env:CCL_WITH = "moonstone,terminal"; irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.ps1 | iex
#   $env:CCL_WITH = "all"   or   "none"
# From a file:  powershell -ExecutionPolicy Bypass -File install.ps1 -With all
# The menu only appears in an interactive console; otherwise the default is none.
# Other settings: CCL_DIR (install folder), CCL_SKIP_MODEL=1 / -SkipModel (no llama.cpp or model
# download), CCL_NO_DESKTOP=1 / -NoDesktop (no Desktop launcher), CCL_NONINTERACTIVE=1 (never prompt).
# Uninstall: uninstall.ps1 in the install folder (can remove the add-ons too).

function Install-ClaudeCodeLocal {
  param(
    [string]$With = $env:CCL_WITH,
    [switch]$SkipModel = ($env:CCL_SKIP_MODEL -eq "1"),
    [switch]$NoDesktop = ($env:CCL_NO_DESKTOP -eq "1")
  )
  $ErrorActionPreference = "Stop"
  $RepoUrl    = "https://github.com/nicedreamzapp/claude-code-local"
  $InstallDir = if ($env:CCL_DIR) { $env:CCL_DIR } else { Join-Path $env:USERPROFILE "claude-code-local" }
  $Raw        = "https://raw.githubusercontent.com/nicedreamzapp"
  $AddOns = [ordered]@{
    moonstone = @{ Name = "Moonstone";      Url = "$Raw/moonstone/main/install.ps1";                 What = "every Claude Code chat in one browser panel (needs Node 18+)" }
    browser   = @{ Name = "browser-broker"; Url = "$Raw/browser-agent/main/broker/windows/get.ps1";  What = "agents share your real browser without stealing your tabs" }
    remote    = @{ Name = "FiaOS";          Url = "$Raw/FiaOS/main/windows/install.ps1";             What = "your desktop and a terminal from any browser tab" }
    terminal  = @{ Name = "Trinidad Head";  Url = "$Raw/trinidad-head/main/install.ps1";             What = "the terminal we recommend for local models" }
    voice     = @{ Name = "NarrateClaude";  Url = "$Raw/NarrateClaude/main/install.ps1";             What = "talk to Claude Code and hear it answer (local whisper.cpp + Piper)" }
  }
  $Alias = @{ fiaos = "remote"; broker = "browser"; "browser-broker" = "browser"; trinidad = "terminal"; "trinidad-head" = "terminal"; narrate = "voice"; narrateclaude = "voice" }

  function Test-Interactive {
    if ($env:CCL_NONINTERACTIVE -eq "1") { return $false }
    if (-not [Environment]::UserInteractive) { return $false }
    if ([Environment]::GetCommandLineArgs() | Where-Object { $_ -like "-NonI*" }) { return $false }
    try { if ([Console]::IsInputRedirected) { return $false } } catch { }
    return $true
  }
  function Resolve-AddOns([string]$spec) {
    $keys = @($AddOns.Keys)
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($w in ($spec -split '[,\s]+' | Where-Object { $_ })) {
      $w = $w.Trim().ToLower()
      if ($w -eq "all") { return $keys }
      if ($w -eq "none") { continue }
      if ($w -match '^\d+$' -and [int]$w -ge 1 -and [int]$w -le $keys.Count) { $w = $keys[[int]$w - 1] }
      if ($Alias.ContainsKey($w)) { $w = $Alias[$w] }
      if ($AddOns.Contains($w)) { if (-not $out.Contains($w)) { $out.Add($w) } }
      else { Write-Host "   (no add-on called '$w', skipped)" }
    }
    return $out.ToArray()
  }

  Write-Host ""
  Write-Host "=============================================="
  Write-Host "    Claude Code Local v1.0.0 - Windows installer"
  Write-Host "=============================================="
  Write-Host ""

  if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found. Install Git for Windows from https://git-scm.com/download/win" }
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python not found. Install Python 3 from https://www.python.org/downloads/ (check 'Add to PATH')" }

  Write-Host ("  Windows: " + (Get-CimInstance Win32_Processor).Name)
  Write-Host ""

  if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Host "-> Updating existing install at $InstallDir"
    git -C $InstallDir pull --ff-only
  } elseif (Test-Path $InstallDir) {
    throw "$InstallDir exists and is not a git checkout. Move it aside or set CCL_DIR."
  } else {
    Write-Host "-> Cloning into $InstallDir"
    git clone --depth 1 $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
  }

  $ccl = Join-Path $InstallDir "portable\ccl.py"

  if ($SkipModel) {
    Write-Host ""
    Write-Host "Skipping llama.cpp and the model (CCL_SKIP_MODEL). Run later:  python -I `"$ccl`" install"
  } else {
    Write-Host ""
    Write-Host "Detecting hardware and picking a model..."
    python -I $ccl detect
    Write-Host ""
    Write-Host "Downloading llama.cpp + model (one time, can be large)..."
    python -I $ccl install
    if ($LASTEXITCODE -ne 0) { throw "model install failed (exit $LASTEXITCODE)" }
  }

  # Launcher: a small .cmd in the install dir and on the Desktop.
  $launcher = Join-Path $InstallDir "claude-local.cmd"
  "@echo off`r`npython -I `"$ccl`" run %*`r`n" | Set-Content -Encoding ASCII $launcher
  if (-not $NoDesktop) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    Copy-Item $launcher (Join-Path $desktop "Claude Local.cmd") -Force
  }

  Write-Host ""
  Write-Host "=============================================="
  Write-Host "    Setup complete!"
  if (-not $NoDesktop) { Write-Host "    Double-click 'Claude Local' on your Desktop," }
  Write-Host "    or run:  $launcher"
  Write-Host "=============================================="
  if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Note: Claude Code CLI not found. Install it, then launch again."
  }

  # ---- Add-ons ----
  $interactive = Test-Interactive
  if (-not $With -and $interactive) {
    Write-Host ""
    Write-Host "Add-ons that run on Windows (each is its own repo and installer):"
    $i = 1
    foreach ($k in $AddOns.Keys) { Write-Host ("  {0}  {1,-10} {2}: {3}" -f $i, $k, $AddOns[$k].Name, $AddOns[$k].What); $i++ }
    $With = Read-Host "Pick numbers or names separated by commas, 'all', or press Enter for none"
  }
  $picked = @(Resolve-AddOns $With)
  if (-not $picked.Count) {
    Write-Host ""
    Write-Host "No add-ons. Add them any time: `$env:CCL_WITH = `"all`"; then run this installer again."
    return
  }
  $prevNI = $env:CCL_NONINTERACTIVE
  if (-not $interactive) { $env:CCL_NONINTERACTIVE = "1" }   # the add-on installers never prompt either

  $results = @()
  foreach ($k in $picked) {
    $a = $AddOns[$k]
    Write-Host ""
    Write-Host "---- $($a.Name) ----"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ccl-addon-$k-" + [guid]::NewGuid().ToString("N") + ".ps1")
    try {
      $src = (Invoke-WebRequest -UseBasicParsing -Uri $a.Url -Headers @{ "Cache-Control" = "no-cache" }).Content
      if ($src -is [byte[]]) { $src = [Text.Encoding]::UTF8.GetString($src) }
      [IO.File]::WriteAllText($tmp, $src)
      # Its own PowerShell, so one add-on failing (or calling exit) can't stop the others or close this window.
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tmp
      $code = $LASTEXITCODE
      $results += ("  {0,-15} {1}" -f $a.Name, $(if ($code -eq 0) { "installed" } else { "failed (exit $code), see above" }))
    } catch {
      $results += ("  {0,-15} failed: {1}" -f $a.Name, $_.Exception.Message)
    } finally {
      Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
  }
  $env:CCL_NONINTERACTIVE = $prevNI
  Write-Host ""
  Write-Host "Add-ons:"
  $results | ForEach-Object { Write-Host $_ }
}

Install-ClaudeCodeLocal @args
