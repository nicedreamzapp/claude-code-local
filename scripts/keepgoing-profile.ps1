# Claude Code Local - "keep going" for PowerShell (the Windows twin of scripts/keepgoing-shell.sh).
# install.ps1 adds this line to your PowerShell profile for you:
#   . "$env:USERPROFILE\claude-code-local\scripts\keepgoing-profile.ps1"
#
# Then, when Claude runs out of credits: type `exit` in Claude Code, and `keep going`.
# `keepgoing --auto` skips the menu; anything after `--` goes to claude.

$global:CclKeepGoingPy = Join-Path (Split-Path -Parent $PSScriptRoot) "portable\ccl.py"

function global:keepgoing {
  & python -I $global:CclKeepGoingPy keepgoing @args
}

function global:keep {
  if ($args.Count -ge 1 -and $args[0] -eq "going") {
    $rest = @($args | Select-Object -Skip 1)
    & python -I $global:CclKeepGoingPy keepgoing @rest
  } else {
    Write-Host "did you mean: keep going"
  }
}
