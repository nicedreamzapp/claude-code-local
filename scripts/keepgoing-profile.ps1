# Claude Code Local - "keep going" for PowerShell (the Windows twin of scripts/keepgoing-shell.sh).
# install.ps1 adds this line to your PowerShell profile for you:
#   . "$env:USERPROFILE\claude-code-local\scripts\keepgoing-profile.ps1"
#
# Then, when Claude runs out of credits: type `exit` in Claude Code, and `keep going`.
# `keepgoing --auto` skips the menu; anything after `--` goes to claude.

$global:CclKeepGoingPy = Join-Path (Split-Path -Parent $PSScriptRoot) "portable\ccl.py"

# PowerShell drops a bare `--` from a function's arguments, so `keep going --auto -- -p "..."`
# reached ccl.py as `--auto -p ...`: the -p never went to claude, which started interactively and
# waited forever. Put the `--` back before the first argument that isn't keep going's own --auto.
function global:CclKeepGoingArgs {
  $in = @($args)
  if ($in -contains "--") { return ,$in }
  $out = New-Object System.Collections.Generic.List[string]
  $added = $false
  foreach ($a in $in) {
    if (-not $added -and $a -ne "--auto") { $out.Add("--"); $added = $true }
    $out.Add([string]$a)
  }
  return ,$out.ToArray()
}

function global:keepgoing {
  $kgArgs = CclKeepGoingArgs @args
  & python -I $global:CclKeepGoingPy keepgoing @kgArgs
}

function global:keep {
  if ($args.Count -ge 1 -and $args[0] -eq "going") {
    $rest = @($args | Select-Object -Skip 1)
    $kgArgs = CclKeepGoingArgs @rest
    & python -I $global:CclKeepGoingPy keepgoing @kgArgs
  } else {
    Write-Host "did you mean: keep going"
  }
}
