param(
  [string]$Base = "origin/main",
  [string]$Head = "HEAD",
  [string]$Include = "",
  [string]$Exclude = "**/*.map,**/*.lock,**/node_modules/**",
  [string]$FailOn = "+1.5MB or +12%",
  [switch]$Report,
  [switch]$Json
)
$python = $null
try { $cmd = Get-Command py -ErrorAction Stop; $python = $cmd.Path } catch {}
if (-not $python) { try { $cmd = Get-Command python -ErrorAction Stop; $python = $cmd.Path } catch {} }
if (-not $python) { Write-Error "Python not found (py/python) in PATH."; exit 2 }
$env:BG_INCLUDE = $Include
$env:BG_EXCLUDE = $Exclude
$argv = @(".\bloat_guard.py", "--ref-base", $Base, "--ref-head", $Head, "--fail-on", $FailOn)
if ($Report) { $argv += "--report" }
if ($Json)   { $argv += "--json" }
& $python @argv
exit $LASTEXITCODE