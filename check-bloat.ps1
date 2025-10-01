param(
  [string]$Base = "origin/main",
  [string]$Head = "HEAD",
  [string]$Include = "",
  [string]$Exclude = "**/*.map,**/*.lock,**/node_modules/**",
  [string]$FailOn = "+1.5MB or +12%",
  [switch]$Report,
  [switch]$Json
)
$python = (Get-Command py -ErrorAction SilentlyContinue)?.Path
if (-not $python) { $python = (Get-Command python -ErrorAction SilentlyContinue)?.Path }
if (-not $python) { Write-Error "Python not found (py/python)."; exit 2 }
$env:BG_INCLUDE = $Include
$env:BG_EXCLUDE = $Exclude
$args = @(".\bloat_guard.py", "--ref-base", $Base, "--ref-head", $Head, "--fail-on", $FailOn)
if ($Report) { $args += "--report" }
if ($Json)   { $args += "--json" }
& $python @args
exit $LASTEXITCODE