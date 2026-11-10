$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& 'C:\Users\admin\miniconda3\envs\genesis-world\python.exe' (Join-Path $projectRoot 'remote.py') start >> (Join-Path $PSScriptRoot 'startup.log') 2>&1
exit $LASTEXITCODE
