param(
    [Parameter(Mandatory = $true)]
    [int]$FinalLitterCoordinatorProcessId
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$coordinatorLog = Join-Path $resultRoot 'final_regression.log'
$exitPath = Join-Path $resultRoot 'final_regression.exitcode'
$litterExitPath = Join-Path $resultRoot 'litter_scoop_final.exitcode'

Set-Location -LiteralPath $repositoryRoot
Set-Content -LiteralPath $coordinatorLog -Value (
    "Waiting for final litter coordinator $FinalLitterCoordinatorProcessId at $((Get-Date).ToString('o'))"
)
Wait-Process -Id $FinalLitterCoordinatorProcessId -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $litterExitPath)) {
    Add-Content -LiteralPath $coordinatorLog -Value 'Final litter trajectory exit code is missing.'
    Set-Content -LiteralPath $exitPath -Value 1 -Encoding ascii
    exit 1
}
$litterExitCode = [int](Get-Content -LiteralPath $litterExitPath -Raw)
if ($litterExitCode -ne 0) {
    Add-Content -LiteralPath $coordinatorLog -Value "Final litter trajectory failed: $litterExitCode"
    Set-Content -LiteralPath $exitPath -Value $litterExitCode -Encoding ascii
    exit $litterExitCode
}

$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$ruff = Join-Path $repositoryRoot '.venv\Scripts\ruff.exe'
$tests = @(
    'tests\core\test_multiphysics_controller_state.py',
    'tests\core\test_multiphysics_public_api.py',
    'tests\core\test_multiphysics_adapter_boundaries.py',
    'tests\core\test_scene_restore.py',
    'tests\core\test_coffee_water_scenario.py',
    'tests\core\test_granular_fluid_integration.py',
    'tests\core\test_garment_folding_scenario.py',
    'tests\core\test_butter_spreading_scenario.py'
)

Add-Content -LiteralPath $coordinatorLog -Value "Starting final regression at $((Get-Date).ToString('o'))"
& $python -m pytest -q -n 0 @tests 2>&1 | Tee-Object -FilePath $coordinatorLog -Append
$regressionExitCode = $LASTEXITCODE
if ($regressionExitCode -eq 0) {
    & $ruff check examples\multiphysics tests\core\test_multiphysics_controller_state.py `
        tests\core\test_multiphysics_public_api.py tests\core\test_multiphysics_adapter_boundaries.py `
        tests\core\test_scene_restore.py scripts\validate_multiphysics_checkpoint.py `
        scripts\smoke_dem_tilt_checkpoint.py 2>&1 | Tee-Object -FilePath $coordinatorLog -Append
    $regressionExitCode = $LASTEXITCODE
}
if ($regressionExitCode -eq 0) {
    & $python -m compileall -q examples\multiphysics genesis\engine\scene.py genesis\engine\states\solvers.py `
        genesis\engine\solvers\dem_solver.py genesis\engine\entities\pbstf_entity.py `
        genesis\engine\solvers\rigid\collider\collider.py tests\core scripts 2>&1 | `
        Tee-Object -FilePath $coordinatorLog -Append
    $regressionExitCode = $LASTEXITCODE
}
if ($regressionExitCode -eq 0) {
    git diff --check 2>&1 | Tee-Object -FilePath $coordinatorLog -Append
    $regressionExitCode = $LASTEXITCODE
}

Set-Content -LiteralPath $exitPath -Value $regressionExitCode -Encoding ascii
Add-Content -LiteralPath $coordinatorLog -Value (
    "Final regression ended with exit code $regressionExitCode at $((Get-Date).ToString('o'))"
)
exit $regressionExitCode
