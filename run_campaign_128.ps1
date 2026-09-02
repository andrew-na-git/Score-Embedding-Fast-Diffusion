<#
.SYNOPSIS
    Sequential driver for the 128 px DAVIS inpainting campaign.

.DESCRIPTION
    Runs, per seed, the three stages the headline table needs, then FVD across
    all seeds:

      1. control-off : trains the model and samples every method
                       (video / per_frame / copy_prev) into
                       saves/video/davis128_seed<S>
      2. control-on  : reuses stage 1's checkpoint with the Krylov flow-consistency
                       projection enabled, into ..._control
      3. propainter  : the external baseline, into ..._propainter

    Stages are skipped when their results.json already exists, so the script is
    safe to re-run after an interruption; training itself also resumes from
    train_ckpt.pth within a stage.

    ProPainter runs under its own interpreter (-PropainterPython) because its
    dependency set is installed separately; the stage is skipped with a warning
    if that interpreter or external/ProPainter is missing.

.EXAMPLE
    pwsh -File run_campaign_128.ps1 -Seeds 9,42,123
#>
[CmdletBinding()]
param(
    [int[]]$Seeds = @(9, 42, 123),
    [string]$Config = "fast_diffusion/configs/video/davis_inpaint_128.yml",
    [string]$Base = "saves/video/davis128",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$PropainterPython = ".\.venv-propainter\Scripts\python.exe",
    [switch]$SkipPropainter,
    [string]$FvdOut = "figures/fvd_128.json"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Write-Stage([string]$Message) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

# A stage is complete when it wrote results.json; anything less is treated as
# unfinished so a killed run is retried rather than silently reported as done.
function Test-StageDone([string]$Dir) {
    return Test-Path (Join-Path $Dir "results.json")
}

function Invoke-Stage([string]$Label, [string]$Exe, [string[]]$Arguments, [string]$LogFile) {
    Write-Host "-> $Label" -ForegroundColor Yellow
    Write-Host "   $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    $started = Get-Date
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE (log: $LogFile)"
    }
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    Write-Host "   done in $mins min" -ForegroundColor Green
}

$campaignStart = Get-Date

foreach ($seed in $Seeds) {
    $offDir = "${Base}_seed${seed}"
    $onDir = "${Base}_seed${seed}_control"
    $ppDir = "${Base}_seed${seed}_propainter"

    # ---------------------------------------------------------- 1. control off
    Write-Stage "seed $seed  |  stage 1/3  control OFF (train + sample all methods)"
    if (Test-StageDone $offDir) {
        Write-Host "   already complete, skipping" -ForegroundColor DarkGray
    }
    else {
        Invoke-Stage "seed $seed control-off" $Python @(
            "run_video.py", "--config", $Config, "--seed", "$seed", "--out-dir", $offDir
        ) "logs/davis128_seed${seed}_off.log"
    }

    # ----------------------------------------------------------- 2. control on
    Write-Stage "seed $seed  |  stage 2/3  control ON (flow_consistency projection)"
    if (Test-StageDone $onDir) {
        Write-Host "   already complete, skipping" -ForegroundColor DarkGray
    }
    else {
        # --no-train loads model.pth from the *output* directory, so stage 1's
        # checkpoint is copied across rather than retrained. Same weights on both
        # sides of the ablation is the whole point: any difference is the control.
        New-Item -ItemType Directory -Force -Path $onDir | Out-Null
        Copy-Item (Join-Path $offDir "model.pth") (Join-Path $onDir "model.pth") -Force
        Invoke-Stage "seed $seed control-on" $Python @(
            "run_video.py", "--config", $Config, "--seed", "$seed", "--out-dir", $onDir,
            "--no-train", "--skip-baselines",
            "--constraints", "flow_consistency",
            "--constraint-weight", "1.0", "--constraint-ridge", "0.0", "--cg-maxiter", "8"
        ) "logs/davis128_seed${seed}_on.log"
    }

    # ----------------------------------------------------------- 3. propainter
    Write-Stage "seed $seed  |  stage 3/3  ProPainter baseline"
    if ($SkipPropainter) {
        Write-Host "   -SkipPropainter given, skipping" -ForegroundColor DarkGray
    }
    elseif (Test-StageDone $ppDir) {
        Write-Host "   already complete, skipping" -ForegroundColor DarkGray
    }
    elseif (-not (Test-Path $PropainterPython) -or -not (Test-Path "external/ProPainter")) {
        Write-Warning "ProPainter not set up ($PropainterPython / external/ProPainter); skipping."
    }
    else {
        Invoke-Stage "seed $seed propainter" $PropainterPython @(
            "run_propainter.py", "--config", $Config, "--seed", "$seed", "--out-dir", $ppDir
        ) "logs/davis128_seed${seed}_propainter.log"
    }
}

# -------------------------------------------------------------------- 4. FVD
Write-Stage "FVD across seeds: $($Seeds -join ', ')"
$fvdArgs = @("compute_fvd.py", "--config", $Config, "--base", $Base, "--seeds")
$fvdArgs += ($Seeds | ForEach-Object { "$_" })
$fvdArgs += @("--out", $FvdOut)
Invoke-Stage "fvd" $Python $fvdArgs "logs/davis128_fvd.log"

$hours = [math]::Round(((Get-Date) - $campaignStart).TotalHours, 2)
Write-Stage "campaign complete in $hours h -- FVD written to $FvdOut"
