# Brain Tumor Demo - Setup Script (Windows PowerShell)
# Run this script to copy model and sample images into the demo folder

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Brain Tumor Classifier Demo - Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$demoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $demoDir

# Create directories
Write-Host "`n[1/3] Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "$demoDir\model",
    "$demoDir\samples\glioma",
    "$demoDir\samples\meningioma",
    "$demoDir\samples\notumor",
    "$demoDir\samples\pituitary"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
}

# Copy model
Write-Host "`n[2/3] Copying model..." -ForegroundColor Yellow
$modelSrc = "$projectDir\models\brain_tumor_efficientnet_b0_v2_trained.pt"
$modelDst = "$demoDir\model\brain_tumor_efficientnet_b0.pt"

if (Test-Path $modelSrc) {
    Copy-Item $modelSrc $modelDst -Force
    $size = [math]::Round((Get-Item $modelDst).Length / 1MB, 1)
    Write-Host "  Copied: $modelDst ($size MB)" -ForegroundColor Green
}
else {
    Write-Host "  WARNING: Model not found at $modelSrc" -ForegroundColor Red
    Write-Host "  You need to train the model first or copy it manually." -ForegroundColor Red
}

# Copy sample images (5 per class)
Write-Host "`n[3/3] Copying sample images..." -ForegroundColor Yellow
$testingDir = "$projectDir\data\Brain_Tumor_Dataset\Testing"
$classes = @("glioma", "meningioma", "notumor", "pituitary")
$samplesPerClass = 5

foreach ($class in $classes) {
    $srcDir = "$testingDir\$class"
    $dstDir = "$demoDir\samples\$class"

    if (Test-Path $srcDir) {
        $files = Get-ChildItem "$srcDir\*.jpg" | Select-Object -First $samplesPerClass
        foreach ($file in $files) {
            Copy-Item $file.FullName $dstDir -Force
        }
        Write-Host "  $class : $($files.Count) images" -ForegroundColor Green
    }
    else {
        Write-Host "  $class : Source not found" -ForegroundColor Red
    }
}

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "`nDemo folder contents:" -ForegroundColor Yellow
Get-ChildItem $demoDir -Recurse -File |
Group-Object Directory |
ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) files"
}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. cd $demoDir"
Write-Host "  2. docker-compose up -d"
Write-Host "  3. Open http://localhost:5000"

Write-Host "`nOr run without Docker:" -ForegroundColor Yellow
Write-Host "  1. pip install -r requirements.txt"
Write-Host "  2. python app.py"
