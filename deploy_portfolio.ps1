# Medical Imaging Analysis - Portfolio Setup Script
# This script helps prepare your project for portfolio publication

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Medical Imaging Analysis - Portfolio Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create portfolio assets folder
Write-Host "[1/6] Creating portfolio assets folder..." -ForegroundColor Yellow
if (-not (Test-Path "portfolio_assets")) {
    New-Item -ItemType Directory -Path "portfolio_assets" | Out-Null
    Write-Host "  Created portfolio_assets folder" -ForegroundColor Green
} else {
    Write-Host "  portfolio_assets folder already exists" -ForegroundColor Green
}

# Step 2: Create demo data folder
Write-Host "[2/6] Creating demo data folder..." -ForegroundColor Yellow
if (-not (Test-Path "demo_data")) {
    New-Item -ItemType Directory -Path "demo_data" | Out-Null
    Write-Host "  Created demo_data folder" -ForegroundColor Green
} else {
    Write-Host "  demo_data folder already exists" -ForegroundColor Green
}

# Step 3: Check Git status
Write-Host "[3/6] Checking Git repository status..." -ForegroundColor Yellow
$gitExists = Get-Command git -ErrorAction SilentlyContinue
if ($gitExists) {
    $gitStatus = git status --porcelain 2>$null
    if ($gitStatus) {
        Write-Host "  You have uncommitted changes" -ForegroundColor Yellow
        git status --short
        Write-Host ""
        $commit = Read-Host "  Do you want to commit these changes? (y/n)"
        if ($commit -eq "y") {
            $message = Read-Host "  Enter commit message"
            git add .
            git commit -m "$message"
            Write-Host "  Changes committed" -ForegroundColor Green
        }
    } else {
        Write-Host "  Git repository is clean" -ForegroundColor Green
    }
} else {
    Write-Host "  Git not found. Skipping Git checks." -ForegroundColor Yellow
}

# Step 4: Check Docker
Write-Host "[4/6] Checking Docker installation..." -ForegroundColor Yellow
$dockerExists = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerExists) {
    $dockerVersion = docker --version 2>$null
    Write-Host "  Docker found: $dockerVersion" -ForegroundColor Green
    
    $buildDocker = Read-Host "  Do you want to build Docker image? (y/n)"
    if ($buildDocker -eq "y") {
        Write-Host "  Building Docker image..." -ForegroundColor Cyan
        docker build -t medical-imaging-analysis:latest .
        Write-Host "  Docker image built successfully" -ForegroundColor Green
    }
} else {
    Write-Host "  Docker not found. Install Docker to use containerized deployment." -ForegroundColor Yellow
}

# Step 5: Check Python and dependencies
Write-Host "[5/6] Checking Python environment..." -ForegroundColor Yellow
$pythonExists = Get-Command python -ErrorAction SilentlyContinue
if ($pythonExists) {
    $pythonVersion = python --version 2>&1
    Write-Host "  Python found: $pythonVersion" -ForegroundColor Green
    
    $installDeps = Read-Host "  Do you want to install/update dependencies? (y/n)"
    if ($installDeps -eq "y") {
        Write-Host "  Installing dependencies..." -ForegroundColor Cyan
        pip install -r requirements.txt
        Write-Host "  Dependencies installed" -ForegroundColor Green
    }
} else {
    Write-Host "  Python not found. Please install Python 3.8+" -ForegroundColor Red
}

# Step 6: Launch application
Write-Host "[6/6] Ready to launch application!" -ForegroundColor Yellow
$launch = Read-Host "  Do you want to launch the Streamlit app now? (y/n)"
if ($launch -eq "y") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Launching Medical Imaging Analysis App" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Remember to capture screenshots for your portfolio!" -ForegroundColor Magenta
    Write-Host "Save them to: portfolio_assets/" -ForegroundColor Magenta
    Write-Host ""
    streamlit run app_enhanced.py
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Setup Complete! Next Steps:" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Run the app: streamlit run app_enhanced.py" -ForegroundColor White
    Write-Host "2. Capture screenshots and save to portfolio_assets/" -ForegroundColor White
    Write-Host "3. Follow PORTFOLIO_GUIDE.md for deployment options" -ForegroundColor White
    Write-Host "4. Deploy to Streamlit Cloud for live demo" -ForegroundColor White
    Write-Host ""
    Write-Host "Read PORTFOLIO_GUIDE.md for detailed instructions" -ForegroundColor Green
}
