@echo off
setlocal EnableDelayedExpansion

echo Starting Digital Child Development System...

:: Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate
    set FIRST_RUN=1
) else (
    call venv\Scripts\activate
)

REM Install packages with specific versions and order
echo Installing dependencies...
pip install --upgrade pip
pip install numpy==1.24.3
pip install --upgrade setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit==1.24.0
pip install plotly==5.15.0
pip install psutil==5.9.5
pip install transformers==4.30.2
pip install python-dateutil==2.8.2
pip install wandb==0.15.4 --no-deps
pip install accelerate==0.20.3
pip install bitsandbytes==0.39.1
pip install gradio==3.35.2

REM Verify torch installation
python -c "import torch; print('PyTorch version:', torch.__version__)" || (
    echo Error: PyTorch installation failed
    pause
    exit /b 1
)

REM Launch the application
streamlit run app.py
if errorlevel 1 (
    echo Error: Failed to start Streamlit app
    pause
)
