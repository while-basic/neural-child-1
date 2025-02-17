@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

:: Create and activate virtual environment
if not exist docs_venv (
    echo Creating virtual environment...
    python -m venv docs_venv
)

:: Activate virtual environment
call docs_venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Build documentation
echo Building documentation...
mkdocs build

:: Handle arguments
if "%1"=="serve" (
    echo Starting documentation server...
    mkdocs serve
    goto :end
)

if "%1"=="version" (
    if "%2"=="" (
        echo Please provide a version number
        exit /b 1
    )
    echo Creating documentation for version %2...
    mike deploy --push --update-aliases %2 latest
    goto :end
)

:end
if "%1" neq "serve" (
    deactivate
    echo Documentation built successfully!
    echo You can find the static files in the 'site' directory
)

endlocal 