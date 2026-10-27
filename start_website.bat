@echo off
REM Starts the FabricScan AI website on http://localhost:5173 and opens it in the browser.
REM Close this window (or press Ctrl+C) to stop the website.
cd /d "%~dp0web"
if not exist node_modules (
  echo Installing website dependencies, first time only...
  call npm install || goto :error
)
if not exist public\models\fabricscan.onnx (
  echo.
  echo No model found in web\public\models\. Train one first, for example:
  echo   .venv\Scripts\python scripts\build_demo_model.py
  echo.
  pause
  exit /b 1
)
call npm run dev -- --open
exit /b 0

:error
echo Installing the website dependencies failed. Check your internet connection and try again.
pause
exit /b 1
