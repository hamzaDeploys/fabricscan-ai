@echo off
REM Runs every automated test of the project: Python unit tests, the pipeline smoke test,
REM the website unit tests and the end-to-end browser test (Microsoft Edge).
cd /d "%~dp0"

echo.
echo ===== 1/4  Python unit tests =====
.venv\Scripts\python -m pytest || goto :fail

echo.
echo ===== 2/4  Data pipeline smoke test =====
.venv\Scripts\python scripts\smoke_test.py || goto :fail

cd web
echo.
echo ===== 3/4  Website unit tests =====
call npx vitest run || goto :fail

echo.
echo ===== 4/4  Website end-to-end test in Edge =====
call npx playwright test || goto :fail

echo.
echo ALL TESTS PASSED
pause
exit /b 0

:fail
echo.
echo SOME TESTS FAILED. Scroll up to see which one.
pause
exit /b 1
