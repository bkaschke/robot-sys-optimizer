@echo off
rem Launch the Robot Cell Optimizer with the Miniforge "rso" environment.
rem Usage: scripts\run_app.bat [--dark] [--port 8080]
set PY=%USERPROFILE%\miniforge3\envs\rso\python.exe
if not exist "%PY%" (
  echo Conda env "rso" not found at %PY%
  echo See docs\USER_GUIDE.md, section "Set up on a new computer".
  exit /b 1
)
echo Starting... then open http://localhost:8080 in your browser (Ctrl+C here to stop).
"%PY%" "%~dp0..\app.py" %*
