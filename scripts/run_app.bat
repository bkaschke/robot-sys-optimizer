@echo off
rem Launch the Robot Cell Optimizer.
rem Uses the bundled environment in env\ (portable zip, see scripts\make_portable.py) when present,
rem otherwise the Miniforge "rso" environment.
rem Usage: scripts\run_app.bat [--dark] [--port 8080]
setlocal
set "ENV=%~dp0..\env"
if exist "%ENV%\python.exe" goto portable
set "PY=%USERPROFILE%\miniforge3\envs\rso\python.exe"
if exist "%PY%" goto run
echo No Python environment found: neither "%ENV%" nor "%PY%".
echo See docs\USER_GUIDE.md, section "Set up on a new computer".
exit /b 1

:portable
set "PY=%ENV%\python.exe"
if exist "%ENV%\.unpacked" goto run
echo First start: preparing the bundled Python environment (once, about a minute)...
"%ENV%\Scripts\conda-unpack.exe"
if errorlevel 1 goto unpackfail
echo done> "%ENV%\.unpacked"
goto run

:unpackfail
echo conda-unpack failed. Unzip the bundle again into a folder you can write to and retry.
exit /b 1

:run
echo Starting... then open the http://localhost address shown below in your browser (default 8080; Ctrl+C here to stop).
"%PY%" "%~dp0..\app.py" %*
