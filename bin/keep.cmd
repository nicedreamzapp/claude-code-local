@echo off
setlocal
rem "keep going" for Windows cmd and PowerShell: same as keepgoing.cmd.
if /i "%~1"=="going" (
  shift
  goto run
)
echo did you mean: keep going
exit /b 1
:run
set "CCL_KG_ARGS="
:collect
if "%~1"=="" goto go
set CCL_KG_ARGS=%CCL_KG_ARGS% %1
shift
goto collect
:go
python -I "%~dp0..\portable\ccl.py" keepgoing %CCL_KG_ARGS%
exit /b %ERRORLEVEL%
