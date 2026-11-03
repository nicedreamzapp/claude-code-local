@echo off
setlocal
rem keepgoing for Windows: resume this folder's Claude Code conversation on a free or local model.
rem   keepgoing            menu
rem   keepgoing --auto     no menu (free cloud model first, then the installed llama.cpp model)
rem   keepgoing --auto -- -p "finish the tests"     anything after -- goes to claude
python -I "%~dp0..\portable\ccl.py" keepgoing %*
exit /b %ERRORLEVEL%
