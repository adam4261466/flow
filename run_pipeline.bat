@echo off
title AI Video Creator - Launcher
echo Starting AI Video Creator...
echo.

:: Navigate to the project directory
cd /d "C:\Users\kabli\OneDrive\Desktop\projects\flow"

:: Run the pipeline using Python 3.13
:: Defaults to the current configuration. 
:: To target a specific story, change this to: py -3.13 main.py --story story_001
py -3.13 main.py

echo.
echo ========================================
echo Process Finished.
echo ========================================
pause
