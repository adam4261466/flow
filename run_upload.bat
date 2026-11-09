@echo off
title AI Video Factory - YouTube Upload
echo Scheduling finished videos on YouTube...
echo.

cd /d "%~dp0"
if not exist output mkdir output

py -3.13 youtube_uploader.py --per-day 3 >> "output\youtube_upload.log" 2>&1

echo.
echo ========================================
echo Upload pass finished. Log: output\youtube_upload.log
echo ========================================
