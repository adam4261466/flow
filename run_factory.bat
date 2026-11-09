@echo off
title AI Video Factory - Kids Bedtime Stories + Upload
echo Starting the kids bedtime-story factory...
echo.

cd /d "%~dp0"

:: Generate the queue (if prompts.txt is empty), render the stories, then
:: schedule them on YouTube at 3 videos/day. --kids keeps narration, images
:: and thumbnails gentle and child-friendly. --upload-each schedules each
:: story as soon as it finishes so the 3/day calendar keeps moving even while
:: the rest of the batch is still rendering.
::
:: --voices-first (default) writes and voices every queued story continuously
:: before rendering. --max-wait-minutes 0 means a Gemini/Claude limit is
:: announced once and then waited out quietly, however long it takes; the run
:: no longer stops after a fixed number of hours.
py -3.13 batch_generator.py --count 30 --per-day 3 --kids --voices-first --max-wait-minutes 0 --upload-each --upload

echo.
echo ========================================
echo Factory pass finished.
echo ========================================
pause
