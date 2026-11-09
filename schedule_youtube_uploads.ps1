<#
.SYNOPSIS
    Register (or remove) three daily Windows Scheduled Tasks that keep the
    YouTube uploader running.

.DESCRIPTION
    The uploader schedules every finished outbox/output video on YouTube at
    3 videos/day. Running it three times a day means freshly rendered videos
    are picked up within a few hours. Tasks run run_upload.bat as the current
    user, so the signed-in Chrome/CDP session is available.

.EXAMPLES
    powershell -ExecutionPolicy Bypass -File .\schedule_youtube_uploads.ps1
    powershell -ExecutionPolicy Bypass -File .\schedule_youtube_uploads.ps1 -Remove
#>
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string[]]$Times = @("09:30", "13:30", "17:30"),
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$bat = Join-Path $ProjectRoot "run_upload.bat"
if (-not (Test-Path $bat)) {
    throw "run_upload.bat not found at $bat"
}

foreach ($time in $Times) {
    # Task names cannot contain ':' so "09:30" becomes "0930".
    $taskName = "AI Video Upload " + ($time -replace ':', '')

    if ($Remove) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed task: $taskName"
        continue
    }

    $action = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument "/c `"$bat`"" `
        -WorkingDirectory $ProjectRoot
    $trigger = New-ScheduledTaskTrigger -Daily -At $time
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 6)

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Schedule new AI videos on YouTube (3/day cadence)." `
        -Force | Out-Null

    Write-Host "Registered task: $taskName (daily at $time)"
}

Write-Host ""
Write-Host "Done. Verify with: Get-ScheduledTask -TaskName 'AI Video Upload*'"
