# Configure Claude CLI to work with ModelMuxer permanently
# This script sets up environment variables so Claude CLI routes through ModelMuxer

param(
    [string]$ModelMuxerUrl = "http://localhost:8000",
    [string]$ApiKey = "sk-test-claude-dev",
    [switch]$UserProfile,
    [switch]$SystemWide
)

Write-Host "=== Configuring Claude CLI for ModelMuxer ===" -ForegroundColor Green
Write-Host "ModelMuxer URL: $ModelMuxerUrl" -ForegroundColor Cyan
Write-Host "API Key: $ApiKey" -ForegroundColor Cyan

# Environment variables to set
$envVars = @{
    "ANTHROPIC_API_URL" = $ModelMuxerUrl
    "ANTHROPIC_BASE_URL" = $ModelMuxerUrl
    "ANTHROPIC_API_KEY" = $ApiKey
}

if ($SystemWide) {
    Write-Host "`nSetting system-wide environment variables..." -ForegroundColor Yellow
    foreach ($var in $envVars.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($var.Key, $var.Value, "Machine")
        Write-Host "Set $($var.Key) = $($var.Value)" -ForegroundColor Gray
    }
    Write-Host "System-wide variables set. Restart your terminal for changes to take effect." -ForegroundColor Green
}
elseif ($UserProfile) {
    Write-Host "`nSetting user profile environment variables..." -ForegroundColor Yellow
    foreach ($var in $envVars.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($var.Key, $var.Value, "User")
        Write-Host "Set $($var.Key) = $($var.Value)" -ForegroundColor Gray
    }
    Write-Host "User profile variables set. Restart your terminal for changes to take effect." -ForegroundColor Green
}
else {
    Write-Host "`nSetting session environment variables..." -ForegroundColor Yellow
    foreach ($var in $envVars.GetEnumerator()) {
        Set-Item -Path "env:$($var.Key)" -Value $var.Value
        Write-Host "Set $($var.Key) = $($var.Value)" -ForegroundColor Gray
    }
    Write-Host "Session variables set. Valid for current PowerShell session only." -ForegroundColor Green
}

# Create a startup script for easy activation
$startupScript = @"
# ModelMuxer Claude CLI Configuration
# Run this script to configure Claude CLI to use ModelMuxer

`$env:ANTHROPIC_API_URL = "$ModelMuxerUrl"
`$env:ANTHROPIC_BASE_URL = "$ModelMuxerUrl"
`$env:ANTHROPIC_API_KEY = "$ApiKey"

Write-Host "Claude CLI configured for ModelMuxer routing" -ForegroundColor Green
Write-Host "URL: `$env:ANTHROPIC_BASE_URL" -ForegroundColor Cyan
Write-Host "Key: `$env:ANTHROPIC_API_KEY" -ForegroundColor Cyan
"@

$startupScriptPath = Join-Path $PSScriptRoot "activate_claude_modelmuxer.ps1"
$startupScript | Out-File -FilePath $startupScriptPath -Encoding UTF8

Write-Host "`nCreated activation script: $startupScriptPath" -ForegroundColor Cyan
Write-Host "Run '. .\activate_claude_modelmuxer.ps1' to activate in any new session" -ForegroundColor Cyan

# Test the configuration
Write-Host "`nTesting configuration..." -ForegroundColor Yellow
try {
    $testResult = claude --print "Hello from ModelMuxer!" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Test successful! Claude CLI is working with ModelMuxer." -ForegroundColor Green
        Write-Host "Response: $testResult" -ForegroundColor Gray
    } else {
        Write-Host "❌ Test failed. Check ModelMuxer is running." -ForegroundColor Red
        Write-Host "Error: $testResult" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Configuration Complete ===" -ForegroundColor Green
Write-Host "Usage:" -ForegroundColor White
Write-Host "  claude 'Your question or task'"
Write-Host "  claude --print 'Your question or task'"
Write-Host "  (Interactive mode): claude"
Write-Host "`nFeatures enabled:" -ForegroundColor White
Write-Host "  ✅ Intelligent routing to best model for your task"
Write-Host "  ✅ Cost tracking and budget management"
Write-Host "  ✅ Support for multiple model providers"
Write-Host "  ✅ Enhanced mode features (PII protection, caching)"
Write-Host "  ✅ All requests logged and monitored"
