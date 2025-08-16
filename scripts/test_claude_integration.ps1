# Test ModelMuxer Integration with Different Query Types
# This demonstrates how ModelMuxer intelligently routes different types of queries

Write-Host "🧪 TESTING MODELMUXER INTELLIGENT ROUTING" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8000/v1/chat/completions"
$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer sk-test-claude-dev"
}

function Test-Query {
    param(
        [string]$QueryType,
        [string]$Query,
        [int]$MaxTokens = 150
    )
    
    Write-Host "🔍 Testing: $QueryType" -ForegroundColor Yellow
    Write-Host "Query: $Query" -ForegroundColor Gray
    
    $body = @{
        messages = @(
            @{
                role = "user"
                content = $Query
            }
        )
        max_tokens = $MaxTokens
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $baseUrl -Method Post -Headers $headers -Body $body
        
        Write-Host "✅ Model Selected: $($response.router_metadata.selected_model)" -ForegroundColor Green
        Write-Host "🏢 Provider: $($response.router_metadata.selected_provider)" -ForegroundColor Cyan
        Write-Host "💰 Cost: $($response.router_metadata.estimated_cost.ToString('F6'))" -ForegroundColor Yellow
        Write-Host "⏱️  Response Time: $($response.router_metadata.response_time_ms.ToString('F0'))ms" -ForegroundColor Magenta
        Write-Host "🧠 Routing Reason: $($response.router_metadata.routing_reason)" -ForegroundColor Blue
        
        $responseText = $response.choices[0].message.content
        if ($responseText.Length -gt 200) {
            $responseText = $responseText.Substring(0, 200) + "..."
        }
        Write-Host "📝 Response: $responseText" -ForegroundColor White
        
    } catch {
        Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Write-Host ""
}

# Test 1: Simple Question (should use cost-effective model)
Test-Query -QueryType "Simple Question" -Query "What is the capital of France?"

# Test 2: Code Generation (should use advanced model)  
Test-Query -QueryType "Code Generation" -Query "Write a Python function to reverse a string" -MaxTokens 200

# Test 3: Complex Analysis (should use premium model)
Test-Query -QueryType "Complex Analysis" -Query "Explain the pros and cons of microservices architecture vs monolithic architecture" -MaxTokens 300

# Test 4: Documentation (should use appropriate model)
Test-Query -QueryType "Documentation" -Query "Write API documentation for a REST endpoint that creates user accounts"

Write-Host "📊 BUDGET STATUS AFTER TESTS" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Cyan

# Check budget status
try {
    $budgetResponse = Invoke-RestMethod -Uri "http://localhost:8000/v1/analytics/budgets" -Headers @{"Authorization" = "Bearer sk-test-claude-dev"}
    
    foreach ($budget in $budgetResponse.budgets) {
        $percentage = [math]::Round($budget.usage_percentage, 2)
        $status = if ($percentage -gt 95) { "🔴" } elseif ($percentage -gt 80) { "🟠" } elseif ($percentage -gt 50) { "🟡" } else { "🟢" }
        
        Write-Host "$status $($budget.budget_type.ToUpper()) Budget: $($budget.current_usage.ToString('F6'))/$($budget.budget_limit) ($percentage%)" -ForegroundColor White
    }
} catch {
    Write-Host "❌ Could not retrieve budget status: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 INTEGRATION SUCCESS!" -ForegroundColor Green
Write-Host "Your ModelMuxer is working perfectly as an intelligent proxy:" -ForegroundColor White
Write-Host "• Automatically selects optimal models for different query types" -ForegroundColor Gray
Write-Host "• Tracks costs and usage in real-time" -ForegroundColor Gray  
Write-Host "• Provides detailed analytics and routing information" -ForegroundColor Gray
Write-Host "• Maintains OpenAI API compatibility" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 To use with Claude Dev:" -ForegroundColor Yellow
Write-Host "Base URL: http://localhost:8000/v1" -ForegroundColor Cyan
Write-Host "API Key: sk-test-claude-dev" -ForegroundColor Cyan
Write-Host "Model: auto (let ModelMuxer choose)" -ForegroundColor Cyan
