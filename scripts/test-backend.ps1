param(
    [int]$Runs = 3,
    [string]$Model = "kilo/kilo-auto/free",
    [string]$ApiKey = $env:KILO_API_KEY
)

$ErrorActionPreference = "Stop"
$Base = "https://api.kilo.ai/api/gateway"
$ApiKey = $env:KILO_API_KEY
if (-not $ApiKey) { Write-Host "KILO_API_KEY not set"; exit 1 }

$headers = @{ Authorization = "Bearer $ApiKey"; "Content-Type" = "application/json" }
$body = @{
    model = $Model
    messages = @(@{ role = "user"; content = "ping" })
    max_tokens = 16
} | ConvertTo-Json -Depth 5

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$Base/chat/completions" -Method Post -Headers $headers -Body $body -TimeoutSec 60
        $sw.Stop()
        $usage = $resp.usage
        $results += [pscustomobject]@{
            Run = $i
            Status = "OK"
            LatencyMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
            PromptTokens = $usage.prompt_tokens
            CompletionTokens = $usage.completion_tokens
            TotalTokens = $usage.total_tokens
        }
    } catch {
        $sw.Stop()
        $results += [pscustomobject]@{ Run = $i; Status = "FAIL: $($_.Exception.Message)"; LatencyMs = [math]::Round($sw.Elapsed.TotalMilliseconds,1); PromptTokens=$null; CompletionTokens=$null; TotalTokens=$null }
    }
}

$results | Format-Table -AutoSize
$ok = $results | Where-Object { $_.Status -eq "OK" }
if ($ok.Count -gt 0) {
    $avg = [math]::Round(($ok | Measure-Object -Property LatencyMs -Average).Average, 1)
    Write-Host "Avg latency: $avg ms over $($ok.Count)/$Runs runs"
}