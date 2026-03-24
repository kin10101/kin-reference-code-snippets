## FastAPI + Ollama Example

This project exposes a small FastAPI app that proxies prompts to your local Ollama server.

### 1) Start Ollama

```powershell
ollama serve
ollama pull gemma3
```

### 2) Run the API

```powershell
uv run uvicorn run:app --reload --port 8000
```

### 3) Call the endpoint

```powershell
$body = @{
	prompt = "Explain recursion in one paragraph"
	model = "gemma3"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
	-Uri "http://127.0.0.1:8000/chat" `
	-ContentType "application/json" `
	-Body $body
```

Optional health check:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```
