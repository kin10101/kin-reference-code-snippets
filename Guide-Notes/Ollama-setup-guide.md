# How to get Ollama running in your system

1. Download ollama from https://ollama.com/download
2. Pull a model
3. Host your model

## Getting models

Pull a model:
```bash
ollama pull llama3.1:8b
ollama pull mistral
ollama pull gemma3
```

Run interactively in terminal:
```bash
ollama run llama3.2
```

## Common commands

```bash
ollama list          # see downloaded models
ollama ps            # see running models
ollama rm llama3.2   # delete a model
```

## Serving as an API

To run as an API locally:
```bash
ollama serve
# runs on http://localhost:11434 by default
```

### Option 1: Expose Ollama on your network (simplest, LAN only)

By default Ollama only listens on 127.0.0.1. Change it to listen on all interfaces.

**Windows — set environment variable:**

In PowerShell (permanent):
```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
# Then restart Ollama
```

Or in the Ollama service settings, add:
```
OLLAMA_HOST=0.0.0.0:11434
```

Then from another machine on your LAN:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://192.168.1.x:11434"  # your machine's local IP
)
```

Find your local IP:
```powershell
ipconfig  # look for IPv4 Address
```

⚠️ **No auth** — anyone on your network can use it.

### Option 2: Cloudflare Tunnel (remote access, free, no port forwarding)

Best option for accessing from outside your home network securely.

```bash
# Install cloudflared on Windows
winget install Cloudflare.cloudflared

# Create a quick tunnel (no account needed)
cloudflared tunnel --url http://localhost:11434
```

It gives you a public HTTPS URL like:
```
https://some-random-name.trycloudflare.com
```

Use it in LangChain:
```python
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="https://some-random-name.trycloudflare.com"
)
```

⚠️ **URL changes every restart.** For a permanent URL you need a free Cloudflare account.

### Option 3: Ngrok (simple, permanent URL on free tier)

```powershell
winget install Ngrok.Ngrok

# Authenticate (free account at ngrok.com)
ngrok config add-authtoken YOUR_TOKEN

# Expose Ollama
ngrok http 11434
```

Gives you:
```
https://xxxx-xx-xx-xx.ngrok-free.app
```

Use in code:
```python
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="https://xxxx-xx-xx-xx.ngrok-free.app"
)
```

### Option 4: Wrap it in a FastAPI service (most control)

Good if you want to add auth, rate limiting, logging etc:

**server.py:**
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_ollama import ChatOllama
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer()
API_KEY = "your-secret-key"

def verify_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_KEY:
        raise HTTPException(status_code=403)

class Query(BaseModel):
    question: str
    model: str = "llama3.1:8b"

@app.post("/ask", dependencies=[Depends(verify_key)])
async def ask(query: Query):
    llm = ChatOllama(model=query.model, num_thread=8)
    response = llm.invoke(query.question)
    return {"answer": response.content}
```

Install and run:
```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

Call it from anywhere:
```python
import requests

response = requests.post(
    "https://your-tunnel-url/ask",
    headers={"Authorization": "Bearer your-secret-key"},
    json={"question": "What is RAG?"}
)
print(response.json()["answer"])
```

## Which to pick

| Scenario | Recommendation |
|----------|---|
| Same home network only | Option 1 (OLLAMA_HOST) |
| Access from anywhere, quick setup | Option 2 (Cloudflare) |
| Access from anywhere, stable URL | Option 3 (Ngrok) |
| Production / multi-user / auth | Option 4 (FastAPI) |

**For a personal dev setup, Cloudflare Tunnel + OLLAMA_HOST=0.0.0.0 is the sweet spot — free, secure, and no code needed.**