from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx


app = FastAPI(title="Local Ollama API Example")


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to send to Ollama")
    model: str = Field(default="gemma3", description="Installed Ollama model name")


class ChatResponse(BaseModel):
    model: str
    response: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama(payload: ChatRequest) -> ChatResponse:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": payload.model,
                    "prompt": payload.prompt,
                    "stream": False,
                },
            )
            res.raise_for_status()
            data = res.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama at http://localhost:11434. Start it with 'ollama serve'.",
        ) from exc

    return ChatResponse(model=payload.model, response=data.get("response", ""))