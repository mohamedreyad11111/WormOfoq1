import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReyadAI")

# ---------------- App Instance ----------------
app = FastAPI(title="Reyad-AI Backend (FastAPI + Google GenAI)")

# ---------------- Models ----------------
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-1.5-flash"
    max_output_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "text-embedding-004"

# ---------------- Google GenAI Init ----------------
try:
    from google import genai
except ImportError:
    genai = None
    logger.warning("⚠️ google-genai SDK not installed!")

GENAI_CLIENT = None

@app.on_event("startup")
def init_genai_client():
    global GENAI_CLIENT
    if genai is None:
        logger.error("google-genai SDK missing. Install via: pip install google-genai")
        return

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠️ No GOOGLE_API_KEY found in environment.")
        GENAI_CLIENT = genai.Client()  # Allow local dev
    else:
        GENAI_CLIENT = genai.Client(api_key=api_key)
        logger.info("✅ Google GenAI client initialized successfully.")

# ---------------- Routes ----------------
@app.get("/")
async def home():
    return {"message": "🚀 Reyad-AI Backend is running successfully!"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Reyad-AI"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if GENAI_CLIENT is None:
        raise HTTPException(status_code=500, detail="GenAI client not initialized. Missing GOOGLE_API_KEY.")

    try:
        response = GENAI_CLIENT.models.generate_content(
            model=req.model,
            contents=req.prompt,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens
        )
        generated_text = getattr(response, "text", str(response))
        return {"response": generated_text, "model": req.model}
    except Exception as e:
        logger.exception("Error generating chat response")
        raise HTTPException(status_code=500, detail=f"GenAI error: {e}")

@app.post("/api/embeddings")
async def embeddings(req: EmbeddingRequest):
    if GENAI_CLIENT is None:
        raise HTTPException(status_code=500, detail="GenAI client not initialized.")

    try:
        if hasattr(GENAI_CLIENT.models, "embed_content"):
            resp = GENAI_CLIENT.models.embed_content(model=req.model, contents=req.texts)
        elif hasattr(GENAI_CLIENT.embeddings, "create"):
            resp = GENAI_CLIENT.embeddings.create(model=req.model, input=req.texts)
        else:
            raise RuntimeError("Embeddings API not supported by current SDK version.")
        return {"model": req.model, "raw": repr(resp)}
    except Exception as e:
        logger.exception("Error creating embeddings")
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")
