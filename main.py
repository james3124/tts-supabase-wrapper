"""
Chatterbox Turbo (via HF Inference API) → Supabase Wrapper
POST /api/tts  →  calls HF Inference, uploads audio to Supabase Storage,
                  logs request to Supabase DB, returns public URL.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
HF_TOKEN             = os.getenv("HF_TOKEN")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
STORAGE_BUCKET       = os.getenv("SUPABASE_STORAGE_BUCKET", "tts-audio")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

hf_client = InferenceClient(provider="auto", api_key=HF_TOKEN)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chatterbox × Supabase", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    model: Optional[str] = "ResembleAI/chatterbox-turbo:preferred"

class TTSResponse(BaseModel):
    request_id: str
    audio_url: str
    storage_path: str
    duration_ms: int
    created_at: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "provider": "HuggingFace Inference API"}


@app.post("/api/tts", response_model=TTSResponse)
async def generate_tts(req: TTSRequest):
    request_id = str(uuid.uuid4())
    started_at = time.time()

    # 1. Call HuggingFace Inference API
    try:
        audio_bytes = hf_client.text_to_speech(
            req.text,
            model=req.model,
        )
    except Exception as exc:
        raise HTTPException(502, f"HuggingFace Inference API error: {exc}")

    duration_ms = int((time.time() - started_at) * 1000)

    # 2. Upload to Supabase Storage
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    storage_path = f"{today}/{request_id}.wav"

    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=bytes(audio_bytes),
            file_options={
                "content-type": "audio/wav",
                "cache-control": "3600",
                "upsert": "false",
            },
        )
    except Exception as exc:
        raise HTTPException(500, f"Supabase Storage upload failed: {exc}")

    audio_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
    created_at = datetime.now(timezone.utc).isoformat()

    # 3. Log to Supabase DB (non-fatal)
    try:
        supabase.table("tts_requests").insert({
            "id":           request_id,
            "text_preview": req.text[:300],
            "voice_mode":   "hf-inference",
            "voice_id":     req.model,
            "output_format": "wav",
            "duration_ms":  duration_ms,
            "storage_path": storage_path,
            "audio_url":    audio_url,
            "created_at":   created_at,
        }).execute()
    except Exception as exc:
        print(f"[WARN] DB log failed (audio still saved): {exc}")

    return TTSResponse(
        request_id=request_id,
        audio_url=audio_url,
        storage_path=storage_path,
        duration_ms=duration_ms,
        created_at=created_at,
    )


@app.get("/api/history")
async def get_history(limit: int = 20, offset: int = 0):
    result = (
        supabase.table("tts_requests")
        .select("id, text_preview, voice_id, output_format, duration_ms, audio_url, created_at")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return {"data": result.data, "total": len(result.data)}


@app.delete("/api/audio/{request_id}")
async def delete_audio(request_id: str):
    row = supabase.table("tts_requests").select("storage_path").eq("id", request_id).single().execute()
    if not row.data:
        raise HTTPException(404, "Request not found")
    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([row.data["storage_path"]])
    except Exception as exc:
        raise HTTPException(500, f"Storage delete failed: {exc}")
    supabase.table("tts_requests").delete().eq("id", request_id).execute()
    return {"deleted": request_id}
