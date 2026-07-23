"""
Chatterbox TTS → Supabase Integration Wrapper
POST /api/tts  →  calls Chatterbox, uploads audio to Supabase Storage,
                  logs request to Supabase DB, returns public URL.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
CHATTERBOX_URL        = os.getenv("CHATTERBOX_URL", "http://localhost:8004")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
STORAGE_BUCKET        = os.getenv("SUPABASE_STORAGE_BUCKET", "tts-audio")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chatterbox × Supabase", version="1.0.0")

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
    voice_mode: Optional[str]           = "predefined"
    predefined_voice_id: Optional[str]  = None
    reference_audio_filename: Optional[str] = None
    output_format: Optional[str]        = "mp3"   # wav | mp3 | opus
    split_text: Optional[bool]          = True
    chunk_size: Optional[int]           = 120
    temperature: Optional[float]        = None
    exaggeration: Optional[float]       = None
    cfg_weight: Optional[float]         = None
    seed: Optional[int]                 = None
    speed_factor: Optional[float]       = None
    language: Optional[str]             = None

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
    """Quick liveness probe for Render."""
    return {"status": "ok", "chatterbox": CHATTERBOX_URL}


@app.post("/api/tts", response_model=TTSResponse)
async def generate_tts(req: TTSRequest):
    request_id = str(uuid.uuid4())
    started_at = time.time()

    # 1. Build Chatterbox payload (drop None values)
    payload = {k: v for k, v in req.dict().items() if v is not None}
    payload["stream"] = False  # storage workflow needs the full file

    fmt = payload.get("output_format", "mp3")
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/opus"}
    mime = mime_map.get(fmt, "audio/mpeg")

    # 2. Call Chatterbox
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            cb_resp = await client.post(
                f"{CHATTERBOX_URL}/tts",
                json=payload,
            )
            cb_resp.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(504, "Chatterbox TTS request timed out (>5 min)")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Chatterbox returned {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        raise HTTPException(502, f"Cannot reach Chatterbox at {CHATTERBOX_URL}: {exc}")

    duration_ms = int((time.time() - started_at) * 1000)
    audio_bytes = cb_resp.content

    # 3. Upload to Supabase Storage
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    storage_path = f"{today}/{request_id}.{fmt}"

    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=audio_bytes,
            file_options={
                "content-type": mime,
                "cache-control": "3600",
                "upsert": "false",
            },
        )
    except Exception as exc:
        raise HTTPException(500, f"Supabase Storage upload failed: {exc}")

    audio_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
    created_at = datetime.now(timezone.utc).isoformat()

    # 4. Log to Supabase DB (non-fatal if it fails)
    try:
        supabase.table("tts_requests").insert({
            "id":           request_id,
            "text_preview": req.text[:300],
            "voice_mode":   req.voice_mode,
            "voice_id":     req.predefined_voice_id,
            "output_format": fmt,
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
    """Return recent TTS requests from the DB log."""
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
    """Delete a specific audio file from Storage and its DB log row."""
    # Find the row first
    row = supabase.table("tts_requests").select("storage_path").eq("id", request_id).single().execute()
    if not row.data:
        raise HTTPException(404, "Request not found")

    storage_path = row.data["storage_path"]

    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([storage_path])
    except Exception as exc:
        raise HTTPException(500, f"Storage delete failed: {exc}")

    supabase.table("tts_requests").delete().eq("id", request_id).execute()
    return {"deleted": request_id}
