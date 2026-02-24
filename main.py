"""
Heart Staging — FastAPI Backend  
Endpoints:
  POST /transcribe         — audio -> Whisper -> testo
  POST /segment-audio      — trascrizione + stanze -> segmenti per stanza
  POST /analyze-clarity    — foto stanza -> GPT-4V -> score + gap
  POST /staging-brief      — foto + audio + livello -> brief per generazione
  POST /generate           — brief + foto -> immagine staged (Replicate SDXL)
  GET  /health             — health check
"""

import os, base64, json, httpx, asyncio
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Heart Staging API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("it")):
    try:
        audio_bytes = await audio.read()
        audio_file = BytesIO(audio_bytes)
        audio_file.name = audio.filename or "recording.webm"
        response = await client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language=language,
            response_format="verbose_json", timestamp_granularities=["segment"]
        )
        segments = response.segments or []
        return {
            "ok": True,
            "text": response.text,
            "segments": [
                {
                    "start": s["start"] if isinstance(s, dict) else s.start,
                    "end": s["end"] if isinstance(s, dict) else s.end,
                    "text": (s["text"] if isinstance(s, dict) else s.text).strip()
                }
                for s in segments
            ],
            "duration": getattr(response, "duration", None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-audio")
async def segment_audio(transcript: str = Form(...), rooms: str = Form(...)):
    try:
        room_list = json.loads(rooms)
        prompt = f"""Hai una trascrizione audio di un appartamento.
Stanze: {', '.join(room_list)}
Trascrizione: {transcript}
Segmenta il testo per stanza. Rispondi SOLO con JSON:
{{"segments": [{{"room": "nome stanza", "text": "descrizione"}}]}}"""
        response = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.1
        )
        return {"ok": True, **json.loads(response.choices[0].message.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-clarity")
async def analyze_clarity(room_name: str = Form(...), photos: list[UploadFile] = File(...), audio_transcript: Optional[str] = Form(None)):
    try:
        content = []
        intro = f"Stanza: {room_name}."
        if audio_transcript:
            intro += f" Audio: {audio_transcript}"
        content.append({"type": "text", "text": intro + " Analizza e rispondi con JSON."})
        for photo in photos[:6]:
            pb = await photo.read()
            b64 = base64.b64encode(pb).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{photo.content_type or 'image/jpeg'};base64,{b64}", "detail": "high"}})
        system = """Sei un esperto di homestaging. Analizza le foto e rispondi SOLO con JSON:
{"score": <0-100>, "score_breakdown": {"layout": <0-25>, "lighting": <0-25>, "surfaces": <0-25>, "perspectives": <0-25>}, "gaps": ["<gap>"], "visible_elements": ["<elemento>"], "staging_notes": "<note>"}"""
        response = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_object"}, temperature=0.2, max_tokens=1000
        )
        return {"ok": True, **json.loads(response.choices[0].message.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/staging-brief")
async def staging_brief(room_name: str = Form(...), staging_level: str = Form(...), photos: list[UploadFile] = File(...), audio_transcript: Optional[str] = Form(None), linked_rooms: Optional[str] = Form(None)):
    try:
        linked = json.loads(linked_rooms) if linked_rooms else []
        level_desc = {"soft": "Solo elementi decorativi. Mantieni arredo esistente.", "medium": "Sostituisci o aggiungi mobili principali.", "full": "Rifai completamente l'interior."}.get(staging_level, "Soft")
        ctx = f"Stanza: {room_name}. Livello: {staging_level} - {level_desc}"
        if audio_transcript: ctx += f". Audio: {audio_transcript}"
        if linked: ctx += f". Visibile da: {', '.join(linked)}."
        content = [{"type": "text", "text": ctx}]
        for photo in photos[:4]:
            pb = await photo.read()
            b64 = base64.b64encode(pb).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{photo.content_type or 'image/jpeg'};base64,{b64}", "detail": "high"}})
        system = """Sei un esperto homestaging Airbnb di lusso. Rispondi SOLO con JSON:
{"style": "<stile>", "palette": ["<c1>","<c2>","<c3>"], "elements_to_add": [{"type": "<tipo>","description": "<desc>","position": "<dove>","priority": "high|medium|low"}], "elements_to_remove": ["<elem>"], "lighting_adjustments": "<note>", "sd_prompt": "<prompt SD XL>", "sd_negative_prompt": "<negative>", "consistency_notes": "<note coerenza>"}"""
        response = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_object"}, temperature=0.3, max_tokens=1500
        )
        return {"ok": True, "room": room_name, "level": staging_level, **json.loads(response.choices[0].message.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_image(photo: UploadFile = File(...), sd_prompt: str = Form(...), sd_negative_prompt: str = Form(""), staging_level: str = Form("soft"), room_name: str = Form("")):
    replicate_key = os.getenv("REPLICATE_API_KEY")
    if not replicate_key:
        raise HTTPException(status_code=503, detail="REPLICATE_API_KEY non configurata")
    try:
        photo_bytes = await photo.read()
        b64_image = base64.b64encode(photo_bytes).decode()
        mime = photo.content_type or "image/jpeg"
        full_prompt = f"luxury airbnb interior, professional real estate photography, {sd_prompt}, 8k, photorealistic, warm lighting, staged home, editorial quality"
        negative = sd_negative_prompt or "blurry, distorted, low quality, cartoon, anime, unrealistic, cluttered, dark, overexposed, people, text, watermark"
        strength_map = {"soft": 0.40, "medium": 0.60, "full": 0.80}
        strength = strength_map.get(staging_level, 0.50)
        async with httpx.AsyncClient(timeout=30) as http:
            create_res = await http.post(
                "https://api.replicate.com/v1/predictions",
                headers={"Authorization": f"Token {replicate_key}", "Content-Type": "application/json"},
                json={"version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                      "input": {"image": f"data:{mime};base64,{b64_image}", "prompt": full_prompt, "negative_prompt": negative,
                                "prompt_strength": strength, "num_inference_steps": 30, "guidance_scale": 7.5,
                                "width": 1024, "height": 768, "scheduler": "DPMSolverMultistep",
                                "refine": "expert_ensemble_refiner", "high_noise_frac": 0.8}}
            )
        if create_res.status_code != 201:
            raise HTTPException(status_code=502, detail=f"Replicate error: {create_res.text}")
        prediction = create_res.json()
        poll_url = prediction["urls"]["get"]
        async with httpx.AsyncClient(timeout=30) as http:
            for _ in range(40):
                await asyncio.sleep(3)
                poll_res = await http.get(poll_url, headers={"Authorization": f"Token {replicate_key}"})
                poll_data = poll_res.json()
                status = poll_data.get("status")
                if status == "succeeded":
                    output = poll_data.get("output", [])
                    return {"ok": True, "room": room_name, "image_url": output[0] if output else "", "prompt_used": full_prompt, "strength": strength}
                elif status == "failed":
                    raise HTTPException(status_code=502, detail=f"Replicate failed: {poll_data.get('error')}")
        raise HTTPException(status_code=504, detail="Replicate timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
