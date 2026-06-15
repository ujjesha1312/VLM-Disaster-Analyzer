"""Quick probe for CLIP and BLIP-2 only."""
import io, sys, time, json
import requests
from pathlib import Path
from PIL import Image

BACKEND = "http://localhost:8000"
FRAME   = "datasets/video_dataset/extracted_frames/flood/413ye9QDIYo_41_43/frame_001.jpg"

def to_jpeg_bytes(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def probe(model, frame_path, timeout=90):
    url  = f"{BACKEND}/predict/{model}"
    jpeg = to_jpeg_bytes(frame_path)
    t0   = time.perf_counter()
    try:
        resp = requests.post(url, files={"file": ("frame.jpg", jpeg, "image/jpeg")}, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 2)
        data = resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text
        return resp.status_code, elapsed, data
    except Exception as e:
        return 0, round(time.perf_counter() - t0, 2), str(e)

for model in ["clip", "blip2", "llava"]:
    print(f"\n{'='*50}")
    print(f"MODEL: {model.upper()}")
    code, elapsed, data = probe(model, FRAME)
    print(f"HTTP  : {code}  |  Time: {elapsed}s")
    if isinstance(data, dict):
        print(f"Keys  : {list(data.keys())}")
        status = data.get("status","")
        if status in ("disabled","unavailable"):
            print(f"Status: {status.upper()}")
            print(f"Detail: {data.get('message') or data.get('error','')}")
            print(f"Hint  : {data.get('hint','')}")
        else:
            m = data.get("metrics", {})
            print(f"Status      : SUCCESS")
            print(f"disaster_type: {m.get('disaster_type','N/A')}")
            print(f"confidence  : {m.get('confidence_score','N/A')}")
            print(f"raw_analysis: {str(m.get('raw_analysis',''))[:120]}")
    else:
        print(f"Response: {str(data)[:300]}")
