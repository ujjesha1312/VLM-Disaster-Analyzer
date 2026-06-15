"""
Probe all 4 VLM endpoints with a single sample frame.
Reports: success/failure, response format, inference time.
"""
import io, sys, time, json
import requests
from pathlib import Path
from PIL import Image

BACKEND = "http://localhost:8000"
MODELS  = ["clip", "blip2", "llava", "qwen"]
TIMEOUT = 120  # seconds per request

# Pick one frame from each category for the test
SAMPLE_FRAMES = {
    "flood":      "datasets/video_dataset/extracted_frames/flood/413ye9QDIYo_41_43/frame_001.jpg",
    "wildfire":   "datasets/video_dataset/extracted_frames/wildfire/5hghT1W33cY_139_143/frame_001.jpg",
    "earthquake": "datasets/video_dataset/extracted_frames/earthquake/9HhA9IlQ3f0_2_6/frame_001.jpg",
}

def to_jpeg_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def probe(model: str, frame_path: str) -> dict:
    url  = f"{BACKEND}/predict/{model}"
    jpeg = to_jpeg_bytes(frame_path)
    t0   = time.perf_counter()
    try:
        resp = requests.post(
            url,
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
            timeout=TIMEOUT,
        )
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            return {"status": "ok", "elapsed": round(elapsed, 2), "data": data}
        else:
            return {"status": f"http_{resp.status_code}", "elapsed": round(elapsed, 2), "data": resp.text[:300]}
    except requests.Timeout:
        return {"status": "timeout", "elapsed": TIMEOUT, "data": None}
    except requests.ConnectionError as e:
        return {"status": "connection_error", "elapsed": 0, "data": str(e)[:200]}
    except Exception as e:
        return {"status": "error", "elapsed": 0, "data": str(e)[:200]}

if __name__ == "__main__":
    # Use the flood frame as the standard test image
    frame = SAMPLE_FRAMES["flood"]
    if not Path(frame).exists():
        print(f"Sample frame not found: {frame}")
        sys.exit(1)

    print(f"Probe image : {frame}")
    print(f"Backend     : {BACKEND}")
    print("=" * 60)

    for model in MODELS:
        print(f"\n[{model.upper()}] probing...")
        result = probe(model, frame)

        status  = result["status"]
        elapsed = result["elapsed"]
        data    = result["data"]

        if status == "ok":
            metrics  = data.get("metrics", {})
            dis_type = metrics.get("disaster_type") or metrics.get("scene_description", "")[:60]
            conf     = metrics.get("confidence_score", "N/A")
            raw      = str(metrics.get("raw_analysis") or metrics.get("scene_description", ""))[:100]
            # Check if model was disabled/unavailable
            if data.get("status") in ("disabled", "unavailable"):
                print(f"  Status  : DISABLED/UNAVAILABLE")
                print(f"  Message : {data.get('message') or data.get('error','')}")
                print(f"  Time    : {elapsed}s")
            else:
                print(f"  Status       : SUCCESS")
                print(f"  Time         : {elapsed}s")
                print(f"  disaster_type: {dis_type}")
                print(f"  confidence   : {conf}")
                print(f"  raw_analysis : {raw[:80]}...")
                print(f"  Response keys: {list(data.keys())}")
                print(f"  Metrics keys : {list(metrics.keys())}")
        else:
            print(f"  Status  : FAILED ({status})")
            print(f"  Time    : {elapsed}s")
            print(f"  Detail  : {str(data)[:200]}")
