"""Quick probe for Qwen only — long timeout for CPU model loading."""
import io, time
import requests
from PIL import Image

BACKEND = "http://localhost:8000"
FRAME   = "datasets/video_dataset/extracted_frames/flood/413ye9QDIYo_41_43/frame_001.jpg"

def to_jpeg_bytes(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

print("Probing Qwen (CPU model — may take 2-5 min to load on first call)...")
jpeg = to_jpeg_bytes(FRAME)
t0   = time.perf_counter()
resp = requests.post(
    f"{BACKEND}/predict/qwen",
    files={"file": ("frame.jpg", jpeg, "image/jpeg")},
    timeout=600,
)
elapsed = round(time.perf_counter() - t0, 2)

print(f"HTTP    : {resp.status_code}  |  Time: {elapsed}s")
data = resp.json()
print(f"Keys    : {list(data.keys())}")
status = data.get("status","")
if status in ("disabled","unavailable"):
    print(f"Status  : {status.upper()}")
    print(f"Detail  : {data.get('message') or data.get('error','')}")
else:
    m = data.get("metrics", {})
    print(f"Status        : SUCCESS")
    print(f"disaster_type : {m.get('disaster_type','N/A')}")
    print(f"confidence    : {m.get('confidence_score','N/A')}")
    raw = str(m.get("raw_analysis") or m.get("scene_description",""))
    print(f"raw_analysis  : {raw[:200]}")
    print(f"Metrics keys  : {list(m.keys())}")
