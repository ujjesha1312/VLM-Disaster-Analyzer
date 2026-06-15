import io, time
import requests
from PIL import Image

BACKEND = "http://localhost:8000"
FRAME   = "datasets/video_dataset/extracted_frames/flood/413ye9QDIYo_41_43/frame_001.jpg"

img = Image.open(FRAME).convert("RGB")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90)
jpeg = buf.getvalue()

t0   = time.perf_counter()
resp = requests.post(f"{BACKEND}/predict/qwen", files={"file": ("frame.jpg", jpeg, "image/jpeg")}, timeout=600)
elapsed = round(time.perf_counter() - t0, 2)
print(f"HTTP {resp.status_code}  |  {elapsed}s")
print("Full response:")
print(resp.text[:1000])
