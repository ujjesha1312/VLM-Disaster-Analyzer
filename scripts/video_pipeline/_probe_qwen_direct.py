"""
Run predict_response directly (no HTTP, no asyncio.to_thread) to isolate
whether the error is in qwen_model.py itself or in the async dispatch layer.
"""
import sys, logging, traceback
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# Bootstrap src/ path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FRAME = ROOT / "datasets/video_dataset/extracted_frames/flood/413ye9QDIYo_41_43/frame_001.jpg"

print(f"Frame: {FRAME}")
print(f"Exists: {FRAME.exists()}")
print()

try:
    print("Importing predict_response...")
    from models.qwen_model import predict_response
    print("Import OK")
    print()
    print("Running inference (this may take several minutes on CPU)...")
    result = predict_response(str(FRAME))
    print()
    print("SUCCESS:")
    print(result)
except Exception as e:
    print()
    print("FAILED with full traceback:")
    traceback.print_exc()
    print()
    print(f"Error type: {type(e).__name__}")
    print(f"Error msg : {e}")
