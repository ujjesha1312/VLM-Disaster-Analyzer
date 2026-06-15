"""
Package the extracted frames + frame manifest into a single zip file
for upload to Google Drive before running the Colab evaluation notebook.

Output: datasets/video_dataset/colab_frames.zip
  ├── frame_manifest.csv
  └── extracted_frames/
        ├── flood/
        ├── wildfire/
        ├── earthquake/
        ├── landslide/
        └── cyclone/
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent.parent
FRAMES_ROOT   = ROOT / "datasets/video_dataset/extracted_frames"
MANIFEST_CSV  = ROOT / "datasets/video_dataset/metadata/frame_manifest.csv"
OUTPUT_ZIP    = ROOT / "datasets/video_dataset/colab_frames.zip"

def main() -> None:
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Frame manifest not found: {MANIFEST_CSV}")

    frame_files = list(FRAMES_ROOT.rglob("*.jpg"))
    print(f"Found {len(frame_files)} frames to package")
    print(f"Output: {OUTPUT_ZIP}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add frame manifest
        zf.write(MANIFEST_CSV, "frame_manifest.csv")
        print(f"  + frame_manifest.csv")

        # Add all frames (preserving relative path from FRAMES_ROOT parent)
        for frame in sorted(frame_files):
            arcname = "extracted_frames/" + str(frame.relative_to(FRAMES_ROOT)).replace("\\", "/")
            zf.write(frame, arcname)

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 ** 2)
    print(f"\nDone: {OUTPUT_ZIP.name} ({size_mb:.1f} MB)")
    print(f"\nUpload this file to your Google Drive, then run the Colab notebook.")
    print(f"In the notebook Cell 3, set DRIVE_ZIP_PATH to the path in your Drive.")

if __name__ == "__main__":
    main()
