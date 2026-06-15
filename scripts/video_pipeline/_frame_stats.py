import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts/video_pipeline")
from config import METADATA_DIR, FRAMES_ROOT, TARGET_CATEGORIES

fm = pd.read_csv(METADATA_DIR / "frame_manifest.csv")

print("=== FRAME COUNTS PER CATEGORY ===")
counts = fm.groupby("category").size()
for cat in TARGET_CATEGORIES:
    n = counts.get(cat, 0)
    videos = fm[fm["category"] == cat]["video_stem"].nunique()
    print(f"  {cat:<12}: {n:3d} frames  ({videos} videos x {n//max(videos,1)} frames each)")
print(f"  {'TOTAL':<12}: {len(fm):3d} frames")

print()
print("=== FRAME MANIFEST SAMPLE (2 per category) ===")
for cat in TARGET_CATEGORIES:
    rows = fm[fm["category"] == cat][["video_stem", "frame_filename", "frame_number"]].head(2)
    for _, r in rows.iterrows():
        print(f"  {cat}/{r['video_stem']}/{r['frame_filename']}")

print()
print("=== FRAME STORAGE ===")
total_bytes = 0
for cat in TARGET_CATEGORIES:
    cat_dir = FRAMES_ROOT / cat
    files = list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.png"))
    size_kb = sum(f.stat().st_size for f in files) / 1024
    total_bytes += size_kb * 1024
    print(f"  {cat:<12}: {len(files):3d} files  {size_kb/1024:.2f} MB")
print(f"  {'TOTAL':<12}:          {total_bytes/1024/1024:.2f} MB")

print()
print("=== MANIFEST FILES ===")
for name in ["video_manifest.csv", "frame_manifest.csv", "dataset_statistics.xlsx"]:
    p = METADATA_DIR / name
    size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING"
    rows = ""
    if name.endswith(".csv") and p.exists():
        rows = f"  ({len(pd.read_csv(p))} rows)"
    print(f"  {name:<30}: {size}{rows}")

print()
print("=== SAMPLE FRAME PATHS ===")
for cat in TARGET_CATEGORIES:
    cat_dir = FRAMES_ROOT / cat
    frames = sorted(cat_dir.rglob("frame_001.jpg"))[:1]
    for f in frames:
        rel = f.relative_to(FRAMES_ROOT.parent.parent)
        print(f"  {rel}")
