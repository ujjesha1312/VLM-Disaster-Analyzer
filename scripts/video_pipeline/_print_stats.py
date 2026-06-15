import sys, pandas as pd
sys.path.insert(0, "scripts/video_pipeline")
from config import METADATA_DIR, TARGET_CATEGORIES

vm = pd.read_csv(METADATA_DIR / "video_manifest.csv")
healthy = vm[vm["corrupted"] == False]

print("=== CATEGORY COUNTS & STORAGE ===")
counts  = vm.groupby("category").size()
storage = vm.groupby("category")["file_size_mb"].sum()
for cat in TARGET_CATEGORIES:
    print(f"  {cat:<12}: {counts.get(cat,0):2d} videos  {storage.get(cat,0):.2f} MB")
print(f"  {'TOTAL':<12}: {len(vm)} videos  {vm['file_size_mb'].sum():.2f} MB")

print()
print("=== VIDEO METADATA ===")
print(f"  Healthy videos  : {len(healthy)} / {len(vm)}")
print(f"  Avg duration    : {healthy['duration_seconds'].mean():.1f}s")
print(f"  Min / Max       : {healthy['duration_seconds'].min():.1f}s / {healthy['duration_seconds'].max():.1f}s")
print(f"  Avg FPS         : {healthy['fps'].mean():.1f}")
resolutions = healthy["resolution"].value_counts().head(4)
print(f"  Resolutions     : {dict(resolutions)}")

print()
print("=== SAMPLE FILES (3 per category) ===")
for cat in TARGET_CATEGORIES:
    rows = vm[vm["category"] == cat][["file_name","duration_seconds","file_size_mb","resolution"]].head(3)
    for _, r in rows.iterrows():
        print(f"  {cat}/{r['file_name']}  ({r['duration_seconds']}s, {r['file_size_mb']}MB, {r['resolution']})")

print()
print("=== GENERATED FILES ===")
for f in ["video_manifest.csv", "frame_manifest.csv", "dataset_statistics.xlsx"]:
    p = METADATA_DIR / f
    exists = "YES" if p.exists() else "NO"
    size   = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "-"
    print(f"  {f:<30}: {exists}  ({size})")
