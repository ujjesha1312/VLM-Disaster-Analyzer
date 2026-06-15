import sys
sys.path.insert(0, "scripts/video_pipeline")
from config import RAW_VIDEOS_ROOT, TARGET_CATEGORIES, MIN_FILE_SIZE_KB

def count_good_files(cat):
    vid_dir = RAW_VIDEOS_ROOT / cat
    yt_id_counts = {}
    existing_keys = set()
    if not vid_dir.exists():
        return 0, yt_id_counts, existing_keys
    for p in vid_dir.glob("*.mp4"):
        if p.stat().st_size < MIN_FILE_SIZE_KB * 1024:
            continue
        parts = p.stem.rsplit("_", 2)
        if len(parts) >= 3:
            yt_id = parts[0]
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            yt_id_counts[yt_id] = yt_id_counts.get(yt_id, 0) + 1
            existing_keys.add((yt_id, start, end))
    return len(existing_keys), yt_id_counts, existing_keys

total = 0
for cat in TARGET_CATEGORIES:
    n, ytc, _ = count_good_files(cat)
    print(f"{cat:<12}: {n:2d} clips from {len(ytc)} YouTube IDs")
    total += n
print(f"TOTAL        : {total}")
