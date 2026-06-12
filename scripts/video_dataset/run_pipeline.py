"""
run_pipeline.py — Full VIDI dataset pipeline orchestrator.

Runs all six stages in dependency order:
  1  metadata    — Fetch all 43 annotation CSVs from GitHub → master_annotations.csv
  2  splits      — Build stratified train / val / test splits at the video level
  3  download    — Download YouTube videos + trim clips with ffmpeg
  4  verify      — ffprobe integrity check on every clip
  5  preprocess  — Extract thumbnails (+ optional resolution normalisation)
  6  statistics  — Generate reports, class distribution, dataset_summary.md

Usage examples:
  # Full pipeline (all stages):
  python run_pipeline.py

  # English-only subset (~2,100 clips, ~47% of dataset — much faster):
  python run_pipeline.py --english-only

  # Resume from a specific stage (e.g. after interrupted download):
  python run_pipeline.py --from-stage download

  # Run a single stage only:
  python run_pipeline.py --stage verify

  # Dry-run download (simulate without fetching):
  python run_pipeline.py --stage download --dry-run

  # Fewer parallel workers (lower bandwidth / CPU pressure):
  python run_pipeline.py --workers 4

Google Colab quick-start:
  !git clone https://github.com/your-repo/vlm-disaster-analyzer
  %cd vlm-disaster-analyzer
  !pip install -r scripts/video_dataset/requirements.txt
  !apt-get install -y ffmpeg
  import os; os.environ["VLM_PROJECT_ROOT"] = "/content/vlm-disaster-analyzer"
  !python scripts/video_dataset/run_pipeline.py --english-only --workers 4
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

STAGES = ["metadata", "splits", "download", "verify", "preprocess", "statistics"]


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_metadata(args) -> None:
    from generate_metadata import build_master
    cats = None
    if args.english_only:
        # Still fetch all categories; language filtering happens inside build_master
        # when config.LANGUAGE_FILTER is set at runtime.
        import config as cfg
        cfg.LANGUAGE_FILTER = ["EN"]
    build_master(categories=getattr(args, "categories", None))


def _run_splits(args) -> None:
    from create_splits import create_splits
    create_splits(seed=args.seed)


def _run_download(args) -> None:
    from download_videos import download_all
    lang = ["EN"] if args.english_only else None
    download_all(
        splits_filter=getattr(args, "splits_filter", None),
        labels_filter=getattr(args, "labels_filter", None),
        lang_filter=lang,
        workers=args.workers,
        dry_run=args.dry_run,
    )


def _run_verify(args) -> None:
    from verify_videos import run_verification
    run_verification(redownload_failed=False)


def _run_preprocess(args) -> None:
    from preprocess_videos import run_preprocessing
    run_preprocessing(normalize=args.normalize, workers=args.workers)


def _run_statistics(args) -> None:
    from dataset_statistics import generate_reports
    generate_reports(plots=not args.no_plots)


_STAGE_RUNNERS = {
    "metadata":   _run_metadata,
    "splits":     _run_splits,
    "download":   _run_download,
    "verify":     _run_verify,
    "preprocess": _run_preprocess,
    "statistics": _run_statistics,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(args) -> None:
    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        idx = STAGES.index(args.from_stage)
        stages_to_run = STAGES[idx:]
    else:
        stages_to_run = STAGES

    log.info("=" * 64)
    log.info("  VIDI Dataset Pipeline")
    log.info(f"  Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"  Stages    : {stages_to_run}")
    log.info(f"  English-only  : {args.english_only}")
    log.info(f"  Dry-run   : {args.dry_run}")
    log.info(f"  Workers   : {args.workers}")
    log.info("=" * 64 + "\n")

    pipeline_start = time.perf_counter()
    stage_times: dict[str, float] = {}

    for stage in stages_to_run:
        log.info(f"\n{'─'*64}")
        log.info(f"  STAGE [{STAGES.index(stage)+1}/{len(STAGES)}]: {stage.upper()}")
        log.info(f"{'─'*64}")
        t0 = time.perf_counter()
        try:
            _STAGE_RUNNERS[stage](args)
            elapsed = time.perf_counter() - t0
            stage_times[stage] = elapsed
            log.info(f"\n  ✓ Stage '{stage}' completed in {elapsed/60:.1f} min")
        except KeyboardInterrupt:
            log.warning(f"\n  ⚠ Pipeline interrupted at stage '{stage}'")
            log.info(f"  Resume with: python run_pipeline.py --from-stage {stage}")
            sys.exit(1)
        except Exception as exc:
            log.error(f"\n  ✗ Stage '{stage}' failed: {exc}", exc_info=True)
            log.info(f"  Fix the error then resume with: python run_pipeline.py --from-stage {stage}")
            sys.exit(2)

    total = time.perf_counter() - pipeline_start
    log.info("\n" + "=" * 64)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Total time : {total/60:.1f} min")
    log.info("\n  Stage breakdown:")
    for stage, t in stage_times.items():
        log.info(f"    {stage:<12}: {t/60:.1f} min")
    log.info("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="VIDI Video Dataset Pipeline — full end-to-end orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage control
    g = p.add_mutually_exclusive_group()
    g.add_argument("--stage",      choices=STAGES,
                   help="Run a single stage and stop")
    g.add_argument("--from-stage", choices=STAGES, dest="from_stage",
                   help="Resume pipeline from this stage (inclusive)")

    # Download options
    p.add_argument("--english-only", action="store_true",
                   help="Process only English-language clips (~47%% of dataset)")
    p.add_argument("--workers",  type=int, default=8,
                   help="Parallel download / processing workers (default: 8)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Simulate download stage without fetching data")

    # Preprocessing options
    p.add_argument("--normalize", action="store_true",
                   help="Preprocess stage: re-encode clips to 256×256 @ 16fps")

    # Statistics options
    p.add_argument("--no-plots", action="store_true",
                   help="Statistics stage: skip matplotlib chart generation")

    # Split control
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val/test split (default: 42)")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
