"""
gpu_queue.py — Serializes GPU inference across all models.

A single asyncio.Lock ensures only one model runs at a time on the GPU.
Without this, concurrent requests trigger OOM errors or severe slowdowns
because PyTorch models do not yield the GPU between forward passes.

Fix L: Added queue depth limit — returns HTTP 503 if more than
        MAX_QUEUE_DEPTH requests are already waiting for the lock.
Fix G: Added per-inference timeout (INFERENCE_TIMEOUT_S seconds).
        Raises asyncio.TimeoutError (caught upstream as 503) rather
        than hanging forever if a model stalls.
"""

import asyncio
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_GPU_LOCK = asyncio.Lock()

# Maximum number of requests that may wait concurrently for the GPU lock.
# If this depth is exceeded the server returns 503 immediately.
MAX_QUEUE_DEPTH: int = 3

# Hard timeout (seconds) on a single inference call.
# Prevents a stalled model from holding the lock forever.
INFERENCE_TIMEOUT_S: float = 600.0  # 10 minutes

# Track how many coroutines are currently queued (waiting + running).
_queue_depth: int = 0


async def run_with_gpu_lock(coro, model_name: str):
    """
    Serialize GPU inference through a single asyncio.Lock.

    Raises HTTP 503 immediately if MAX_QUEUE_DEPTH requests are already
    waiting, preventing unbounded memory growth under load.

    Wraps inference in asyncio.wait_for(INFERENCE_TIMEOUT_S) so a
    stalled model cannot hold the lock indefinitely.
    """
    global _queue_depth

    # ── Queue depth guard (Fix L) ────────────────────────────────────────────
    if _queue_depth >= MAX_QUEUE_DEPTH:
        logger.warning(
            "[GPU Queue] Rejected %s — queue depth %d >= limit %d",
            model_name, _queue_depth, MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server busy — {_queue_depth} requests already queued. "
                "Please retry in a moment."
            ),
        )

    _queue_depth += 1
    logger.info(
        "[GPU Queue] Queued %s — depth now %d", model_name, _queue_depth
    )

    try:
        async with _GPU_LOCK:
            logger.info("[GPU Queue] Lock acquired — %s starting", model_name)
            # ── Per-inference timeout (Fix G) ────────────────────────────────
            try:
                result = await asyncio.wait_for(coro, timeout=INFERENCE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error(
                    "[GPU Queue] %s timed out after %.0f s",
                    model_name, INFERENCE_TIMEOUT_S,
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"{model_name} inference timed out after "
                        f"{int(INFERENCE_TIMEOUT_S // 60)} minutes. "
                        "The model may be overloaded — please retry."
                    ),
                )
            logger.info("[GPU Queue] Lock released — %s complete", model_name)
            return result
    finally:
        _queue_depth -= 1
        logger.info(
            "[GPU Queue] Depth decremented — now %d after %s",
            _queue_depth, model_name,
        )
