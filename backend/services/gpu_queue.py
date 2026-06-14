"""
gpu_queue.py — Serializes GPU inference across all models.

A single asyncio.Lock ensures only one model runs at a time on the GPU.
Without this, concurrent requests trigger OOM errors or severe slowdowns
because PyTorch models do not yield the GPU between forward passes.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_GPU_LOCK = asyncio.Lock()


async def run_with_gpu_lock(coro, model_name: str):
    """Ensures only one model inference runs at a time."""
    logger.info(f"[GPU Queue] Waiting for lock — {model_name}")
    async with _GPU_LOCK:
        logger.info(f"[GPU Queue] Lock acquired — {model_name} starting")
        result = await coro
        logger.info(f"[GPU Queue] Lock released — {model_name} complete")
        return result
