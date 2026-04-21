from __future__ import annotations

from loguru import logger

import torch


class ProfileEvents:
    def __init__(self):
        # sequential events
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc_after_encode = torch.cuda.Event(enable_timing=True)
        self.toc_after_denoise = torch.cuda.Event(enable_timing=True)
        self.toc_after_decode = torch.cuda.Event(enable_timing=True)
        self.toc_after_finalize = torch.cuda.Event(enable_timing=True)

    def summary(self) -> dict[str, float]:
        return {
            "elapsed_time_encode": self.tic.elapsed_time(self.toc_after_encode),
            "elapsed_time_denoise": self.toc_after_encode.elapsed_time(
                self.toc_after_denoise
            ),
            "elapsed_time_decode": self.toc_after_denoise.elapsed_time(
                self.toc_after_decode
            ),
            "elapsed_time_finalize": self.toc_after_decode.elapsed_time(
                self.toc_after_finalize
            ),
            "time_to_decode": self.tic.elapsed_time(self.toc_after_decode),
            "time_to_finalize": self.tic.elapsed_time(self.toc_after_finalize),
        }

    @staticmethod
    def finalize(events: list[ProfileEvents], skip_first_n: int = 0) -> None:
        if skip_first_n > 0:
            events = events[skip_first_n:]

        n = len(events)

        ts = []
        for event in events:
            ts.append(event.summary())

        elapsed_time_encode = sum(t["elapsed_time_encode"] for t in ts)
        elapsed_time_denoise = sum(t["elapsed_time_denoise"] for t in ts)
        elapsed_time_decode = sum(t["elapsed_time_decode"] for t in ts)
        elapsed_time_finalize = sum(t["elapsed_time_finalize"] for t in ts)
        time_to_decode = sum(t["time_to_decode"] for t in ts)
        time_to_finalize = sum(t["time_to_finalize"] for t in ts)

        def perc1(t):
            return f"({t / time_to_decode * 100:06.3f}%)"

        logger.info(
            f"Profiling results for {n} events after skipping first {skip_first_n} events:"
        )
        logger.info(f"Average Latency to Decode: {time_to_decode / n / 1000.0} seconds")
        logger.info(
            f"   ├─{perc1(elapsed_time_encode)} VAE encode HD map {elapsed_time_encode / n:.4f} ms"
        )
        logger.info(
            f"   ├─{perc1(elapsed_time_denoise)} DiT denoise latent {elapsed_time_denoise / n:.4f} ms"
        )
        logger.info(
            f"   ╰─{perc1(elapsed_time_decode)} VAE decode {elapsed_time_decode / n:.4f} ms"
        )
        logger.info(
            f"Average Latency to Finalize: {time_to_finalize / n / 1000.0} seconds"
        )
        logger.info(f"   ╰─finalize KV cache {elapsed_time_finalize / n:.4f} ms")
