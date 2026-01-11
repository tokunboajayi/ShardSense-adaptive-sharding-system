import time
from typing import Any, Iterator

from torch.utils.data import DataLoader

from shardsense.telemetry.collector import MetricsCollector, WorkerMetrics


class MeasurableDataLoader:
    """
    Wraps a standard PyTorch DataLoader.
    Iterating over this records timing metrics.
    """
    def __init__(self, loader: DataLoader, worker_id: int, collector: MetricsCollector):
        self.loader = loader
        self.worker_id = worker_id
        self.collector = collector
        self._iterator = None

    def __iter__(self) -> Iterator[Any]:
        self._iterator = iter(self.loader)
        return self

    def __next__(self) -> Any:
        start_t = time.perf_counter()
        try:
            batch = next(self._iterator)
            end_t = time.perf_counter()
            duration_ms = (end_t - start_t) * 1000.0
            
            # Log immediate batch time (simplified; in prod we might average over N batches or send raw stream)
            self.collector.push_worker_metrics(WorkerMetrics(
                timestamp=time.time(),
                worker_id=self.worker_id,
                cpu_util=0.0, # Placeholder or need psutil
                io_read_mb_s=0.0,
                net_rtt_ms=0.0,
                cache_hit_rate=0.0,
                batch_time_ms=duration_ms
            ))
            
            return batch
        except StopIteration:
            raise
