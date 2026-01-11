from dataclasses import dataclass

@dataclass
class WorkerMetrics:
    timestamp: float
    worker_id: int
    cpu_util: float
    io_read_mb_s: float
    net_rtt_ms: float
    cache_hit_rate: float
    batch_time_ms: float # P50 or mean for the reporting window

@dataclass
class ShardMetrics:
    shard_id: int
    size_mb: float
    mean_decode_ms: float
    hotness_score: float # 0.0 to 1.0

@dataclass
class AssignmentLog:
    epoch: int
    worker_id: int
    shard_id: int
    start_time: float
    end_time: float
    mean_batch_time_ms: float
