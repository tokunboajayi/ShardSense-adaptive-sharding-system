import sqlite3
from typing import Any, Dict, List, Optional

from shardsense.telemetry.schema import AssignmentLog, ShardMetrics, WorkerMetrics


class MetricsCollector:
    """
    Aggregates per-epoch metrics from all workers.
    Supports optional SQLite persistence for dashboarding.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.worker_history: Dict[int, List[WorkerMetrics]] = {}
        self.shard_registry: Dict[int, ShardMetrics] = {}
        self.assignment_logs: List[AssignmentLog] = []
        
        if self.db_path:
            self._init_db()

    def _init_db(self):
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Worker Metrics Table
        c.execute('''CREATE TABLE IF NOT EXISTS worker_metrics (
            timestamp REAL,
            worker_id INTEGER,
            cpu_util REAL,
            io_read_mb_s REAL,
            batch_time_ms REAL
        )''')
        
        # Shard Registry
        c.execute('''CREATE TABLE IF NOT EXISTS shard_metadata (
            shard_id INTEGER PRIMARY KEY,
            size_mb REAL,
            hotness REAL
        )''')
        
        # Assignments
        c.execute('''CREATE TABLE IF NOT EXISTS assignments (
            epoch INTEGER,
            worker_id INTEGER,
            shard_id INTEGER,
            batch_time_ms REAL
        )''')
        
        conn.commit()
        conn.close()

    def register_shard(self, shard: ShardMetrics):
        self.shard_registry[shard.shard_id] = shard
        db_path = self.db_path
        if db_path:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO shard_metadata (shard_id, size_mb, hotness) VALUES (?, ?, ?)',
                    (shard.shard_id, shard.size_mb, shard.hotness_score)
                )

    def push_worker_metrics(self, metrics: WorkerMetrics):
        # 1. In-memory buffer for Model
        if metrics.worker_id not in self.worker_history:
            self.worker_history[metrics.worker_id] = []
        self.worker_history[metrics.worker_id].append(metrics)
        
        # 2. Persistence for Dashboard
        if self.db_path:
            # Type narrowing via local assignment if needed, but simple check usually works. 
            # Doing safe cast just in case.
            path = self.db_path
            with sqlite3.connect(path) as conn:
                conn.execute(
                    'INSERT INTO worker_metrics (timestamp, worker_id, cpu_util, io_read_mb_s, batch_time_ms) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (
                        metrics.timestamp, metrics.worker_id, metrics.cpu_util, 
                        metrics.io_read_mb_s, metrics.batch_time_ms
                    )
                )

    def log_assignment(self, log: AssignmentLog):
        self.assignment_logs.append(log)
        if self.db_path:
             path = self.db_path
             with sqlite3.connect(path) as conn:
                conn.execute(
                    'INSERT INTO assignments (epoch, worker_id, shard_id, batch_time_ms) VALUES (?, ?, ?, ?)',
                    (log.epoch, log.worker_id, log.shard_id, log.mean_batch_time_ms)
                )

    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Joins assignment logs with worker/shard stats to create training rows.
        """
        data = []
        for log in self.assignment_logs:
            shard_meta = self.shard_registry.get(log.shard_id)
            if not shard_meta: 
                continue
                
            worker_metrics_list = self.worker_history.get(log.worker_id, [])
            last_wm = worker_metrics_list[-1] if worker_metrics_list else None
            
            row = {
                "worker_id": log.worker_id,
                "shard_id": log.shard_id,
                "target_batch_time": log.mean_batch_time_ms,
                "shard_size": shard_meta.size_mb,
                "shard_difficulty": shard_meta.mean_decode_ms,
            }
            
            if last_wm:
                row.update({
                    "worker_io": last_wm.io_read_mb_s,
                    "worker_cpu": last_wm.cpu_util
                })
            else:
                row.update({
                    "worker_io": 100.0,
                    "worker_cpu": 0.5
                })
            data.append(row)
        return data
