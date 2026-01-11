import os
import sqlite3

import pytest

from shardsense.telemetry.collector import MetricsCollector
from shardsense.telemetry.schema import ShardMetrics, WorkerMetrics

DB_PATH = "test_metrics.db"

@pytest.fixture
def collector():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    col = MetricsCollector(db_path=DB_PATH)
    yield col
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_collector_in_memory(collector):
    m = WorkerMetrics(1.0, 1, 50.0, 100.0, 10.0, 0.5, 200.0)
    collector.push_worker_metrics(m)
    
    assert len(collector.worker_history[1]) == 1
    assert collector.worker_history[1][0].cpu_util == 50.0

def test_collector_persistence(collector):
    m = WorkerMetrics(1.0, 1, 50.0, 100.0, 10.0, 0.5, 200.0)
    collector.push_worker_metrics(m)
    
    # Verify DB write
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM worker_metrics")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 1
    assert rows[0][1] == 1 # worker_id
    assert rows[0][2] == 50.0 # cpu

def test_shard_registration(collector):
    s = ShardMetrics(99, 10.0, 100.0, 0.5)
    collector.register_shard(s)
    
    assert collector.shard_registry[99].size_mb == 10.0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM shard_metadata WHERE shard_id=99")
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
