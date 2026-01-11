
import torch
from torch.utils.data import DataLoader, TensorDataset

from shardsense.data.dataset import ShardedDataset
from shardsense.data.loader import MeasurableDataLoader
from shardsense.telemetry.collector import MetricsCollector


class MockCollector(MetricsCollector):
    def push_worker_metrics(self, metrics):
        self.last_metric = metrics

def test_sharded_dataset_indices_logic():
    # 100 samples, 10 shards -> 10 samples per shard
    data = torch.randn(100, 1)
    ds = TensorDataset(data)
    
    # Assign shards 0 and 2 (indices 0-9 and 20-29)
    assigned_shards = [0, 2]
    sharded = ShardedDataset(ds, assigned_shards, shard_size=10)
    
    assert len(sharded) == 20
    
    # Test Values
    # Index 0 of sharded -> Index 0 of original
    # Index 10 of sharded -> Index 20 of original
    assert torch.equal(sharded[0][0], ds[0][0])
    assert torch.equal(sharded[10][0], ds[20][0])

def test_dataset_boundary_conditions():
    # 25 samples, 10 shards -> 3 samples per shard (approx)
    # Shard 9 will be cut short
    data = torch.randn(25, 1)
    ds = TensorDataset(data)
    
    shard_size = 3
    
    # Shard 9: Start 27. Should be empty and not crash.
    sharded_empty = ShardedDataset(ds, [9], shard_size)
    assert len(sharded_empty) == 0
    
    # Shard 8: Start 24. End 27 clamped to 25. Should have 1 sample (index 24).
    sharded_partial = ShardedDataset(ds, [8], shard_size)
    assert len(sharded_partial) == 1

def test_measurable_loader():
    data = torch.randn(10, 1)
    ds = TensorDataset(data)
    loader = DataLoader(ds, batch_size=2)
    
    collector = MockCollector()
    m_loader = MeasurableDataLoader(loader, worker_id=1, collector=collector)
    
    batches = 0
    for batch in m_loader:
        batches += 1
        # metric should be pushed
        assert collector.last_metric.worker_id == 1
        assert collector.last_metric.batch_time_ms >= 0.0
        
    assert batches == 5
