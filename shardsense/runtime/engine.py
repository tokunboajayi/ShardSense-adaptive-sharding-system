from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from shardsense.telemetry.collector import MetricsCollector
from shardsense.telemetry.schema import ShardMetrics
from shardsense.model.predictor import RuntimePredictor
from shardsense.planner.solver import GreedyResharder
from shardsense.data.dataset import ShardedDataset
from shardsense.data.loader import MeasurableDataLoader

class ShardSenseRuntime:
    """
    Real-world Runtime implementation.
    Manages sharding for a PyTorch Dataset.
    """
    def __init__(self, 
                 dataset: Dataset, 
                 num_shards: int,
                 num_workers: int = 1, # For MVP, simple config
                 batch_size: int = 32,
                 db_path: Optional[str] = None):
        self.dataset = dataset
        self.num_shards = num_shards
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Calculate shard size (virtual)
        self.total_samples = len(dataset)
        self.shard_size = (self.total_samples + num_shards - 1) // num_shards
        
        self.collector = MetricsCollector(db_path=db_path)
        self.predictor = RuntimePredictor()
        self.planner = GreedyResharder(self.predictor)
        
        # Initial Round Robin assignments
        self.assignments: Dict[int, List[int]] = {i: [] for i in range(num_workers)}
        for i in range(num_shards):
            worker_id = i % num_workers
            self.assignments[worker_id].append(i)
            
        # Register shards
        for i in range(num_shards):
            # approximate size in MB? Hard with generic dataset.
            # We use "count" as proxy for now or 1.0
            self.collector.register_shard(ShardMetrics(
                shard_id=i,
                size_mb=1.0, 
                mean_decode_ms=10.0,
                hotness_score=1.0
            ))

    def get_dataloader(self, worker_id: int) -> MeasurableDataLoader:
        """
        Returns a DataLoader for the specific worker based on current plan.
        """
        assigned_shards = self.assignments.get(worker_id, [])
        sharded_ds = ShardedDataset(self.dataset, assigned_shards, self.shard_size)
        
        # Fix for potential empty dataset with shuffle=True causing crashes
        should_shuffle = True
        if len(sharded_ds) == 0:
            should_shuffle = False

        loader = DataLoader(sharded_ds, batch_size=self.batch_size, shuffle=should_shuffle)
        return MeasurableDataLoader(loader, worker_id, self.collector)

    def epoch_end(self, epoch_id: int):
        """
        Triggered at the end of an epoch to potentially re-shard.
        """
        # Train model
        training_data = self.collector.get_training_data()
        if len(training_data) > 50:
             self.predictor.train(training_data)
        
        # Re-plan
        self._rebalance()

    def _rebalance(self):
        # Construct state for planner
        worker_states = {}
        for w in range(self.num_workers):
            # In real distributed system, we'd query the worker's metrics from the collector
            # aggregate recent history to get 'current' capabilities
            history = self.collector.worker_history.get(w, [])
            if history:
                 last = history[-1]
                 io = last.io_read_mb_s if last.io_read_mb_s > 0 else 100.0
                 cpu = last.cpu_util
            else:
                 io = 100.0
                 cpu = 0.5
                 
            worker_states[w] = {
                "worker_id": w,
                "io_read_mb_s": io,
                "cpu_util": cpu
            }
            
        shard_states = {}
        for i in range(self.num_shards):
            shard_states[i] = {
                "shard_id": i,
                "size_mb": 1.0, # Placeholder
                "mean_decode_ms": 10.0 # Placeholder
            }

        new_map = self.planner.plan(
            self.assignments,
            worker_states,
            shard_states
        )
        
        self.assignments = new_map
        # In distributed setting, we would broadcast this map.
        # Here we just update local state since we generate dataloaders on demand.
