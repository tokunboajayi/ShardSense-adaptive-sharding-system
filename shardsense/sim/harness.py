import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from shardsense.sim.actors import Worker, Shard
from shardsense.telemetry.schema import WorkerMetrics, ShardMetrics

@dataclass
class SimulationState:
    epoch: int = 0
    step_times: List[float] = field(default_factory=list)
    worker_assignments: Dict[int, List[int]] = field(default_factory=dict) # worker_id -> [shard_ids]

class SimulationEngine:
    """
    Simulates the distributed training environment.
    Calculates step times based on assignments and worker/shard properties.
    """
    def __init__(self, workers: List[Worker], shards: List[Shard]):
        self.workers = {w.id: w for w in workers}
        self.shards = {s.id: s for s in shards}
        self.rng = random.Random(42)
        
        # Default strict round-robin assignment
        self.current_assignments: Dict[int, List[int]] = {w.id: [] for w in workers}
        shard_ids = sorted([s.id for s in shards])
        for i, sid in enumerate(shard_ids):
            worker_id = workers[i % len(workers)].id
            self.current_assignments[worker_id].append(sid)

    def set_assignments(self, new_map: Dict[int, List[int]]):
        """Apply a new sharding plan."""
        # Verification: all shards must be assigned exactly once
        assigned_ids = []
        for s_list in new_map.values():
            assigned_ids.extend(s_list)
        
        if sorted(assigned_ids) != sorted(self.shards.keys()):
            raise ValueError(f"Invalid assignment: Shard count mismatch. Expected {len(self.shards)}, got {len(assigned_ids)}")
            
        self.current_assignments = new_map

    def simulate_epoch(self, epoch_id: int) -> Dict[str, float]:
        """
        Runs one epoch. Returns aggregated stats.
        We simulate 'steps' by processing all assigned shards.
        The epoch time is determined by the slowest worker (straggler).
        """
        worker_times = {}
        
        # Inject transient noise
        for w in self.workers.values():
            # 10% chance of transient slowdown
            noise = self.rng.uniform(0.9, 1.1)
            slowdown = 1.5 if self.rng.random() < 0.1 else 1.0
            
            # Calculate total work execution time
            # Time = (Shard_Size / IO) + (Shard_Difficulty / (Compute * Load))
            assigned_shards = self.current_assignments[w.id]
            total_time = 0.0
            
            for sid in assigned_shards:
                shard = self.shards[sid]
                
                # IO Time
                t_io = shard.size_mb / w.io_bandwidth_mb_s
                
                # Compute Time (arbitrary baseline constant 100ms per unit of difficulty)
                # Modified by worker speed
                base_compute_ms = 100.0 * shard.difficulty_factor
                effective_speed = w.compute_speed / (w.current_load_factor * slowdown)
                t_compute = (base_compute_ms / effective_speed) / 1000.0 # to seconds
                
                total_time += (t_io + t_compute) * noise
            
            worker_times[w.id] = total_time

        max_time = max(worker_times.values())
        min_time = min(worker_times.values())
        mean_time = sum(worker_times.values()) / len(worker_times)
        
        return {
            "epoch": epoch_id,
            "max_time": max_time,
            "min_time": min_time,
            "mean_time": mean_time,
            "straggler_gap": max_time - min_time,
            "worker_times": worker_times
        }

    def inject_failure(self, worker_id: int, slowdown_factor: float):
        """Permanently slow down a worker."""
        if worker_id in self.workers:
            self.workers[worker_id].current_load_factor = slowdown_factor
