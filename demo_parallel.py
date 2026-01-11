import threading
import time

import torch
from torch.utils.data import TensorDataset

from shardsense.runtime.engine import ShardSenseRuntime
from shardsense.telemetry.schema import AssignmentLog

# Lock for printing to console without overlapping text
print_lock = threading.Lock()

def worker_routine(w_id: int, runtime: ShardSenseRuntime, epoch: int):
    """
    Function meant to run in a separate thread.
    Simulates one worker processing its assigned shards.
    """
    dl = runtime.get_dataloader(w_id)
    w_start = time.time()
    
    batch_count = 0
    for batch in dl:
        # Simulate Compute
        _ = batch[0] * 2 
        batch_count += 1
        
        # Artificial Slowdown for specific workers
        # Worker 1 and 5 are "slow nodes"
        if w_id in [1, 5]:
            time.sleep(0.005) 
            
    w_end = time.time()
    duration = w_end - w_start
    
    with print_lock:
        # Only print occasionally to avoid spam
        if w_id == 0 or w_id == 1: 
            print(f"  [Worker {w_id}] Epoch {epoch}: {batch_count} batches in {duration:.2f}s")
    
    # Log assignment outcome
    assigned_shards = runtime.assignments.get(w_id, [])
    if assigned_shards:
        per_shard = duration / len(assigned_shards) * 1000.0 
        for sid in assigned_shards:
            runtime.collector.log_assignment(AssignmentLog(
                epoch=epoch,
                worker_id=w_id,
                shard_id=sid,
                start_time=w_start,
                end_time=w_end,
                mean_batch_time_ms=per_shard
            ))
            
    return duration

def run_parallel_demo():
    print("Generating Synthetic Data (20,000 samples)...")
    X = torch.randn(20000, 100)
    y = torch.randint(0, 2, (20000,))
    dataset = TensorDataset(X, y)
    
    # 8 Workers, 64 Shards
    NUM_WORKERS = 8
    NUM_SHARDS = 64
    EPOCHS = 15
    
    print(f"Starting {NUM_WORKERS} Concurrent Workers...")
    
    runtime = ShardSenseRuntime(
        dataset, 
        num_shards=NUM_SHARDS, 
        num_workers=NUM_WORKERS, 
        batch_size=32, 
        db_path="shardsense.db"
    )
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
        epoch_start = time.time()
        
        threads = []
        
        # Spawn threads
        for w_id in range(NUM_WORKERS):
            t = threading.Thread(target=worker_routine, args=(w_id, runtime, epoch))
            threads.append(t)
            t.start()
            
        # Wait for all workers to finish
        for t in threads:
            t.join()

        # Runtime hook at end of epoch (Model training + Rebalancing)
        runtime.epoch_end(epoch)
        
        # Check assignments for top/bottom workers
        with print_lock:
            # Simple stats check
            counts = {w: len(s) for w, s in runtime.assignments.items()}
            print(f"Shard Counts: {counts}")

if __name__ == "__main__":
    run_parallel_demo()
