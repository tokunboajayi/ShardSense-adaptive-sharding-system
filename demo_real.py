import time

import torch
from torch.utils.data import TensorDataset

from shardsense.runtime.engine import ShardSenseRuntime
from shardsense.telemetry.schema import AssignmentLog


def run_real_demo():
    print("Generating Synthetic Data (10,000 samples)...")
    # Synthetic Data: 10k samples, Features=100
    X = torch.randn(10000, 100)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    
    # 2 'Workers' (simulated by sequential processing in this loop)
    # 20 Shards
    NUM_WORKERS = 2
    NUM_SHARDS = 20
    EPOCHS = 10
    
    runtime = ShardSenseRuntime(
        dataset, 
        num_shards=NUM_SHARDS, 
        num_workers=NUM_WORKERS, 
        batch_size=32, 
        db_path="shardsense.db"
    )
    
    # Inject Artificial "Slowdown" for Worker 1
    # We simulate this by sleeping in the loop if worker_id == 1
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
        print(f"\n--- Epoch {epoch} ---")
        
        worker_times = {}
        
        # Simulate distributed training by running each worker's dataloader one by one
        for w_id in range(NUM_WORKERS):
            dl = runtime.get_dataloader(w_id)
            w_start = time.time()
            
            # Training Loop
            batch_count = 0
            for batch in dl:
                # Simulate Compute
                _ = batch[0] * 2 
                batch_count += 1
                
                # Artificial Slowdown for Worker 1
                if w_id == 1:
                    time.sleep(0.002) # 2ms extra per batch
                    
            w_end = time.time()
            duration = w_end - w_start
            worker_times[w_id] = duration
            print(f"Worker {w_id}: Processed {batch_count} batches in {duration:.2f}s")
            
            # Manually log assignment outcome for this MVP (in prod, the MeasurableDataLoader/Worker does this)
            # We need to tell the system "how long" the shards took effectively.
            # Distribute duration across assigned shards
            assigned_shards = runtime.assignments[w_id]
            if assigned_shards:
                per_shard = duration / len(assigned_shards) * 1000.0 # ms
                for sid in assigned_shards:
                    runtime.collector.log_assignment(AssignmentLog(
                        epoch=epoch,
                        worker_id=w_id,
                        shard_id=sid,
                        start_time=w_start,
                        end_time=w_end,
                        mean_batch_time_ms=per_shard # Approximation
                    ))

        # Runtime hook at end of epoch
        runtime.epoch_end(epoch)
        
        max_time = max(worker_times.values())
        print(f"Epoch Duration (Straggler): {max_time:.2f}s")
        
        # Check assignments
        counts = {w: len(s) for w, s in runtime.assignments.items()}
        print(f"Shard Distribution: {counts}")

if __name__ == "__main__":
    run_real_demo()
