# ShardSense: Adaptive Data Sharding with Online Imbalance Prediction

## 1) Problem Definition & Assumptions

**Definitions:**
- **Shard**: A logical block of data samples (e.g., 64-256MB). It is the atomic unit of movement.
- **Worker**: A training node (GPU/TPU + CPU + IO) with variable throughput.
- **Imbalance**: Variance in "step time" (time to process a global batch) caused by the slowest worker (straggler).
- **Reshard Window**: The brief period at epoch boundaries where we can modify the worker->shard mapping.

**Assumptions:**
- **Heterogeneity**: Workers have different stable speeds (e.g., some are older generation, some share host IO).
- **Data Variability**: Different shards take different amounts of time to decode/process (e.g., video length, image resolution).
- **Network Cost**: Moving a shard from Worker A to Worker B takes time and bandwidth.

## 2) System Architecture

The system operates in two planes:

### Control Plane (The "Brain")
1.  **Metric Collector**: Aggregates per-epoch statistics from all workers.
2.  **Model Service**: Predicts `next_epoch_duration` for every possible (worker, shard) pair.
3.  **Planner**: Uses predictions to generate a `ShardMap` (Worker -> [Shard IDs]) that minimizes max(worker_duration) + rebalancing_cost.

### Data Plane (The "Muscle")
1.  **Dataloader Wrapper**: Measures detailed timing metrics (IO wait, decode time, compute time).
2.  **State Machine**:
    - `PHASE_TRAIN`: Normal execution. Telemetry is buffered.
    - `PHASE_SYNC`: At epoch end, send metrics to Control Plane.
    - `PHASE_RECONFIGURE`: Receive new `ShardMap` and adjust local shard list.

## 3) Data & Telemetry Schema

### `worker_metrics`
| Field | Type | Description |
|---|---|---|
| timestamp | int64 | Unix epoch ms |
| worker_id | int | Unique ID |
| cpu_util | float | 0-100% |
| io_read_mb_s | float | Disk bandwidth |
| batch_time_ms | float | Observed step time (P50/P95) |

### `shard_metrics`
| Field | Type | Description |
|---|---|---|
| shard_id | int | Unique ID |
| size_mb | float | data volume |
| mean_decode_ms | float | Complexity proxy |
| hotness_score | float | normalized access freq |

### `assignment_log`
- Used for creating training data: `(worker, shard, epoch) -> observed_duration`

## 4) Model Design

**V1 (MVP): XGBoost Regressor**
- **Objective**: Predict `mean_batch_time_ms` for `(worker_i, shard_j)`.
- **Features**:
    - `worker.rolling_mean_batch_time`: Is this worker historically slow?
    - `shard.size_mb`: Basic cost.
    - `shard.mean_decode_ms`: Historical complexity.
    - `worker.io_read_mb_s`: Current IO saturation.
- **Training**: Offline retraining every N epochs.

**V2 Upgrade**:
- Temporal GNN to model interference between shards on the same worker.

## 5) Resharding Strategy

**Constraint Check**:
- `Total_Movement_MB < MAX_BUDGET` (prevent network saturation).
- `Min_Shards < Count(Shards) < Max_Shards` (memory constraints).

**Algorithm (Greedy Straggler Relief)**:
1.  **Identify Stragglers**: Sort workers by predicted total epoch time.
2.  **Offload**: Take the most expensive shard from the slowest worker.
3.  **Assign**: Move it to the fastest worker *if*:
    - The fast worker doesn't become the new straggler.
    - Movement cost < Time saved.
4.  **Repeat** until convergence or budget exhausted.

## 6) APIs

### User-Facing (`runtime`)
```python
runtime = ShardSenseRuntime(control_plane_url="...")
runtime.register_dataset(dataset)

for epoch in range(epochs):
    runtime.start_epoch(epoch)
    # Get the shards assigned to this worker
    loader = runtime.dataloader(worker_id) 
    
    for batch in loader:
        train_step(batch)
        
    runtime.end_epoch(epoch) # Triggers telemetry push & polls for new plan
```

### Internal
- `Planner.plan(current_map, predictions, budget) -> new_map`
- `Model.predict(features) -> latencies`

## 9) Testing & Failure Injection
- **Deterministic Seeding**: Ensure simulated "random" slowdowns are reproducible.
- **Planner Budget**: Assert that `sum(moved_bytes) <= limit`.
- **Failure Modes**:
    1.  **Worker Slowdown**: Inject 2x latency on Worker 0. System should move shards away.
    2.  **Packet Loss**: Metric upload fails. System falls back to static sharding for that epoch.

## 10) Benchmark Plan
- **Baseline**: Static Round-Robin sharding.
- **Adaptive**: ShardSense enabled.
- **Metrics**:
    - `Speedup = Baseline_Total_Time / Adaptive_Total_Time`
    - `Straggler_Gap = Max_Worker_Time - Min_Worker_Time`

## 12) Future Extensions
- **Multi-node**: Replace local simulation with gRPC.
- **RL Planner**: Use PPO to learn long-term sharding policies rather than greedy greedy steps.
- **Preemption Handling**: Handle spot instance kills by replicating shards.
