# ShardSense

**Adaptive Data Sharding with Online Load Imbalance Prediction**

ShardSense optimizes distributed training throughput by dynamically moving data shards from slow workers ("stragglers") to fast workers in real-time.

## Features
- **Adaptive Sharding**: Automatically detects stragglers and rebalances load.
- **Online Learning**: Uses XGBoost to predict batch processing times.
- **Real-time Dashboard**: Monitor worker performance and shard movement live.
- **PyTorch Native**: Drop-in compatible with `torch.utils.data.Dataset`.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

2. **Run the Demos**
   
   **Option A: Simple Sequential Demo (2 Workers)**
   ```bash
   python demo_real.py
   ```
   
   **Option B: Parallel Production Demo (8 Workers)**
   Runs true concurrent threads to simulate a busy cluster.
   ```bash
   python demo_parallel.py
   ```

3. **Launch the Dashboard**
   In a separate terminal, watch the training progress live:
   ```bash
   streamlit run dashboard.py
   ```

## Architecture
- **Control Plane**: `shardsense.runtime` orchestrates the training loop.
- **Data Plane**: `ShardedDataset` provides virtual views of data indices.
- **Intelligence**: `GreedyResharder` uses XGBoost predictions to optimize assignments.

## Project Structure
- `shardsense/`: Core library.
- `demo_parallel.py`: Threaded 8-worker simulation.
- `dashboard.py`: Streamlit visualization.
