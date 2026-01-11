import argparse
import random

import matplotlib.pyplot as plt

from shardsense.runtime.engine import ShardSenseRuntime
from shardsense.sim.actors import Shard, Worker
from shardsense.sim.harness import SimulationEngine


def create_environment(num_workers: int, num_shards: int) -> SimulationEngine:
    workers = []
    # Heterogeneous workers: 20% slow, 10% fast, 70% normal
    for i in range(num_workers):
        r = random.random()
        if r < 0.2:
            speed = 0.6 # Slow older node
        elif r > 0.9:
            speed = 1.3 # Fast new node
        else:
            speed = 1.0
        workers.append(Worker(id=i, compute_speed=speed, io_bandwidth_mb_s=100 + random.randint(-20, 20)))

    shards = []
    for i in range(num_shards):
        # Variable shard sizes
        size = 100 + random.randint(-50, 50)
        # Variable complexity
        diff = random.uniform(0.8, 1.5)
        shards.append(Shard(id=i, size_mb=size, difficulty_factor=diff))

    return SimulationEngine(workers, shards)

def run_simulation(args):
    print(f"--- Starting Simulation: {args.workers} Workers, {args.shards} Shards, {args.epochs} Epochs ---")
    
    # 1. Baseline Run (Static)
    print("\nRunning Baseline (Static Sharding)...")
    sim_base = create_environment(args.workers, args.shards)
    # Inject specific failure to test straggler handling
    sim_base.inject_failure(worker_id=0, slowdown_factor=2.0)
    
    runtime_base = ShardSenseRuntime(sim_base)
    baseline_times = []
    
    for e in range(args.epochs):
        stats = runtime_base.run_epoch(e, adaptive=False)
        baseline_times.append(stats["max_time"])
        print(f"Epoch {e}: Max Time={stats['max_time']:.2f}s (Straggler Gap={stats['straggler_gap']:.2f}s)")

    # 2. Adaptive Run
    print("\nRunning ShardSense (Adaptive Sharding)...")
    # Re-create mostly identical env (random seed in create_env handles consistency if we set it)
    random.seed(42) # Reset seed for fair comparison
    sim_adapt = create_environment(args.workers, args.shards)
    sim_adapt.inject_failure(worker_id=0, slowdown_factor=2.0)
    
    runtime_adapt = ShardSenseRuntime(sim_adapt)
    adaptive_times = []
    
    for e in range(args.epochs):
        stats = runtime_adapt.run_epoch(e, adaptive=True)
        adaptive_times.append(stats["max_time"])
        print(f"Epoch {e}: Max Time={stats['max_time']:.2f}s (Straggler Gap={stats['straggler_gap']:.2f}s)")

    # 3. Report
    mean_base = sum(baseline_times) / len(baseline_times)
    mean_adapt = sum(adaptive_times) / len(adaptive_times)
    improvement = (mean_base - mean_adapt) / mean_base * 100
    
    print("\n--- RESULTS ---")
    print(f"Baseline Mean Epoch Time: {mean_base:.2f}s")
    print(f"Adaptive Mean Epoch Time: {mean_adapt:.2f}s")
    print(f"Improvement: {improvement:.2f}%")
    
    if args.plot:
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_times, label='Baseline (Static)', marker='o', linestyle='--')
        plt.plot(adaptive_times, label='ShardSense (Adaptive)', marker='x', linewidth=2)
        plt.title('Epoch Step Time: Baseline vs ShardSense')
        plt.ylabel('Max Worker Time (s)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('benchmark_plot.png')
        print("Plot saved to benchmark_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShardSense Simulation CLI")
    parser.add_argument("command", choices=["simulate"], help="Command to run")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--shards", type=int, default=128, help="Number of shards")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to simulate")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        run_simulation(args)
