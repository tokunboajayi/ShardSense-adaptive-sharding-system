from shardsense.model.predictor import RuntimePredictor
from shardsense.planner.solver import GreedyResharder


class MockPredictor(RuntimePredictor):
    def predict_batch_time(self, w_state, s_state):
        # Heuristic: Worker IO * Shard Difficulty
        return (100.0 / w_state["io_read_mb_s"]) * s_state["mean_decode_ms"]

def test_greedy_resharder_basic():
    predictor = MockPredictor()
    planner = GreedyResharder(predictor)
    
    # 2 Workers. Worker 0 is fast (IO=200). Worker 1 is slow (IO=10).
    worker_states = {
        0: {"worker_id": 0, "io_read_mb_s": 200.0, "cpu_util": 0.5},
        1: {"worker_id": 1, "io_read_mb_s": 10.0, "cpu_util": 0.5}
    }
    
    # 2 Shards. Identical.
    shard_states = {
        0: {"shard_id": 0, "size_mb": 100, "mean_decode_ms": 1.0},
        1: {"shard_id": 1, "size_mb": 100, "mean_decode_ms": 1.0}
    }
    
    # Initial: Both on Slow Worker 1
    current_map = {0: [], 1: [0, 1]}
    
    # Plan
    new_map = planner.plan(current_map, worker_states, shard_states)
    
    # Expectation: Should move at least one shard to Worker 0
    assert len(new_map[0]) > 0
    assert len(new_map[1]) < 2
