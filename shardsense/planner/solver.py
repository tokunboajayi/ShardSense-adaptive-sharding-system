import copy
from typing import Dict, List

from shardsense.model.predictor import RuntimePredictor
from shardsense.planner.cost import calculate_movement_cost


class GreedyResharder:
    """
    Iteratively moves shards from the slowest worker to the fastest worker
    until the cost (Time + MovementPenalty) stops improving.
    """
    def __init__(self, predictor: RuntimePredictor, movement_penalty_per_mb: float = 0.05):
        self.predictor = predictor
        self.penalty = movement_penalty_per_mb

    def plan(
        self, 
        current_map: Dict[int, List[int]], 
        worker_states: Dict[int, Dict], # info needed for prediction
        shard_states: Dict[int, Dict]
    ) -> Dict[int, List[int]]:
        
        # 1. Deep copy current map to start modifying
        best_map = copy.deepcopy(current_map)
        
        # Helper to calc expected time for a mapping
        def evaluate_map(assignment_map):
            times = {}
            for wid, sids in assignment_map.items():
                w_time = 0.0
                for sid in sids:
                    # Model prediction
                    w_time += self.predictor.predict_batch_time(
                        worker_states[wid], 
                        shard_states[sid]
                    )
                times[wid] = w_time
            return times

        current_times = evaluate_map(best_map)
        best_objective = max(current_times.values()) # Baseline: just time (no movement)
        
        # Iteration limit to prevent infinite loops
        for _ in range(50):
            # Identify straggler (max time) and receiver (min time)
            slowest_wid = max(current_times, key=current_times.get)
            fastest_wid = min(current_times, key=current_times.get)
            
            if slowest_wid == fastest_wid:
                break
                
            # Try moving each shard from slowest to fastest
            candidate_map = None
            candidate_objective = float('inf')
            
            # Simple heuristic: try moving the BIGGEST shard first to have impact
            # Or the SMALLEST if we just need fine tuning? 
            # Let's try all assigned shards.
            source_shards = best_map[slowest_wid]
            if not source_shards: 
                break

            found_improvement = False
            
            # Sort by predicted cost (descending) to aggressively fix straggler
            # But we need expensive calculation for that. Just try all.
            for sid in source_shards:
                # Create trial map
                trial_map = copy.deepcopy(best_map)
                trial_map[slowest_wid].remove(sid)
                trial_map[fastest_wid].append(sid)
                
                # Evaluate
                times = evaluate_map(trial_map)
                max_t = max(times.values())
                
                # Movement penalty
                # We calculate delta from ORIGINAL input map, not previous step
                move_cost = calculate_movement_cost(current_map, trial_map, {s: d['size_mb'] for s, d in shard_states.items()})
                
                total_obj = max_t + (move_cost * self.penalty)
                
                if total_obj < best_objective:
                    best_objective = total_obj
                    candidate_map = trial_map
                    found_improvement = True
                    # Greedy: take first improvement or best? 
                    # Let's take best to avoid oscillations, but for speed first is okay.
                    # We will keep searching this worker's shards to find BEST move.
                    if total_obj < candidate_objective:
                        candidate_objective = total_obj
                        candidate_map = trial_map

            if found_improvement and candidate_map:
                best_map = candidate_map
                current_times = evaluate_map(best_map)
            else:
                # No single move improves the objective
                break
                
        return best_map
