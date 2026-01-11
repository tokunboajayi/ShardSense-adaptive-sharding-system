from typing import Dict, List


def calculate_imbalance_cost(worker_times: Dict[int, float]) -> float:
    """
    Cost = Max Time (Straggler) - Min Time (Fastest)
    Or simply Max Time, since that determines epoch duration.
    """
    if not worker_times:
        return 0.0
    return max(worker_times.values())

def calculate_movement_cost(
    old_map: Dict[int, List[int]], 
    new_map: Dict[int, List[int]],
    shard_sizes: Dict[int, float]
) -> float:
    """
    Cost = Sum of MB moved * Cost_Per_MB (e.g., bandwidth latency).
    For simulation, we assume arbitrary cost units.
    """
    moved_mb = 0.0
    
    # Invert maps to find shard locations
    old_locs = {}
    for wid, sids in old_map.items():
        for sid in sids:
            old_locs[sid] = wid
            
    for wid, sids in new_map.items():
        for sid in sids:
            prev_wid = old_locs.get(sid)
            if prev_wid != wid and prev_wid is not None:
                # Shard moved
                moved_mb += shard_sizes.get(sid, 0.0)
                
    return moved_mb
