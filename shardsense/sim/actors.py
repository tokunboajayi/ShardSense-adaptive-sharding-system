from dataclasses import dataclass


@dataclass
class Shard:
    """
    Represents a chunk of the dataset. 
    In a real system, this would point to a file path or range.
    """
    id: int
    size_mb: float
    difficulty_factor: float = 1.0  # Multiplier for decode/compute time (1.0 = baseline)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Worker:
    """
    Represents a training node.
    """
    id: int
    compute_speed: float = 1.0  # Multiplier: 1.0 = baseline, 0.8 = slower, 1.2 = faster
    io_bandwidth_mb_s: float = 100.0
    current_load_factor: float = 1.0 # Dynamic slowdown factor (1.0 = normal, >1.0 = throttled)

    def get_effective_speed(self) -> float:
        """Returns effective speed multiplier (higher is faster)."""
        return self.compute_speed / self.current_load_factor
