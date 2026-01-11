from typing import List, Iterator
import torch
from torch.utils.data import Dataset, Subset

class ShardedDataset(Dataset):
    """
    Wraps a standard PyTorch Dataset and exposes only the samples
    belonging to the list of assigned Shard IDs.
    
    Each 'Shard' is conceptually a range of indices in the original dataset.
    """
    def __init__(self, source_dataset: Dataset, assigned_shard_ids: List[int], shard_size: int):
        self.source = source_dataset
        self.assigned_shards = assigned_shard_ids
        self.shard_size = shard_size
        self.valid_indices = self._build_indices()
        # print(f"Debug ShardedDataset: Shards={assigned_shard_ids}, Total={len(self.valid_indices)}")

    def _build_indices(self) -> List[int]:
        """Calculates all global indices belonging to assigned shards."""
        indices = []
        total_len = len(self.source)
        
        for sid in self.assigned_shards:
            start_idx = sid * self.shard_size
            end_idx = start_idx + self.shard_size
            
            # Clamp to dataset bounds
            if start_idx >= total_len:
                continue
            
            # Ensure we don't exceed dataset valid range
            end_idx = min(end_idx, total_len)
            
            if start_idx < end_idx:
                 indices.extend(range(start_idx, end_idx))
            
        return indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        global_idx = self.valid_indices[idx]
        return self.source[global_idx]
