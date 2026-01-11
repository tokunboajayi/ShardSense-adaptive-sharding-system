from typing import List, Dict
import pandas as pd

class FeatureBuilder:
    def __init__(self):
        self.feature_columns = [
            "worker_id",
            "worker_io", "worker_cpu", 
            "shard_size", "shard_difficulty",
            "interaction_io_size"
        ]

    def build_features(self, raw_data: List[Dict]) -> pd.DataFrame:
        if not raw_data:
            return pd.DataFrame(columns=self.feature_columns)
            
        df = pd.DataFrame(raw_data)
        
        # Interaction features
        # Heuristic: Larger shards hurt IO-bound workers more
        df["interaction_io_size"] = df["shard_size"] / (df["worker_io"] + 1e-6)
        
        # Ensure column order
        return df[self.feature_columns]

    def build_labels(self, raw_data: List[Dict]) -> pd.Series:
        if not raw_data:
            return pd.Series()
        return pd.Series([d["target_batch_time"] for d in raw_data])
