from typing import Dict, List, Any
import pandas as pd
from shardsense.model.features import FeatureBuilder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not found. Falling back to heuristic model.")

class RuntimePredictor:
    def __init__(self):
        self.model = None
        if HAS_XGB:
            self.model = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.1, 
                n_jobs=1
            )
        self.features = FeatureBuilder()
        self.is_trained = False

    def train(self, training_data: List[Dict]):
        if len(training_data) < 50:
            # print("Not enough data to train (need 50+ samples).")
            return
            
        if not HAS_XGB:
            return

        X = self.features.build_features(training_data)
        y = self.features.build_labels(training_data)
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict_batch_time(self, worker_state: Dict, shard_state: Dict) -> float:
        """
        Returns predicted ms for a single pairing.
        """
        if not self.is_trained:
            # Fallback heuristic if untrained
            # Time = Size / IO + 100 * Diff
            return (shard_state["size_mb"] / worker_state["io_read_mb_s"]) * 1000 + (100 * shard_state["mean_decode_ms"])

        # Construct single row
        row = {
            "worker_id": worker_state["worker_id"],
            "shard_id": shard_state["shard_id"],
            "worker_io": worker_state["io_read_mb_s"],
            "worker_cpu": worker_state["cpu_util"],
            "shard_size": shard_state["size_mb"],
            "shard_difficulty": shard_state["mean_decode_ms"],
            # dummy target, not used for prediction
            "target_batch_time": 0 
        }
        
        X = self.features.build_features([row])
        pred = self.model.predict(X)[0]
        return max(1.0, float(pred))
