from pathlib import Path
import json
import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

class ModelTrainer:
    """Handles model initialization, training, evaluation, and artifact saving."""

    def __init__(self, feature_names=None, scaler=None):
        self.feature_names = feature_names
        self.scaler = scaler

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> CatBoostClassifier:
        """Train CatBoost classifier."""
        class_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            scale_pos_weight=class_ratio,
            eval_metric="AUC",
            random_seed=42,
            verbose=100,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            use_best_model=True,
        )
        return model

    def evaluate_model(self, model: CatBoostClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate trained model performance."""
        y_probabilities = model.predict_proba(X_test)[:, 1]
        y_predictions = (y_probabilities > 0.5).astype(int)

        return {
            "roc_auc": roc_auc_score(y_test, y_probabilities),
            "confusion_matrix": confusion_matrix(y_test, y_predictions),
            "classification_report": classification_report(y_test, y_predictions),
        }

    def save_artifacts(self, model: CatBoostClassifier, output_dir: str = "models"):
        """Save the CatBoost model (.cbm), scaler, and feature list."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Saves specifically with the .cbm native CatBoost format
        model.save_model(output_path / "diabetes_model.cbm")

        if self.scaler:
            joblib.dump(self.scaler, output_path / "scaler.joblib")

        if self.feature_names:
            with open(output_path / "feature_names.json", "w", encoding="utf-8") as file:
                json.dump(self.feature_names, file, indent=4)
