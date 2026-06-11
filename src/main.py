from pathlib import Path
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Initialize preprocessing engine
    processor = DataProcessor()

    print("Loading dataset...")
    dataframe = processor.load_data(data_path="./diabetic_data.csv")

    print("Preparing data...")
    X_train, X_test, y_train, y_test = processor.prepare_data(dataframe)

    # 2. Hand over metadata to the training module
    trainer = ModelTrainer(
        feature_names=processor.feature_names, 
        scaler=processor.scaler
    )

    print("Training model...")
    model = trainer.train_model(X_train, y_train, X_test, y_test)

    print("Evaluating model...")
    results = trainer.evaluate_model(model, X_test, y_test)

    print(f"\nROC-AUC Score: {results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    print(f"\nClassification Report:\n{results['classification_report']}")

    print("\nSaving artifacts...")
    trainer.save_artifacts(model)

    print("\nTraining pipeline completed successfully.")
