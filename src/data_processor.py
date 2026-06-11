from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Handles data loading, cleaning, feature engineering, and scaling."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load dataset into a pandas DataFrame."""
        return pd.read_csv(data_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform dataset cleaning and filtering."""
        df = df.copy()

        # Replace missing placeholders
        df.replace("?", np.nan, inplace=True)

        # Drop unnecessary columns
        columns_to_drop = [
            "weight",
            "medical_specialty",
            "encounter_id",
            "patient_nbr",
            "payer_code",
        ]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=existing_columns, inplace=True)

        # Remove patients who cannot be readmitted
        excluded_discharge_ids = [11, 13, 14, 19, 20, 21]
        if "discharge_disposition_id" in df.columns:
            df = df[~df["discharge_disposition_id"].isin(excluded_discharge_ids)]

        # Fill missing categorical values
        fill_values = {
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "race": "Unknown",
        }
        for column, value in fill_values.items():
            if column in df.columns:
                df[column] = df[column].fillna(value)

        # Create binary target
        if "readmitted" in df.columns:
            df["target"] = df["readmitted"].apply(
                lambda value: 1 if value == "<30" else 0
            )
            df.drop(columns=["readmitted"], inplace=True)

        return df

    def map_diagnosis(self, code) -> str:
        """Map ICD-9 diagnosis codes into broader categories."""
        if pd.isnull(code) or code == "?":
            return "Other"

        try:
            code_str = str(code)
            if code_str.startswith(("V", "E")):
                return "Other"

            numeric_code = float(code)

            if 390 <= numeric_code <= 459 or numeric_code == 785:
                return "Circulatory"
            if 460 <= numeric_code <= 519 or numeric_code == 786:
                return "Respiratory"
            if 520 <= numeric_code <= 579 or numeric_code == 787:
                return "Digestive"
            if 250 <= numeric_code < 251:
                return "Diabetes"
            if 580 <= numeric_code <= 629 or numeric_code == 788:
                return "Urogenital"
            if 710 <= numeric_code <= 739:
                return "Musculoskeletal"
            if 800 <= numeric_code <= 999:
                return "Injury"

            return "Other"
        except (ValueError, TypeError):
            return "Other"

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering and encoding."""
        df = df.copy()

        # Map diagnosis groups
        if "diag_1" in df.columns:
            df["diag_1_group"] = df["diag_1"].apply(self.map_diagnosis)

        # Remove raw diagnosis columns
        diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
        existing_diagnosis_columns = [col for col in diagnosis_columns if col in df.columns]
        df.drop(columns=existing_diagnosis_columns, inplace=True)

        # Convert age ranges into numeric midpoint values
        age_mapping = {
            "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
            "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
        }
        if "age" in df.columns:
            df["age"] = df["age"].replace(age_mapping).astype(float)

        # One-hot encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        return df

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Prepare train and validation datasets."""
        df = self.clean_data(df)
        df = self.engineer_features(df)

        X = df.drop(columns=["target"])
        y = df["target"].values

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
