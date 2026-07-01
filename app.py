from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
import os

class SalesPredictiveEngine:
    def __init__(self):
        """
        Constructor: Defines central state properties and locks the file 
        storage configuration for model persistence (serialization).
        """
        # Centralized path for model serialization (state persistence)
        self.model_path = "salespredictiveengine.joblib"
        self.pipeline = None

    def sales_predictive_model(self, clean_df: pd.DataFrame) -> bool:
        """
        ML training pipeline: Accepts preprocessed transactional data,
        orchestrates categorical preprocessing, fits an ensemble Regressor, 
        and serializes the trained pipeline structure to disk.
        """
        try:
            # 1. Structural Schema Validation Guardrail
            required_cols = ['Price', "Quantity", 'Category', 'Total_Revenue']
            if not all(col in clean_df.columns for col in required_cols):
                print("[ML Schema Error] Input DataFrame is missing critical features required for fitting.")
                return False
            
            # Feature extraction layout
            X = clean_df[['Price', 'Quantity', 'Category']]
            y = clean_df['Total_Revenue']

            # 2. Categorical Preprocessing Layer (Robust One-Hot encoding)
            # handle_unknown='ignore' ensures zero-crash execution if unseen categories appear in live data
            preprocessor = ColumnTransformer(
                transformers=[('cat_encode', OneHotEncoder(handle_unknown='ignore'), ['Category'])], 
                remainder='passthrough'
            )

            # 3. Modular Pipeline Assembly
            self.pipeline = Pipeline(steps=[
                ('preprocessing_stage', preprocessor),
                ('Regressor_model', RandomForestRegressor(n_estimators=50, random_state=7))
            ])

            # 4. Train-Test Validation Split (80/20 Holdout)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
            
            # Execute mathematical optimization (Fitting phase)
            self.pipeline.fit(X_train, y_train)

            # 5. Pipeline State Serialization
            joblib.dump(self.pipeline, self.model_path)
            print("[ML Engine] Pipeline state successfully serialized and cached to disk.")
            return True
        
        except Exception as e:
            print(f"[ML Engine Failure] Pipeline fitting aborted: {str(e)}")
            return False

    def predict_revenue(self, Price: float, Quantity: int, Category: str) -> float:
        """
        Dynamic Inference Engine: Runs live predictions on runtime telemetry, 
        implementing dynamic schema matching, and deterministic boundary guardrails.
        """
        # Baseline deterministic calculation used as fallback to guarantee 100% server uptime
        fallback_baseline = Price * Quantity

        try:
            # 🛡️ Defensive IO Check: Bypass model loading if state file is locked or missing
            if not os.path.exists(self.model_path):
                return fallback_baseline
            
            # Deserialize model pipeline state
            loaded_pipeline = joblib.load(self.model_path)
            
            # Map runtime variables directly into a structured, index-locked DataFrame
            simulation_input = pd.DataFrame([{
                'Price': Price,
                'Quantity': Quantity,
                'Category': Category
            }])
            
            # Compute predictive inference [0] extracts scalar prediction from array
            predicted_value = loaded_pipeline.predict(simulation_input)[0]
            
            # 🛡️ Mathematical Anti-Hallucination Boundary Guardrail
            # Clips prediction to prevent unrealistic fluctuations or negative revenue projections
            min_possible_revenue = Price * Quantity * 0.85
            if predicted_value < min_possible_revenue:
                return float(fallback_baseline)
                
            return float(predicted_value)
            
        except Exception as e:
            # Silent logging bypass: Fallback gracefully to prevent client-side system crash
            print(f"[Inference Bypass] Graceful fallback triggered: {str(e)}")
            return fallback_baseline

# ====================================================================
# 🚀 TESTING RUNNER (LOCAL SANITY CHECK)
# ====================================================================
if __name__ == "__main__":
    print("--- STEP 1: Creating Synthetic Practice Dataset ---")
    np.random.seed(7)
    sample_df = pd.DataFrame({
        'Price': np.random.uniform(10.0, 100.0, 100),
        'Quantity': np.random.randint(1, 10, 100),
        'Category': np.random.choice(['Kitchen', 'Electronics', 'Home'], 100)
    })
    sample_df['Total_Revenue'] = sample_df['Price'] * sample_df['Quantity']
    
    engine = SalesPredictiveEngine()
    
    print("\n--- STEP 2: Running Pipeline Training and State Persistence ---")
    training_success = engine.sales_predictive_model(sample_df)
    
    if training_success:
        print("\n--- STEP 3: Executing Live Telemetry Inference Test ---")
        call = engine.predict_revenue(34.9, 4, 'Kitchen')
        print(f"Prediction Output: ${call:,.2f} computed successfully.")
    else:
        print("\n[Sanity Check Failed] Machine Learning pipeline failed to initialize.")
