import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# 1. Configuration: Tells the script exactly where to save the model.pkl file.
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

# 2. The Main Class: Responsible for model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            print("🚀 Starting Model Training...")

            # Step A: Split the incoming data into Features (X) and Target (y)
            logging.info("Split training and test input data")
            print("✂️ Splitting inputs & target variables...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Step B: Define a dictionary of our top algorithms
            models = {
                "Ridge Regression": Ridge(),
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42)
            }

            model_report = {}
            
            # Step C: Train and evaluate each model
            print("⚙️ Training and evaluating algorithms...")
            for name, model in models.items():
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_test_pred = model.predict(X_test)
                
                # Calculate the R2 Score
                test_model_score = r2_score(y_test, y_test_pred)
                model_report[name] = test_model_score

            # Step D: Find the absolute best model based on the R2 score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f"🏆 Best Model Found: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Safety check: If our best model is garbage, stop the pipeline
            if best_model_score < 0.6:
                print("⚠️ No suitable model found! Scores are too low.")
                raise CustomException("No suitable model found")

            # Step E: Save the winning model to the artifacts vault
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f"💾 Model saved at: {self.model_trainer_config.trained_model_file_path}")
            print("🎉 Model Training Completed Successfully!")

            return best_model_score
            
        except Exception as e:
            print(f"❌ Error during Model Training: {e}")
            raise CustomException(e, sys)

# --- Code to test the ENTIRE Pipeline ---
if __name__ == "__main__":
    # We need to import our other components just for this test
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # 1. Ingest Data
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # 2. Transform Data
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    # 3. Train Model
    model_trainer = ModelTrainer()
    final_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    print(f"\n✅ PIPELINE SUCCESS! Final Champion R2 Score: {final_score:.4f}")
