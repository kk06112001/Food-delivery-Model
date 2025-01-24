import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from exception_handler.custom_exceptions import (
    DataExtractionException,
    DataTransformationException,
    ModelTrainingException,
    ModelEvaluationException
)
from logging_package import log_info, log_warning, log_error, setup_logging

setup_logging()

class ETLPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def extract(self):
        log_info("Extracting data from CSV...")
        try:
            self.df = pd.read_csv(self.data_path)
            log_info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            log_error("Error loading data", e)
            raise DataExtractionException(f"Failed to extract data from {self.data_path}: {e}")

    def transform(self):
        log_info("Transforming data...")
        try:
            self.df = self.df.drop(columns='Unnamed: 14', axis=1)
            log_info("Dropped column 'Unnamed: 14'.")
            self.df = self.df.dropna()
            log_info("Dropped rows with missing values.")
            self.df['Delivery_person_Age'] = self.df['Delivery_person_Age'].astype(int)
            self.df['TARGET'] = pd.to_numeric(self.df['TARGET'], errors='coerce')
            self.df['TARGET'] = self.df['TARGET'].round(2)
            log_info("Converted 'Delivery_person_Age' to int and 'TARGET' to numeric.")

            le = LabelEncoder()
            categorical_columns = ['Type_of_order', 'Type_of_vehicle', 'weather_description', 'Traffic_Level']
            for col in categorical_columns:
                self.df[col] = le.fit_transform(self.df[col])
            log_info("Encoded categorical columns.")
            X = self.df[['Distance (km)', 'Type_of_vehicle', 'Traffic_Level', 'Delivery_person_Age']]
            y = self.df['TARGET']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            log_info("Data split into train and test sets.")
        except Exception as e:
            log_error("Error during transformation", e)
            raise DataTransformationException(f"Failed to transform data: {e}")

    def load(self):
        log_info("Training models...")
        try:
            models = {
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }

            results = {}

            for model_name, model in models.items():
                model.fit(self.X_train, self.y_train)  
                y_pred = model.predict(self.X_test) 
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)

                results[model_name] = {
                    'R-squared': r2,
                    'MAE': mae,
                    'MSE': mse
                }

            for model_name, metrics in results.items():
                log_info(f"{model_name} Evaluation:")
                for metric, value in metrics.items():
                    log_info(f"  {metric}: {value:.4f}")

            best_model_name = max(results, key=lambda x: results[x]['R-squared'])
            self.model = models[best_model_name]
            joblib.dump(self.model, 'best_model.pkl')  
            log_info(f"Best model ({best_model_name}) saved to disk.")
            return results
        except Exception as e:
            log_error("Error during model training/evaluation", e)
            raise ModelTrainingException(f"Failed to train model: {e}")

if __name__ == "__main__":
    data_path = 'Food_Time_Data_Set.csv' 
    
    pipeline = ETLPipeline(data_path)
    
    try:
        pipeline.extract()
        pipeline.transform()
        results = pipeline.load()
        print("\nModel Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print()
    except (DataExtractionException, DataTransformationException, ModelTrainingException) as e:
        log_error(f"ETL pipeline failed: {e}")
        print(f"ETL pipeline failed. Check the log file for details.")
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        print(f"Unexpected error occurred. Check the log file for details.")
