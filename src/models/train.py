# import mlflow
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

# def train_models(X_train, y_train):
#     models = {}
    
#     # Random Forest
#     with mlflow.start_run(run_name="RandomForest"):
#         rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
#         rf.fit(X_train, y_train)
#         mlflow.log_params({"n_estimators": 100, "max_depth": 10})
#         mlflow.sklearn.log_model(rf, "random_forest")
#         models['rf'] = rf
    
#     # XGBoost
#     with mlflow.start_run(run_name="XGBoost"):
#         xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
#         xgb.fit(X_train, y_train)
#         mlflow.log_params({"n_estimators": 100, "learning_rate": 0.1})
#         mlflow.xgboost.log_model(xgb, "xgboost")
#         models['xgb'] = xgb
    
#     return models


import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and compare multiple regression models"""
    
    def __init__(self, experiment_name: str = "laptop-price-prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Define all models
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        }
        
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split data into train/test sets"""
        X = df.drop(columns=['Price'])
        y = np.log(df['Price'])  # Log transform target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, name: str, model, X_train, y_train, X_test, y_test):
        """Train single model and log to MLflow"""
        
        with mlflow.start_run(run_name=name):
            # Train
            logger.info(f"🚀 Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metrics({
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_r2_mean': cv_mean
            })
            
            # Log model
            if 'XGBoost' in name:
                mlflow.xgboost.log_model(model, name)
            else:
                mlflow.sklearn.log_model(model, name)
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_r2_mean': cv_mean
            }
            
            logger.info(f"✅ {name}: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    
    def train_all(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        logger.info("🎯 Starting training for all models...")
        
        for name, model in self.models.items():
            try:
                self.train_model(name, model, X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
        
        logger.info("✅ All models trained!")
    
    def get_best_model(self):
        """Select best model based on test R²"""
        best_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        best_model = self.results[best_name]['model']
        best_metrics = self.results[best_name]
        
        logger.info(f"🏆 Best Model: {best_name}")
        logger.info(f"   Test R²: {best_metrics['test_r2']:.4f}")
        logger.info(f"   RMSE: {best_metrics['test_rmse']:.4f}")
        logger.info(f"   MAE: {best_metrics['test_mae']:.4f}")
        
        return best_name, best_model, best_metrics
    
    def save_best_model(self, best_model, filepath: str = 'models/best_model.pkl'):
        """Save best model to disk"""
        joblib.dump(best_model, filepath)
        logger.info(f"💾 Best model saved to {filepath}")
    
    def print_comparison_table(self):
        """Print model comparison table"""
        df_results = pd.DataFrame(self.results).T
        df_results = df_results[['train_r2', 'test_r2', 'test_rmse', 'test_mae', 'cv_r2_mean']]
        df_results = df_results.sort_values('test_r2', ascending=False)
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON TABLE")
        print("="*80)
        print(df_results.to_string())
        print("="*80 + "\n")
        
        return df_results