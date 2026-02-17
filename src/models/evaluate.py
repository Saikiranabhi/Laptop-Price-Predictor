# Auto-generated file
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        results[name] = metrics
        print(f"{name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
    
    # Select best model
    best_model = max(results, key=lambda x: results[x]['r2'])
    return best_model, results