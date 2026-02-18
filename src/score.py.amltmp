import json
import os
import pandas as pd
import joblib


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Custom transformer to add engineered features."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["AvgMonthlySpend"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["IsNewCustomer"] = (X["tenure"] < 3).astype(int)
        X["LifetimeValueEstimate"] = X["MonthlyCharges"] * X["tenure"]
        X["TenureGroup"] = pd.qcut(
            X["tenure"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        X["MonthlyChargeTier"] = pd.qcut(
            X["MonthlyCharges"], q=4, labels=["Low", "Med", "High", "Very High"]
        )
        return X


def init():
    global model
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pkl")
    model = joblib.load(model_path)
    print("Modelo cargado")


def run(raw_data):
    try:
        # 1. Parsear datos
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        
        # 2. Guardar y eliminar customerID
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df = df.drop('customerID', axis=1)
        else:
            customer_ids = list(range(len(df)))
        
        # 3. Predecir probabilidades
        probabilities = model.predict_proba(df)[:, 1]
        
        # 4. Aplicar threshold
        # Decision
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)
        
        # 5. Retornar resultado
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "customerID": customer_ids[i],
                "churn_prediction": int(pred),
                "churn_probability": round(float(prob), 4),
                "will_churn": "Yes" if pred == 1 else "No"
            })
        
        return json.dumps(results)
        
    except Exception as e:
        return json.dumps({"error": str(e)})