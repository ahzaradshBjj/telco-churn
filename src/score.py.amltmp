import json
import os
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


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

        # 2. FEATURE ENGINEERING 
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["IsNewCustomer"] = (df["tenure"] < 3).astype(int)
        df["LifetimeValueEstimate"] = df["MonthlyCharges"] * df["tenure"]
        df["TenureGroup"] = pd.qcut(df["tenure"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        df["MonthlyChargeTier"] = pd.qcut(df["MonthlyCharges"], q=4, labels=["Low", "Med", "High", "Very High"])
        
        # 3. Guardar y eliminar customerID
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df = df.drop('customerID', axis=1)
        else:
            customer_ids = list(range(len(df)))      

        # 4. Predecir probabilidades
        probabilities = model.predict_proba(df)[:, 1]
        
        # 5. Aplicar threshold
        # Decision
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)
        
        # 6. Retornar resultado
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