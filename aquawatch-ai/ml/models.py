"""
AquaWatch AI — ML Models (Pune + Mumbai)
1. Isolation Forest  → Anomaly Detection
2. Linear Regression → Demand Forecasting
3. Leakage Risk Scoring

Run: python models.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

print("📂 Loading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"   {len(df)} rows loaded\n")

# Encode area and city as numbers
le_area = LabelEncoder()
le_city = LabelEncoder()
df["area_encoded"] = le_area.fit_transform(df["area"])
df["city_encoded"] = le_city.fit_transform(df["city"])

# ── MODEL 1: ISOLATION FOREST ────────────────────────────────────
print("🤖 Training Isolation Forest (Anomaly Detection)...")
features_anomaly = ["demand_kl", "leakage_pct", "pressure_psi", "hour", "area_encoded", "city_encoded"]
X_anomaly = df[features_anomaly]

iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_model.fit(X_anomaly)

df["predicted_anomaly"] = (iso_model.predict(X_anomaly) == -1).astype(int)
acc = accuracy_score(df["is_anomaly"], df["predicted_anomaly"])
print(f"   ✅ Isolation Forest trained | Accuracy: {acc*100:.1f}%\n")

# ── MODEL 2: LINEAR REGRESSION ───────────────────────────────────
print("📈 Training Linear Regression (Demand Forecasting)...")
features_demand = ["hour", "is_weekend", "temperature_c", "area_encoded", "city_encoded"]
X_demand = df[features_demand]
y_demand = df["demand_kl"]

X_train, X_test, y_train, y_test = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, lr_model.predict(X_test))
print(f"   ✅ Linear Regression trained | MAE: {mae:.2f} kL\n")

# ── SAVE MODELS ──────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/isolation_forest.pkl", "wb") as f: pickle.dump(iso_model, f)
with open("models/demand_forecaster.pkl", "wb") as f: pickle.dump(lr_model, f)
with open("models/label_encoder_area.pkl", "wb") as f: pickle.dump(le_area, f)
with open("models/label_encoder_city.pkl", "wb") as f: pickle.dump(le_city, f)
print("💾 Models saved to models/\n")

# ── TEST ─────────────────────────────────────────────────────────
def leakage_risk(pct):
    if pct >= 55: return "high"
    elif pct >= 30: return "medium"
    else: return "low"

def predict(city, area, hour, leakage_pct, pressure_psi, temperature_c, is_weekend=0):
    try: area_enc = int(le_area.transform([area])[0])
    except: area_enc = 0
    try: city_enc = int(le_city.transform([city])[0])
    except: city_enc = 0

    X_iso = pd.DataFrame([{"demand_kl": 50, "leakage_pct": leakage_pct, "pressure_psi": pressure_psi,
                            "hour": hour, "area_encoded": area_enc, "city_encoded": city_enc}])
    anomaly_score = float(iso_model.decision_function(X_iso)[0])
    is_anomaly    = int(iso_model.predict(X_iso)[0] == -1)

    X_lr = pd.DataFrame([{"hour": hour, "is_weekend": is_weekend,
                           "temperature_c": temperature_c, "area_encoded": area_enc, "city_encoded": city_enc}])
    predicted_demand = float(lr_model.predict(X_lr)[0])

    return {
        "city": city, "area": area, "hour": hour,
        "predicted_demand_kl": round(predicted_demand, 2),
        "leakage_pct": leakage_pct, "leakage_risk": leakage_risk(leakage_pct),
        "is_anomaly": is_anomaly, "anomaly_score": round(anomaly_score, 4)
    }

if __name__ == "__main__":
    print("🧪 Test — Pune:")
    r = predict("Pune", "Pimpri", 14, 62.0, 48.0, 31.0)
    for k, v in r.items(): print(f"   {k}: {v}")

    print("\n🧪 Test — Mumbai:")
    r = predict("Mumbai", "Dharavi", 14, 68.0, 50.0, 29.0)
    for k, v in r.items(): print(f"   {k}: {v}")