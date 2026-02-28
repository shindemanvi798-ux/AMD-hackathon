"""
AquaWatch AI — Synthetic Dataset Generator (Pune + Mumbai)
Run this FIRST before anything else.
Output: data/dataset.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── CITY AREA CONFIG ────────────────────────────────────────────
city_config = {
    "Pune": {
        "areas": ["Pimpri", "Hadapsar", "Wanowrie", "Kothrud", "Shivajinagar", "Aundh", "Yerwada", "Swargate"],
        "base_demand": {
            "Pimpri": 100, "Hadapsar": 75, "Wanowrie": 62,
            "Kothrud": 67, "Shivajinagar": 54, "Aundh": 71,
            "Yerwada": 60, "Swargate": 50
        },
        "base_leakage": {
            "Pimpri": 62, "Hadapsar": 45, "Wanowrie": 38,
            "Kothrud": 20, "Shivajinagar": 15, "Aundh": 32,
            "Yerwada": 48, "Swargate": 12
        },
        "avg_temp": 30, "avg_pressure": 52
    },
    "Mumbai": {
        "areas": ["Dharavi", "Andheri East", "Kurla", "Govandi", "Bandra", "Powai", "Worli", "Malad"],
        "base_demand": {
            "Dharavi": 175, "Andheri East": 158, "Kurla": 188,
            "Govandi": 158, "Bandra": 129, "Powai": 121,
            "Worli": 108, "Malad": 142
        },
        "base_leakage": {
            "Dharavi": 68, "Andheri East": 42, "Kurla": 71,
            "Govandi": 55, "Bandra": 22, "Powai": 18,
            "Worli": 28, "Malad": 35
        },
        "avg_temp": 28, "avg_pressure": 55
    }
}

rows = []

for city, cfg in city_config.items():
    for day in range(30):
        for hour in range(24):
            for area in cfg["areas"]:
                hour_factor = (
                    0.5 + 0.5 * np.sin((hour - 6) * np.pi / 12)
                    if 6 <= hour <= 22 else 0.3
                )
                is_weekend = int((day % 7) >= 5)
                weekend_factor = 0.85 if is_weekend else 1.0

                demand = (
                    cfg["base_demand"][area] * hour_factor * weekend_factor
                    + np.random.normal(0, 4)
                )
                demand = max(demand, 3)

                leakage_pct = cfg["base_leakage"][area] + np.random.normal(0, 2.5)
                leakage_pct = np.clip(leakage_pct, 5, 90)

                pressure    = np.random.normal(cfg["avg_pressure"], 7)
                temperature = np.random.normal(cfg["avg_temp"], 4)

                is_anomaly = 0
                if np.random.random() < 0.05:
                    demand *= np.random.uniform(2.0, 3.5)
                    is_anomaly = 1

                rows.append({
                    "city":         city,
                    "day":          day,
                    "hour":         hour,
                    "area":         area,
                    "demand_kl":    round(demand, 2),
                    "leakage_pct":  round(leakage_pct, 2),
                    "pressure_psi": round(pressure, 2),
                    "temperature_c":round(temperature, 2),
                    "is_weekend":   is_weekend,
                    "is_anomaly":   is_anomaly
                })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)

print(f"✅ Dataset generated: {len(df)} rows")
print(f"   Cities: Pune, Mumbai")
print(f"   Anomalies injected: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
print(f"   Saved to: data/dataset.csv")
print(df.head(8).to_string())