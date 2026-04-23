"""
train_model.py
--------------
Quick-train script — run this to skip the notebook and jump straight to the web app.
Usage:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "flight_price.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_price_rf_model.joblib")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("📂  Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("⚙️   Engineering features …")

# Map stops to numeric
stops_map = {"0": 0, "1": 1, "2 or more": 2}
df["stops_num"] = df["stops"].map(stops_map).fillna(1).astype(int)

# Map time-of-day to numeric
time_map = {
    "Early Morning": 0,
    "Morning":       1,
    "Noon":          2,
    "Afternoon":     3,
    "Evening":       4,
    "Night":         5,
}
df["departure_time_num"] = df["departure_time"].map(time_map).fillna(2).astype(int)
df["arrival_time_num"]   = df["arrival_time"].map(time_map).fillna(2).astype(int)

# Label-encode categoricals
cat_cols = ["airline", "source_city", "destination_city", "travel_class"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 3. Select features ────────────────────────────────────────────────────────
feature_cols = [
    "airline_enc",
    "source_city_enc",
    "departure_time_num",
    "stops_num",
    "arrival_time_num",
    "destination_city_enc",
    "travel_class_enc",
    "duration",
    "days_left",
]
X = df[feature_cols]
y = df["price"]

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("🌲  Training Random Forest Regressor …")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\n📊  Evaluation on test set:")
print(f"    MAE : ₹{mae:,.2f}")
print(f"    R²  : {r2:.4f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
payload = {
    "model":    model,
    "encoders": encoders,
    "feature_cols": feature_cols,
    "stops_map":    stops_map,
    "time_map":     time_map,
}
joblib.dump(payload, MODEL_PATH)
print(f"\n✅  Model saved to: {MODEL_PATH}")
