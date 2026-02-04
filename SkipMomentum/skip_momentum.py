# =========================================================
# SkipMomentum — Session-Based Skip Probability Modeling
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =========================================================
# 1. LOAD & CLEAN
# =========================================================

df = pd.read_csv("spotify_history.csv")

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

df = (
    df.dropna(subset=["ts"])
      .drop_duplicates()
      .query("ms_played > 0")
      .sort_values("ts")
      .reset_index(drop=True)
)

# =========================================================
# 2. SESSION CREATION (30 MIN RULE)
# =========================================================

df["diff_minutes"] = df["ts"].diff() / pd.Timedelta(minutes=1)
df["new_session"] = (df["diff_minutes"] > 30) | (df["diff_minutes"].isna())
df["session_id"] = df["new_session"].cumsum()

# =========================================================
# 3. TARGET
# =========================================================

df["is_skip"] = (df["ms_played"] < 30_000).astype(int)

# =========================================================
# 4. BASE FEATURES
# =========================================================

df["track_pos_in_session"] = df.groupby("session_id").cumcount() + 1

df["time_since_last_session_started"] = (
    df["ts"] - df.groupby("session_id")["ts"].transform("min")
) / pd.Timedelta(minutes=1)

# =========================================================
# 5. STRICTLY PAST FEATURES
# =========================================================

df["prev_track_skipped"] = (
    df.groupby("session_id")["is_skip"].shift(1).fillna(0)
)

df["prev_ms_played"] = (
    df.groupby("session_id")["ms_played"].shift(1)
)

# =========================================================
# 6. MOMENTUM + FATIGUE FEATURES
# =========================================================

# Past consecutive skips
df["past_consecutive_skips"] = (
    df.groupby("session_id")["is_skip"]
      .shift(1)
      .fillna(0)
      .groupby(df["session_id"])
      .apply(lambda x: x * (x.groupby((x == 0).cumsum()).cumcount() + 1))
      .reset_index(level=0, drop=True)
)

# Past skip ratio
df["past_skip_ratio"] = (
    df.groupby("session_id")["is_skip"]
      .shift(1)
      .expanding()
      .mean()
      .reset_index(level=0, drop=True)
      .fillna(0)
)

# =========================================================
# 7. REMOVE FIRST TRACKS
# =========================================================

df = df[df["track_pos_in_session"] > 1].copy()

# =========================================================
# 8. MODEL FEATURES
# =========================================================

FEATURES = [
    "time_since_last_session_started",
    "prev_track_skipped",
    "prev_ms_played",
    "past_consecutive_skips",
    "past_skip_ratio"
]

TARGET = "is_skip"

X = df[FEATURES]
y = df[TARGET]

# =========================================================
# 9. TEMPORAL SPLIT
# =========================================================

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================================================
# 10. SCALE
# =========================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =========================================================
# 11. TRAIN
# =========================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# =========================================================
# 12. EVALUATION
# =========================================================

y_probs = model.predict_proba(X_test_s)[:, 1]
roc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC: {roc:.4f}")

# =========================================================
# 13. COEFFICIENT PLOT
# =========================================================

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_[0]
}).sort_values("coefficient")

plt.figure()
plt.barh(coef_df["feature"], coef_df["coefficient"])
plt.title("Feature Influence on Skip Probability")
plt.tight_layout()
plt.show()

# =========================================================
# 14. SESSION FATIGUE CURVE (BEST SESSION)
# =========================================================

# Pick a long session
session_lengths = df.groupby("session_id").size()
best_session_id = session_lengths.idxmax()

session_df = df[df["session_id"] == best_session_id].copy()

session_X = scaler.transform(session_df[FEATURES])
session_df["skip_probability"] = model.predict_proba(session_X)[:, 1]

plt.figure()
plt.plot(
    session_df["track_pos_in_session"],
    session_df["skip_probability"]
)
plt.xlabel("Track Position in Session")
plt.ylabel("Skip Probability")
plt.title("Skip Probability Progression Within a Session")
plt.tight_layout()
plt.show()

# =========================================================
# 15. FINAL SNAPSHOT OUTPUT
# =========================================================

print("\nSample Session Prediction:")
print(
    session_df[
        ["track_pos_in_session", "skip_probability", "is_skip"]
    ].head(10)
)
