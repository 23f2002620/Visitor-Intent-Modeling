"""
============================================================
  VISITOR INTENT MODELING — Full ML Pipeline
  Generates dummy data, trains GBM classifier, scores all
  sessions, ranks traffic sources, outputs dashboard JSON.
============================================================
"""

import json, warnings, os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 62)
print("VISITOR INTENT MODELING PIPELINE")
print("=" * 62)


# 1. GENERATE DUMMY DATA


def generate_sessions(n=5000, seed=42):
    np.random.seed(seed)

    SOURCES = {
        "Organic Search":  {"weight": 0.25, "boost": 1.70},
        "Email Campaign":  {"weight": 0.12, "boost": 1.50},
        "Paid Search":     {"weight": 0.20, "boost": 1.35},
        "Direct":          {"weight": 0.13, "boost": 1.15},
        "Referral":        {"weight": 0.09, "boost": 1.05},
        "Social Organic":  {"weight": 0.09, "boost": 0.60},
        "Social Paid":     {"weight": 0.07, "boost": 0.50},
        "Display Ads":     {"weight": 0.05, "boost": 0.38},
    }
    DEVICES = {"Desktop": 0.52, "Mobile": 0.38, "Tablet": 0.10}
    GEOS = {
        "US":0.30, "UK":0.10, "CA":0.07, "DE":0.07,
        "AU":0.05, "FR":0.05, "IN":0.09, "BR":0.07,
        "MX":0.05, "Other":0.15
    }
    TIER1 = {"US","UK","CA","DE","AU","FR"}

    src_keys   = list(SOURCES.keys())
    src_w      = [SOURCES[s]["weight"] for s in src_keys]
    dev_keys   = list(DEVICES.keys())
    dev_w      = list(DEVICES.values())
    geo_keys   = list(GEOS.keys())
    geo_w      = list(GEOS.values())

    traffic_source = np.random.choice(src_keys, n, p=src_w)
    device_type    = np.random.choice(dev_keys, n, p=dev_w)
    geography      = np.random.choice(geo_keys, n, p=geo_w)
    hour_of_day    = np.random.randint(0, 24, n)

    boost = np.array([SOURCES[s]["boost"] for s in traffic_source])
    device_factor  = np.where(device_type=="Desktop",1.0,
                      np.where(device_type=="Tablet",0.65,0.35))
    geo_factor     = np.where(np.isin(geography, list(TIER1)), 1.0, 0.5)
    business_hour  = ((hour_of_day >= 9) & (hour_of_day <= 18)).astype(float)

    # Session duration (seconds): skewed log-normal
    base_duration = np.random.lognormal(mean=5.0, sigma=1.1, size=n)
    session_duration_sec = np.clip(
        base_duration * boost * device_factor + np.random.normal(0, 20, n),
        5, 1800
    ).astype(int)

    # Pages visited: Poisson
    pages_visited = np.clip(
        np.random.poisson(3.0 * boost * device_factor, n) + 1,
        1, 25
    ).astype(int)

    # Bounce flag
    is_bounce = ((session_duration_sec < 30) & (pages_visited == 1)).astype(int)

    # Build intent score from features (ground truth proxy)
    intent_signal = (
        0.28 * np.clip(session_duration_sec / 600, 0, 1) +
        0.22 * np.clip(pages_visited / 10, 0, 1) +
        0.20 * (boost / 1.7) +
        0.12 * device_factor +
        0.10 * geo_factor +
        0.08 * business_hour
    )
    noise = np.random.normal(0, 0.07, n)
    converted = ((intent_signal + noise) > 0.52).astype(int)

    return pd.DataFrame({
        "session_id":           [f"S-{100000+i}" for i in range(n)],
        "traffic_source":       traffic_source,
        "session_duration_sec": session_duration_sec,
        "pages_visited":        pages_visited,
        "device_type":          device_type,
        "geography":            geography,
        "hour_of_day":          hour_of_day,
        "is_bounce":            is_bounce,
        "converted":            converted,
    })


print("\n[1/6] Generating 5,000 dummy sessions...")
df = generate_sessions(5000)
df.to_csv("sessions.csv", index=False)
print(f"      ✓ {len(df):,} sessions generated")
print(f"      ✓ Converted (high-intent): {df['converted'].sum():,} ({df['converted'].mean()*100:.1f}%)")
print(f"      ✓ Low-intent: {(df['converted']==0).sum():,} ({(df['converted']==0).mean()*100:.1f}%)")


# 2. FEATURE ENGINEERING

print("\n[2/6] Engineering features...")

TIER1_GEOS = {"US","UK","CA","DE","AU","FR"}

le_src = LabelEncoder()
le_dev = LabelEncoder()
le_geo = LabelEncoder()

df["source_enc"]    = le_src.fit_transform(df["traffic_source"])
df["device_enc"]    = le_dev.fit_transform(df["device_type"])
df["geo_enc"]       = le_geo.fit_transform(df["geography"])

df["duration_min"]   = df["session_duration_sec"] / 60
df["pages_per_min"]  = df["pages_visited"] / df["duration_min"].clip(lower=0.1)
df["is_desktop"]     = (df["device_type"] == "Desktop").astype(int)
df["is_mobile"]      = (df["device_type"] == "Mobile").astype(int)
df["is_tier1"]       = df["geography"].isin(TIER1_GEOS).astype(int)
df["is_biz_hour"]    = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 18)).astype(int)
df["dur_bucket"]     = pd.cut(
    df["session_duration_sec"],
    bins=[0,30,120,300,600,1800],
    labels=[0,1,2,3,4]
).astype(int)
df["engagement"]     = df["pages_visited"] * np.log1p(df["session_duration_sec"] / 60)
df["deep_session"]   = ((df["session_duration_sec"] > 240) & (df["pages_visited"] >= 4)).astype(int)

FEATURE_COLS = [
    "session_duration_sec", "pages_visited", "source_enc",
    "device_enc", "geo_enc", "duration_min", "pages_per_min",
    "is_desktop", "is_mobile", "is_tier1", "is_biz_hour",
    "dur_bucket", "engagement", "deep_session", "is_bounce",
    "hour_of_day",
]
FEATURE_LABELS = {
    "session_duration_sec": "Session Duration",
    "pages_visited":        "Pages Visited",
    "source_enc":           "Traffic Source",
    "device_enc":           "Device Type",
    "geo_enc":              "Geography",
    "duration_min":         "Duration (min)",
    "pages_per_min":        "Pages per Minute",
    "is_desktop":           "Is Desktop",
    "is_mobile":            "Is Mobile",
    "is_tier1":             "Tier-1 Geo",
    "is_biz_hour":          "Business Hour",
    "dur_bucket":           "Duration Bucket",
    "engagement":           "Engagement Score",
    "deep_session":         "Deep Session Flag",
    "is_bounce":            "Bounce Flag",
    "hour_of_day":          "Hour of Day",
}

X = df[FEATURE_COLS]
y = df["converted"]
print(f"      ✓ {len(FEATURE_COLS)} features engineered")


# 3. TRAIN MODEL

print("\n[3/6] Training Gradient Boosting Classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    min_samples_split=20,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=15,
)
model.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"      ✓ CV AUC-ROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")


# 4. EVALUATE

print("\n[4/6] Evaluating model...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
    "auc_roc":   round(roc_auc_score(y_test, y_proba), 3),
    "f1":        round(f1_score(y_test, y_pred) * 100, 1),
    "precision": round(precision_score(y_test, y_pred) * 100, 1),
    "recall":    round(recall_score(y_test, y_pred) * 100, 1),
    "cv_auc_mean": round(cv_auc.mean(), 3),
    "cv_auc_std":  round(cv_auc.std(), 3),
}
cm = confusion_matrix(y_test, y_pred)
metrics["tp"] = int(cm[1,1]); metrics["fp"] = int(cm[0,1])
metrics["fn"] = int(cm[1,0]); metrics["tn"] = int(cm[0,0])

for k, v in metrics.items():
    print(f"      {k:>14}: {v}")


# 5. SCORE ALL SESSIONS + RANK SOURCES

print("\n[5/6] Scoring all sessions & ranking sources...")

df["intent_score"] = model.predict_proba(X)[:, 1]
df["intent_label"] = np.where(df["intent_score"] >= 0.5, "HIGH", "LOW")

# Traffic source ranking
src_rank = (
    df.groupby("traffic_source")
    .agg(
        sessions        = ("session_id", "count"),
        avg_score       = ("intent_score", "mean"),
        high_intent_pct = ("intent_label", lambda x: round((x=="HIGH").mean()*100, 1)),
        avg_pages       = ("pages_visited", "mean"),
        avg_duration    = ("session_duration_sec", "mean"),
        conv_rate       = ("converted", "mean"),
    )
    .reset_index()
    .sort_values("avg_score", ascending=False)
    .reset_index(drop=True)
)
src_rank["rank_num"] = src_rank.index + 1

# Feature importance
fi = pd.DataFrame({
    "feature":    [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "raw":        FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
fi["importance_pct"] = (fi["importance"] * 100).round(2)

print("      ✓ All sessions scored")
print("      ✓ Traffic sources ranked")
print("      ✓ Feature importance computed")

print("\n  📡 TRAFFIC SOURCE RANKING:")
print(f"  {'Rank':<5} {'Source':<20} {'Avg Score':<12} {'High-Intent%':<15} {'Conv%'}")
print("  " + "-"*62)
for _, r in src_rank.iterrows():
    print(f"  #{int(r.rank_num):<4} {r.traffic_source:<20} {r.avg_score:.3f}       {r.high_intent_pct}%          {r.conv_rate*100:.1f}%")

print("\n  🧠 TOP BEHAVIORAL INDICATORS:")
print(f"  {'Feature':<25} {'Importance%'}")
print("  " + "-"*40)
for _, r in fi.head(8).iterrows():
    bar = "█" * int(r.importance_pct * 1.2)
    print(f"  {r.feature:<25} {bar:<30} {r.importance_pct:.1f}%")



# 6. EXPORT DATA FOR DASHBOARD


print("\n[6/6] Exporting data for dashboard...")

# Sample sessions for table (mix of high/low)
sample_hi = df[df["intent_label"]=="HIGH"].sample(12, random_state=1)
sample_lo = df[df["intent_label"]=="LOW"].sample(8,  random_state=1)
sample    = pd.concat([sample_hi, sample_lo]).sample(frac=1, random_state=3)

sessions_json = []
for _, r in sample.iterrows():
    sessions_json.append({
        "id":       r.session_id,
        "source":   r.traffic_source,
        "duration": f"{int(r.session_duration_sec//60)}m {int(r.session_duration_sec%60)}s",
        "pages":    int(r.pages_visited),
        "device":   r.device_type,
        "geo":      r.geography,
        "score":    round(float(r.intent_score), 3),
        "label":    r.intent_label,
        "converted":int(r.converted),
    })

sources_json = []
for _, r in src_rank.iterrows():
    sources_json.append({
        "rank":         int(r.rank_num),
        "name":         r.traffic_source,
        "sessions":     int(r.sessions),
        "avg_score":    round(float(r.avg_score), 3),
        "hi_pct":       float(r.high_intent_pct),
        "avg_pages":    round(float(r.avg_pages), 1),
        "avg_dur":      round(float(r.avg_duration), 0),
        "conv_rate":    round(float(r.conv_rate)*100, 1),
    })

features_json = []
for _, r in fi.head(10).iterrows():
    features_json.append({
        "name":  r.feature,
        "pct":   round(float(r.importance_pct), 2),
    })

# Score distribution (binned)
bins = np.linspace(0, 1, 21)
hist_hi, _ = np.histogram(df[df["intent_label"]=="HIGH"]["intent_score"], bins=bins)
hist_lo, _ = np.histogram(df[df["intent_label"]=="LOW"]["intent_score"],  bins=bins)
dist_json  = [{"bin": round(bins[i],2), "high": int(hist_hi[i]), "low": int(hist_lo[i])}
              for i in range(len(hist_hi))]

# Device x intent breakdown
dev_breakdown = df.groupby("device_type").apply(
    lambda g: {"hi_pct": round((g.intent_label=="HIGH").mean()*100,1), "count": len(g)}
).to_dict()

# Overall KPIs
total = len(df)
hi    = (df["intent_label"]=="HIGH").sum()
lo    = total - hi
avg_score_hi = round(df[df["intent_label"]=="HIGH"]["intent_score"].mean(), 3)
avg_score_lo = round(df[df["intent_label"]=="LOW"]["intent_score"].mean(), 3)
conv_lift    = round(
    df[df["intent_label"]=="HIGH"]["converted"].mean() /
    max(df[df["intent_label"]=="LOW"]["converted"].mean(), 0.001), 1
)

dashboard_data = {
    "kpis": {
        "total_sessions": total,
        "high_intent": int(hi),
        "low_intent":  int(lo),
        "hi_pct":      round(hi/total*100, 1),
        "avg_score":   round(float(df["intent_score"].mean()), 3),
        "conv_lift":   conv_lift,
    },
    "metrics":    metrics,
    "sessions":   sessions_json,
    "sources":    sources_json,
    "features":   features_json,
    "dist":       dist_json,
    "devices":    {k: v for k, v in dev_breakdown.items()},
}

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

df[["session_id","traffic_source","session_duration_sec","pages_visited",
    "device_type","geography","hour_of_day","is_bounce",
    "intent_score","intent_label","converted"]].to_csv("session_scores.csv", index=False)

src_rank.to_csv("source_ranking.csv", index=False)
fi.to_csv("feature_importance.csv", index=False)
joblib.dump(model, "intent_model.joblib")

print("      ✓ dashboard_data.json")
print("      ✓ session_scores.csv")
print("      ✓ source_ranking.csv")
print("      ✓ feature_importance.csv")
print("      ✓ intent_model.joblib")
print("\n✅ Pipeline complete!\n")
