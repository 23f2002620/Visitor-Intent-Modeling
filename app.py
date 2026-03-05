"""
=============================================================
  Visitor Intent Modeling — Flask Server
  Serves dashboard_data.json via REST API and renders the
  HTML dashboard via Jinja2 template.
=============================================================

Project layout:
    flask_app/
    ├── app.py                  ← THIS FILE (Flask server)
    ├── dashboard_data.json     ← Model output (from intent_model.py)
    ├── templates/
    │   └── dashboard.html      ← Jinja2 dashboard template
    └── static/
        └── (optional assets)

Run:
    pip install flask
    python app.py
    → http://localhost:5000
"""

import json
import os
from datetime import datetime
from flask import Flask, jsonify, render_template, abort

# ── Config ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "dashboard_data.json")

app = Flask(__name__)


# ── Helper ────────────────────────────────────────────────────
def load_data() -> dict:
    """Load dashboard_data.json, raising 503 if missing."""
    if not os.path.exists(DATA_FILE):
        abort(503, description=(
            "dashboard_data.json not found. "
            "Run intent_model.py first to generate it."
        ))
    with open(DATA_FILE, "r") as f:
        return json.load(f)


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the full analytics dashboard."""
    data = load_data()
    return render_template(
        "dashboard.html",
        data=data,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.route("/api/data")
def api_data():
    """
    GET /api/data
    Returns the full dashboard_data.json as JSON.
    Used by the dashboard's fetch() call to hydrate charts.
    """
    data = load_data()
    return jsonify(data)


@app.route("/api/kpis")
def api_kpis():
    """GET /api/kpis — KPI summary only."""
    return jsonify(load_data().get("kpis", {}))


@app.route("/api/sessions")
def api_sessions():
    """GET /api/sessions — Sample session predictions."""
    return jsonify(load_data().get("sessions", []))


@app.route("/api/sources")
def api_sources():
    """GET /api/sources — Traffic source ranking."""
    return jsonify(load_data().get("sources", []))


@app.route("/api/features")
def api_features():
    """GET /api/features — Feature importance scores."""
    return jsonify(load_data().get("features", []))


@app.route("/api/metrics")
def api_metrics():
    """GET /api/metrics — Model evaluation metrics."""
    return jsonify(load_data().get("metrics", {}))


@app.route("/health")
def health():
    """Health check endpoint."""
    data_ok = os.path.exists(DATA_FILE)
    return jsonify({
        "status": "ok" if data_ok else "degraded",
        "data_file": data_ok,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200 if data_ok else 503


# ── Error Handlers ────────────────────────────────────────────
@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({"error": str(e.description)}), 503

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Visitor Intent Modeling — Flask Server")
    print("=" * 55)

    if not os.path.exists(DATA_FILE):
        print(f"\n  WARNING: dashboard_data.json not found at:")
        print(f"     {DATA_FILE}")
        print(f"  → Run:  python ../intent_model.py  first!\n")
    else:
        print(f"\n  Data file found: dashboard_data.json")

    print(f"\n  API Endpoints:")
    print(f"     GET /           → Full dashboard UI")
    print(f"     GET /api/data   → Complete JSON payload")
    print(f"     GET /api/kpis   → KPI summary")
    print(f"     GET /api/sessions  → Session predictions")
    print(f"     GET /api/sources   → Source ranking")
    print(f"     GET /api/features  → Feature importance")
    print(f"     GET /api/metrics   → Model metrics")
    print(f"     GET /health     → Health check")
    print(f"\n   Running on  http://localhost:5000\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
