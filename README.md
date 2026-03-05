# Visitor Intent Modeling — Flask App

## Project Structure
```
flask_app/
├── app.py                     ← Flask server (THIS IS THE NEW FILE)
├── dashboard_data.json        ← Model output (from intent_model.py)
├── requirements.txt
├── templates/
│   └── dashboard.html         ← Jinja2 template (fetches /api/data)
└── static/                    ← (optional: CSS/JS assets)
```

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate data by running the ML pipeline first
python ../../intent_model.py   # produces dashboard_data.json

# 3. Start the Flask server
python app.py

# 4. Open browser
http://localhost:5000
```

## API Endpoints

| Method | Endpoint       | Description |
|--------|----------------|-------------|
| GET    | /              | Full dashboard UI (Jinja2 rendered) |
| GET    | /api/data      | Complete dashboard_data.json |
| GET    | /api/kpis      | KPI summary only |
| GET    | /api/sessions  | Sample session predictions |
| GET    | /api/sources   | Traffic source ranking |
| GET    | /api/features  | Feature importance scores |
| GET    | /api/metrics   | Model evaluation metrics |
| GET    | /health        | Health check |

## How it works
1. `intent_model.py` trains the GBM and exports `dashboard_data.json`
2. `app.py` serves it via `/api/data`
3. `dashboard.html` calls `fetch('/api/data')` on load and renders all charts
