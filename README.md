# Beachvolleyball Video Analyzer

Analysiert Beachvolleyball-Videos mit Roboflow Workflow. Erkennt Spieler (Team A/B), Aktionen (Attack, Reception, Serve, Block) und zeigt Statistiken pro Spieler mit Fotos.

## Lokale Installation
1. `pip install -r requirements.txt`
2. Erstelle `.streamlit/secrets.toml` mit API-Key, Workspace und Workflow-ID.
3. `streamlit run app.py`

## Deployment auf Streamlit Cloud
- Setze die Secrets entsprechend in der App-Oberfläche.
- Lade nur den Code (ohne secrets.toml) auf GitHub.
