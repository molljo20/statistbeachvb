import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer")

# --- 1. API-Key aus den Secrets laden ---
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Bitte setze den ROBOFLOW_API_KEY in den Secrets.")
    st.stop()

# Client initialisieren
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)

# Modell-IDs (direkter Aufruf, kein Workflow)
MODEL_PLAYERS = "masters-jzkco/beach-volleyball-players-teams/2"
MODEL_ACTIONS = "activity-graz-uni/volleyball-activity-dataset/3"

def extract_player_photo(frame, bbox, margin=20):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

# --- 2. Datei-Upload ---
uploaded_file = st.file_uploader("Beachvolleyball-Video (MP4)", type=["mp4", "mov"])

if uploaded_file and st.button("Analyse starten"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    with st.spinner("Analysiere Video (max. 200 Frames, jeden 5.)..."):
        cap = cv2.VideoCapture(video_path)
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 200)
        frame_skip = 5

        # Datenstrukturen für die Statistik
        player_stats = defaultdict(lambda: {"attacks": 0, "receptions": 0, "serves": 0, "blocks": 0, "photo": None, "best_conf": 0})
        last_positions = {}
        next_temp_id = 1
        id_mapping = {}
        a_count = 0
        b_count = 0

        progress_bar = st.progress(0)

        for frame_idx in range(0, max_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Frame temporär speichern
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
                cv2.imwrite(tmp_img.name, frame)

                # 1. Spielererkennung
                players = []
                try:
                    res_players = client.infer(tmp_img.name, model_id=MODEL_PLAYERS)
                    for pred in res_players.get('predictions', []):
                        cls = pred.get('class')
                        if cls not in ('player_a', 'player_b'):
                            continue
                        x = pred.get('x', 0); y = pred.get('y', 0)
                        w = pred.get('width', 0); h = pred.get('height', 0)
                        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                        conf = pred.get('confidence', 0)
                        players.append((cls, x, y, conf, (x1, y1, x2, y2)))
                except Exception: pass

                # 2. Aktionserkennung
                actions = []
                try:
                    res_actions = client.infer(tmp_img.name, model_id=MODEL_ACTIONS)
                    for pred in res_actions.get('predictions', []):
                        cls = pred.get('class')
                        if cls in ('Serve', 'Attack', 'Reception', 'Block'):
                            actions.append(cls)
                except Exception: pass

                os.unlink(tmp_img.name)

            # --- Tracking & Zuordnung der Spieler ---
            current_positions = {}
            for cls, cx, cy, conf, bbox in players:
                best_id = None; best_dist = 150
                for tid, (px, py) in last_positions.items():
                    dist = np.hypot(cx - px, cy - py)
                    if dist < best_dist: best_dist = dist; best_id = tid
                if best_id is None:
                    best_id = next_temp_id; next_temp_id += 1
                    if cls == 'player_a': a_count += 1; perm = f"A{a_count}"
                    else: b_count += 1; perm = f"B{b_count}"
                    id_mapping[best_id] = perm
                perm_id = id_mapping[best_id]
                current_positions[best_id] = (cx, cy)

                if conf > player_stats[perm_id]["best_conf"]:
                    player_stats[perm_id]["best_conf"] = conf
                    img = extract_player_photo(frame, bbox)
                    if img is not None: player_stats[perm_id]["photo"] = img

            # --- Aktionen den Spielern zuordnen ---
            if actions and current_positions:
                perm_ids = [id_mapping[pid] for pid in current_positions.keys()]
                for act in actions:
                    for perm_id in perm_ids:
                        if act == 'Attack': player_stats[perm_id]["attacks"] += 1
                        elif act == 'Reception': player_stats[perm_id]["receptions"] += 1
                        elif act == 'Serve': player_stats[perm_id]["serves"] += 1
                        elif act == 'Block': player_stats[perm_id]["blocks"] += 1

            last_positions = current_positions
            progress_bar.progress(min(frame_idx / max_frames, 1.0))

        cap.release()
        os.unlink(video_path)

    # --- 3. Ergebnisse anzeigen ---
    if not player_stats:
        st.error("Keine Spieler erkannt. Bitte Video mit besserer Qualität.")
    else:
        data = []
        for pid, stats in player_stats.items():
            success_rate = 50.0  # Platzhalter
            data.append({
                "Spieler": pid, "Angriffe": stats["attacks"], "Erfolgsquote": f"{success_rate:.1f}%",
                "Annahmen": stats["receptions"], "Aufschläge": stats["serves"], "Blocks": stats["blocks"],
                "Foto": stats["photo"]
            })
        df = pd.DataFrame(data)

        st.subheader("📊 Spielerstatistiken")
        cols = st.columns(4)
        for idx, row in df.iterrows():
            with cols[idx % 4]:
                if row["Foto"] is not None:
                    img = Image.fromarray(cv2.cvtColor(row["Foto"], cv2.COLOR_BGR2RGB))
                    st.image(img, caption=row["Spieler"], width=120)
                else: st.write(f"**{row['Spieler']}** (kein Foto)")
                st.metric("Angriffe", row["Angriffe"])
                st.metric("Erfolgsquote", row["Erfolgsquote"])
                st.metric("Annahmen", row["Annahmen"])
                st.metric("Aufschläge", row["Aufschläge"])
                st.metric("Blocks", row["Blocks"])

        # PDF-Export
        if st.button("📄 PDF exportieren"):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            story = []; styles = getSampleStyleSheet()
            story.append(Paragraph("Beachvolleyball Analysebericht", styles['Title']))
            story.append(Spacer(1,12))
            table_data = [["Spieler", "Angriffe", "Erfolgsquote", "Annahmen", "Aufschläge", "Blocks"]] + \
                         [[row["Spieler"], str(row["Angriffe"]), row["Erfolgsquote"], str(row["Annahmen"]), str(row["Aufschläge"]), str(row["Blocks"])] for _, row in df.iterrows()]
            t = Table(table_data)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey), ('GRID',(0,0),(-1,-1),1,colors.black)]))
            story.append(t)
            doc.build(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button("PDF herunterladen", pdf_buffer, file_name="beach_stats.pdf")
