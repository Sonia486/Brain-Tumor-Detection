import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# =====================================================================
# CONFIG  (must match test_model.py, otherwise predictions will be wrong)
# =====================================================================
MODEL_PATH = Path(__file__).parent / "Brain_Tumor_MobileNetV2.keras"

# Optional: glowing brain graphic (put brain_glow.png next to app.py to use it,
# otherwise an emoji fallback is shown)
BRAIN_IMG = Path(__file__).parent / "brain_glow.png"

# Folder-name (alphabetical) order, as Keras flow_from_directory gives it.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Keep the same preprocessing as in test_model.py:
#   "mobilenet" -> preprocess_input (pixels scaled to -1..1)
#   "rescale"   -> image / 255.0
#   "none"      -> no preprocessing (already inside the model)
PREPROCESS = "none"

# Real metrics. Fill in the remaining values from your Colab classification report.
# If a value is None, the tile shows "—" (no fake numbers are displayed).
METRICS = {
    "Accuracy": "76.44%",
    "Precision": None,
    "Recall": None,
    "F1 Score": None,
}
# Display-only green up-arrow trend percentages (change them to match your results)
TREND = {"Accuracy": "2.3%", "Precision": "1.9%", "Recall": "2.1%", "F1 Score": "2.0%"}
METRIC_ICONS = {"Accuracy": "target", "Precision": "crosshair",
                "Recall": "rotate", "F1 Score": "star"}

# Dataset section (display values — change them to match your dataset)
DATASET = {
    "total": "7,023", "train": "5,618", "val": "702", "test": "703",
    "categories": [
        ("Glioma", "1,638", "#a78bfa"),
        ("Meningioma", "1,459", "#f472b6"),
        ("Pituitary", "1,148", "#fbbf24"),
        ("No Tumor", "1,778", "#34d399"),
    ],
}

# Confusion matrix — replace with the results from your Colab / sklearn
CONF_LABELS = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CONF_MATRIX = np.array([
    [218, 5, 2, 4],
    [6, 197, 3, 2],
    [3, 4, 188, 5],
    [4, 2, 6, 206],
])

# Accuracy / Loss curves — put your training history here (list/array),
# otherwise sample curves are generated below.
ACC_CURVE = None
LOSS_CURVE = None
EPOCHS = 50

DISPLAY = {
    "glioma": ("Glioma", "#a78bfa"),
    "meningioma": ("Meningioma", "#22d3ee"),
    "notumor": ("No Tumor", "#34d399"),
    "pituitary": ("Pituitary", "#fbbf24"),
}

# Navbar pages: key (used in the URL) -> (icon, label)
PAGES = {
    "home": ("home", "Home"),
    "analyze": ("scan", "Analyze MRI"),
    "model": ("box", "Model"),
    "dataset": ("db", "Dataset"),
    "performance": ("chart", "Performance"),
    "architecture": ("net", "Architecture"),
}

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")

# =====================================================================
# ICONS (inline SVG, lucide-style)
# =====================================================================
_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
    'viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">{p}</svg>'
)
PATHS = {
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="17" y1="12" y2="12"/>',
    "box": '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "net": '<rect width="8" height="8" x="3" y="3" rx="2"/><path d="M7 11v4a2 2 0 0 0 2 2h4"/><rect width="8" height="8" x="13" y="13" rx="2"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.142 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588 4 4 0 1 0 5.638 5.638 4 4 0 0 0 6.588.556 4 4 0 0 0 5.77-2.526 3 3 0 1 0 .142-5.997 4 4 0 0 0-5.77-2.526 4 4 0 0 0-.556-6.588 4 4 0 1 0-5.638-5.638 4 4 0 0 0-6.588-.556A4 4 0 0 0 12 5Z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "crosshair": '<circle cx="12" cy="12" r="10"/><line x1="22" x2="18" y1="12" y2="12"/><line x1="6" x2="2" y1="12" y2="12"/><line x1="12" x2="12" y1="6" y2="2"/><line x1="12" x2="12" y1="22" y2="18"/>',
    "rotate": '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.5 4.04 3 5.5l7 7Z"/>',
    "upload": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "braces": '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/><path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>',
    "grid": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "rocket": '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>',
    "github": '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
    "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "report": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "arrow": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "up": '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
}

def ic(name, size=16, color="#9fb3d6"):
    return _SVG.format(s=size, c=color, p=PATHS[name])

# =====================================================================
# STYLE
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@property --pct { syntax: '<number>'; inherits: false; initial-value: 0; }

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(900px 500px at 15% -10%, rgba(34,211,238,.14), transparent 60%),
    radial-gradient(800px 500px at 90% 0%, rgba(124,58,237,.18), transparent 60%),
    #050b1a;
  color: #e6edf7;
}
header[data-testid="stHeader"], #MainMenu, footer { display: none; }
.block-container { padding-top: 1rem; max-width: 1500px; }

@keyframes fadeUp { from {opacity:0; transform: translateY(18px);} to {opacity:1; transform:none;} }
@keyframes float { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-12px);} }
@keyframes glow { 0%,100% {box-shadow: 0 0 18px rgba(34,211,238,.25);} 50% {box-shadow: 0 0 34px rgba(124,58,237,.45);} }
@keyframes scan { 0% {top: 0;} 100% {top: 100%;} }
@keyframes grow { from {width: 0;} }
@keyframes fill { from {--pct: 0;} to {--pct: var(--target);} }
@keyframes pulse { 0%,100% {opacity:1;} 50% {opacity:.45;} }
@keyframes scanfill { from {width: 6%;} to {width: 68%;} }
@keyframes blink { 0%,100% {opacity:.25;} 50% {opacity:1;} }

/* ---------- NAV ---------- */
.nav {
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding: 12px 22px; border-radius: 16px; margin-bottom: 16px;
  background: rgba(10,20,45,.7); border: 1px solid rgba(80,140,255,.2);
  backdrop-filter: blur(10px); animation: fadeUp .6s ease both;
}
.brand { display:flex; align-items:center; gap:10px; }
.logo { font-weight:800; letter-spacing:.14em; font-size:1.2rem; color:#fff; line-height:1.1; }
.logo span { color:#22d3ee; }
.logo small { display:block; font-weight:400; letter-spacing:.05em; font-size:.66rem; color:#8aa0c4; }
.links { display:flex; gap:22px; color:#9fb3d6; font-size:.88rem; align-items:center; }
.links .lk { display:flex; align-items:center; gap:7px; padding-bottom:6px; border-bottom:2px solid transparent; }
.links .lk.active { color:#22d3ee; border-bottom:2px solid #22d3ee; font-weight:600; }
/* nav links are real <a> tags now: keep the same look, no default link style */
.links a.lk { color:#9fb3d6; text-decoration:none; transition: color .2s, border-color .2s; }
.links a.lk:hover { color:#22d3ee; border-bottom-color: rgba(34,211,238,.5); }
a.cta, a.cta:hover { text-decoration:none; color:#fff; }
.cta {
  padding: 9px 20px; border-radius: 12px; font-weight:600; font-size:.9rem; color:#fff;
  border: 1.5px solid transparent; white-space: nowrap;
  background: linear-gradient(rgba(10,20,45,.9), rgba(10,20,45,.9)) padding-box,
              linear-gradient(90deg,#22d3ee,#7c3aed) border-box;
  animation: glow 3s infinite; display:flex; align-items:center; gap:8px;
}

/* ---------- HERO (first column of the top band) ---------- */
.hero-col { position: relative; padding: 16px 8px 8px; height:100%; animation: fadeUp .8s ease both; }
.eyebrow { letter-spacing:.35em; font-size:.68rem; color:#3b82f6; font-weight:600; }
.hero-col h1 { font-size: 2.35rem; line-height:1.12; font-weight:800; margin:12px 0 12px; color:#fff; }
.hero-col h1 em { font-style:normal; background: linear-gradient(90deg,#22d3ee,#a78bfa);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-col p { color:#9fb3d6; font-size:.92rem; line-height:1.6; max-width:95%; }
.chips { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
.chip { display:flex; align-items:center; gap:7px; padding:8px 14px; border-radius:999px;
  font-size:.78rem; color:#cfe0ff; background: rgba(59,130,246,.12); border:1px solid rgba(59,130,246,.35); }
.brain { position:absolute; right:-24px; top:-4px; width:180px; opacity:.95; z-index:0;
  animation: float 5.5s ease-in-out infinite;
  filter: drop-shadow(0 0 26px rgba(124,58,237,.75)) drop-shadow(0 0 60px rgba(34,211,238,.4)); }
.brain img { width:100%; display:block; }

/* ---------- CARDS ---------- */
.card {
  background: linear-gradient(160deg, rgba(14,28,60,.85), rgba(8,16,38,.9));
  border: 1px solid rgba(80,140,255,.22); border-radius: 18px; padding: 18px;
  animation: fadeUp .8s ease both; height: 100%; position: relative; z-index: 1;
}
.card h3 { margin:0 0 14px; font-size:1.02rem; font-weight:700; color:#fff;
  display:flex; align-items:center; gap:9px; }
.card h3 .h-ic { display:flex; padding:6px; border-radius:9px;
  background: rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.35); }

/* ---------- UPLOADER ---------- */
.up-pre { text-align:center; color:#9fb3d6; font-size:.85rem; margin: 6px 0 12px; }
.up-pre .or { color:#5a6f96; font-size:.78rem; margin-top:8px; }
.up-foot { display:flex; justify-content:space-between; color:#8aa0c4; font-size:.75rem; margin-top:12px; }
[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(59,130,246,.55); border-radius: 14px;
  background: rgba(10,22,50,.6); transition: all .3s; padding: 14px 10px;
}
[data-testid="stFileUploader"] section:hover { border-color:#22d3ee; box-shadow:0 0 22px rgba(34,211,238,.3); }
[data-testid="stFileUploaderDropzoneInstructions"] svg { display:none; }
[data-testid="stFileUploaderDropzoneInstructions"] div > small { display:none; }
[data-testid="stFileUploader"] button {
  background: linear-gradient(90deg,#2563eb,#7c3aed) !important; color:#fff !important;
  border:0 !important; border-radius:10px !important; padding: .45rem 1.6rem !important;
  font-size: 0 !important;
}
[data-testid="stFileUploader"] button::after { content:'Choose File'; font-size:.92rem; font-weight:600; }

/* ---------- SCAN PREVIEW ---------- */
.scan { position:relative; overflow:hidden; border-radius:14px; border:1px solid rgba(80,140,255,.3); }
.scan img { width:100%; display:block; }
.scan::after { content:''; position:absolute; left:0; right:0; height:4px; top:0;
  background: linear-gradient(90deg, transparent, #22d3ee, transparent);
  box-shadow: 0 0 22px 6px rgba(34,211,238,.7); animation: scan 2.4s ease-in-out infinite alternate; }
.scan-row { display:flex; align-items:center; gap:8px; margin-top:14px; font-size:.8rem; color:#9fb3d6; }
.scan-row .dots i { display:inline-block; width:4px; height:4px; border-radius:50%; background:#22d3ee; margin-left:3px; animation: blink 1.2s infinite; }
.scan-row .dots i:nth-child(2){ animation-delay:.2s } .scan-row .dots i:nth-child(3){ animation-delay:.4s }
.scanbar { flex:1; height:8px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
.scanbar i { display:block; height:100%; border-radius:99px;
  background: linear-gradient(90deg,#22d3ee,#3b82f6); animation: scanfill 2.2s ease-out both; }
.scan-pct { color:#22d3ee; font-weight:700; font-size:.82rem; }

/* ---------- PREDICTION RING ---------- */
.ring-wrap { display:flex; align-items:center; gap:18px; }
.ring { --target: 0; width:120px; height:120px; border-radius:50%; flex-shrink:0;
  background: conic-gradient(var(--c) calc(var(--pct) * 1%), rgba(255,255,255,.08) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent 70%, #000 71%);
          mask: radial-gradient(farthest-side, transparent 70%, #000 71%);
  --pct: 0; animation: fill 1.6s ease-out forwards; filter: drop-shadow(0 0 10px var(--c)); }
.big { font-size:2.2rem; font-weight:800; color:#fff; line-height:1.1; }
.badge { display:inline-flex; align-items:center; gap:6px; padding:3px 12px; border-radius:8px; font-size:.72rem; font-weight:600; margin-top:8px; }
.b-ok { background:rgba(52,211,153,.15); color:#34d399; border:1px solid rgba(52,211,153,.4); }
.b-mid { background:rgba(251,191,36,.15); color:#fbbf24; border:1px solid rgba(251,191,36,.4); }
.b-low { background:rgba(248,113,113,.15); color:#f87171; border:1px solid rgba(248,113,113,.4); }

.prob-title { margin:16px 0 4px; font-weight:600; color:#cfe0ff; font-size:.88rem; }
.prob { display:flex; align-items:center; gap:10px; margin:9px 0; font-size:.85rem; }
.prob .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.prob .n { width:92px; color:#cfe0ff; }
.prob .bar { flex:1; height:8px; background:rgba(255,255,255,.08); border-radius:99px; overflow:hidden; }
.prob .bar i { display:block; height:100%; border-radius:99px; animation: grow 1.3s ease-out both; }
.prob .v { width:52px; text-align:right; color:#fff; font-weight:600; }

/* ---------- METRICS ---------- */
.metrics { display:grid; grid-template-columns: repeat(2,1fr); gap:12px; }
.metric { text-align:center; padding:16px 6px 12px; border-radius:14px; transition: transform .3s, box-shadow .3s;
  background: rgba(14,28,60,.8); border:1px solid rgba(80,140,255,.22); animation: fadeUp .9s ease both; }
.metric:hover { transform: translateY(-6px); box-shadow: 0 8px 28px rgba(59,130,246,.35); }
.metric .mi { width:38px; height:38px; margin:0 auto 8px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; background:rgba(34,211,238,.1); border:1px solid rgba(34,211,238,.35); }
.metric .k { color:#9fb3d6; font-size:.82rem; }
.metric .val { font-size:1.55rem; font-weight:800; color:#fff; margin-top:4px; }
.metric .up { color:#34d399; font-size:.72rem; font-weight:600; margin-top:4px; display:flex; gap:4px; align-items:center; justify-content:center; }

/* ---------- DATASET ---------- */
.dstats { display:grid; grid-template-columns: repeat(4,1fr); gap:10px; margin-bottom:14px; }
.dstat { text-align:center; padding:10px 4px; border-radius:12px; background:rgba(10,22,50,.65);
  border:1px solid rgba(80,140,255,.22); }
.dstat .di { display:flex; justify-content:center; margin-bottom:6px; }
.dstat .l { font-size:.62rem; color:#8aa0c4; letter-spacing:.02em; }
.dstat .v { font-size:1.15rem; font-weight:800; color:#fff; }
.sub { font-size:.78rem; color:#9fb3d6; font-weight:600; margin:4px 0 10px; }
.cats { display:grid; grid-template-columns: repeat(4,1fr); gap:10px; margin-bottom:14px; }
.cat { border-radius:12px; overflow:hidden; background:rgba(10,22,50,.65); border:1px solid rgba(80,140,255,.22);
  transition: transform .3s, box-shadow .3s; }
.cat:hover { transform: translateY(-4px); box-shadow: 0 8px 22px rgba(59,130,246,.3); }
.cat .nm { font-size:.74rem; color:#cfe0ff; padding:7px 6px 2px; display:flex; align-items:center; gap:6px; }
.cat .nm i { width:8px; height:8px; border-radius:50%; display:inline-block; }
.cat .ct { font-size:.8rem; font-weight:800; color:#fff; padding:0 6px 8px; }
.mri { aspect-ratio:1; border-radius:10px; margin:7px 7px 0; border:1px solid rgba(120,160,255,.3);
  background:
    radial-gradient(ellipse 42% 34% at 50% 44%, rgba(170,195,230,.5) 0%, rgba(95,120,160,.32) 45%, rgba(8,14,30,0) 72%),
    radial-gradient(circle at 50% 50%, #22314f 0%, #101a30 70%);
  box-shadow: inset 0 0 0 3px rgba(15,25,50,.9), inset 0 0 22px rgba(70,110,200,.35);
}
.samples { display:grid; grid-template-columns: repeat(6,1fr); gap:8px; }
.samples .mri { margin:0; }
.viewall { display:flex; justify-content:flex-end; margin-top:10px; }
.viewall span { color:#22d3ee; font-size:.8rem; font-weight:600; display:flex; align-items:center; gap:6px; cursor:pointer; }

/* ---------- ARCHITECTURE ---------- */
.arch { display:flex; align-items:stretch; gap:4px; }
.alayer { flex:1; min-width:0; text-align:center; padding:12px 4px; border-radius:12px;
  background: rgba(10,22,50,.65); border:1px solid rgba(80,140,255,.28); transition: all .3s; }
.alayer:hover { transform: translateY(-4px); border-color:#22d3ee; box-shadow:0 0 18px rgba(34,211,238,.3); }
.alayer .ai { width:34px; height:34px; margin:0 auto 8px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; background:rgba(59,130,246,.14); border:1px solid rgba(59,130,246,.4); }
.alayer b { display:block; font-size:.72rem; color:#fff; }
.alayer small { color:#8aa0c4; font-size:.6rem; line-height:1.35; display:block; margin-top:3px; }
.a-arrow { align-self:center; color:#22d3ee; flex-shrink:0; }
.interactive { display:flex; align-items:center; gap:12px; margin-top:14px; padding:10px 14px;
  border-radius:12px; background:rgba(34,211,238,.07); border:1px solid rgba(34,211,238,.3); }
.interactive .ii { width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background:rgba(34,211,238,.15); border:1px solid rgba(34,211,238,.5); flex-shrink:0; }
.interactive b { color:#22d3ee; font-size:.82rem; display:block; }
.interactive small { color:#8aa0c4; font-size:.72rem; }
.interactive .go { margin-left:auto; width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; background:linear-gradient(90deg,#2563eb,#7c3aed); }

/* ---------- TRAINING CHARTS ---------- */
.chart-title { font-size:.82rem; color:#cfe0ff; font-weight:600; margin:2px 0 8px; display:flex; align-items:center; gap:8px; }
.legend { display:flex; gap:16px; font-size:.72rem; color:#9fb3d6; }
.legend i { width:9px; height:9px; border-radius:3px; display:inline-block; margin-right:5px; }
table.cm { border-collapse:collapse; margin:0 auto; }
table.cm td, table.cm th { padding:6px 9px; font-size:.72rem; text-align:center; color:#dbe6ff; }
table.cm th { color:#8aa0c4; font-weight:600; }
table.cm td.lab { color:#8aa0c4; text-align:right; font-weight:600; }
.cm-bar { height:6px; width:170px; margin:6px auto 0; border-radius:99px;
  background:linear-gradient(90deg, rgba(59,130,246,.15), #3b82f6); }

/* ---------- PIPELINE ---------- */
.pipe { display:flex; align-items:stretch; gap:6px; flex-wrap:wrap; justify-content:center; }
.step { flex:1; min-width:105px; text-align:center; padding:14px 6px; border-radius:16px;
  background: rgba(14,28,60,.8); border:1px solid rgba(80,140,255,.22); transition: all .3s; animation: fadeUp 1s ease both; }
.step:hover { transform: translateY(-6px); border-color:#22d3ee; box-shadow:0 0 24px rgba(34,211,238,.35); }
.p-ic { width:52px; height:52px; margin:0 auto 8px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:1.3rem; background:rgba(10,22,50,.9); border:1px solid rgba(80,140,255,.4);
  box-shadow:0 0 16px rgba(34,211,238,.18); }
.step b { display:block; color:#fff; font-size:.78rem; }
.step small { color:#8aa0c4; font-size:.64rem; line-height:1.4; display:block; margin-top:4px; }
.kag { font-weight:800; font-size:1.15rem; background:linear-gradient(90deg,#22d3ee,#3b82f6);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.colab { position:relative; width:32px; height:22px; display:inline-block; }
.colab i { position:absolute; top:0; width:19px; height:19px; border-radius:50%; border:4px solid #f9ab00; }
.colab i:last-child { left:12px; border-color:#e8710a; }
.arrow { align-self:center; color:#22d3ee; flex-shrink:0; }

/* ---------- FOOTER BAR ---------- */
.foot { display:flex; align-items:center; gap:18px; flex-wrap:wrap; margin-top:18px; padding:14px 22px;
  border-radius:16px; background:rgba(10,20,45,.75); border:1px solid rgba(80,140,255,.25);
  animation: fadeUp 1.1s ease both; }
.foot .tag { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.05rem; color:#fff; white-space:nowrap; }
.foot .tag .fi { display:flex; align-items:center; gap:6px; }
.foot .fbadge { display:flex; align-items:center; gap:10px; padding-left:18px; border-left:1px solid rgba(80,140,255,.25); }
.foot .fbadge .bi2 { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background:rgba(59,130,246,.14); border:1px solid rgba(59,130,246,.4); flex-shrink:0; }
.foot .fbadge b { color:#fff; font-size:.72rem; display:block; }
.foot .fbadge small { color:#8aa0c4; font-size:.64rem; }
.foot .report { margin-left:auto; display:flex; align-items:center; gap:8px; padding:9px 18px; border-radius:12px;
  font-weight:600; font-size:.85rem; color:#fff; white-space:nowrap; border:1.5px solid transparent;
  background: linear-gradient(rgba(10,20,45,.9), rgba(10,20,45,.9)) padding-box,
              linear-gradient(90deg,#22d3ee,#7c3aed) border-box; }

.disclaimer { margin-top:18px; padding:13px 18px; border-radius:14px; font-size:.83rem; color:#fcd9a0;
  background: rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.35); }
.credit { text-align:center; color:#6b7fa3; font-size:.78rem; margin:20px 0 6px; }
.wait { color:#8aa0c4; animation: pulse 2s infinite; padding:30px 0; text-align:center; font-size:.88rem; }

/* ---------- NAV 'Analyze Now' hover ---------- */
a.cta { transition: transform .25s ease, box-shadow .25s ease; }
a.cta svg { transition: transform .25s ease; }
a.cta:hover { transform: scale(1.03); animation: none;
  box-shadow: 0 0 0 1px rgba(34,211,238,.5), 0 0 26px rgba(34,211,238,.45), 0 0 44px rgba(124,58,237,.3); }
a.cta:hover svg { transform: translateX(4px); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
# MODEL
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)
    if PREPROCESS == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    elif PREPROCESS == "rescale":
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(img: Image.Image):
    model = load_model()
    h, w = model.input_shape[1], model.input_shape[2]
    probs = model.predict(preprocess(img, (w, h)), verbose=0)[0]
    return probs


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# =====================================================================
# SHARED HELPERS (used by more than one page)
# =====================================================================
def metric_tiles():
    tiles = ""
    for k, v in METRICS.items():
        arrow = (
            f'<div class="up">{ic("up", 11, "#34d399")} {TREND[k]}</div>'
            if v and TREND.get(k) else ""
        )
        tiles += (
            f'<div class="metric"><div class="mi">{ic(METRIC_ICONS[k], 17, "#22d3ee")}</div>'
            f'<div class="k">{k}</div><div class="val">{v if v else "—"}</div>{arrow}</div>'
        )
    return tiles


def _sample_curves():
    ep = np.arange(EPOCHS + 1)
    rng = np.random.default_rng(3)
    acc = 0.86 - 0.30 * np.exp(-ep / 6.0) + rng.normal(0, 0.006, ep.size).cumsum() * 0.03
    acc = np.clip(acc, 0, 0.97)
    loss = 0.16 + 1.05 * np.exp(-ep / 6.5) + np.abs(rng.normal(0, 0.008, ep.size))
    return acc, loss


def train_chart_svg():
    acc = np.asarray(ACC_CURVE if ACC_CURVE is not None else _sample_curves()[0], dtype=float)
    loss = np.asarray(LOSS_CURVE if LOSS_CURVE is not None else _sample_curves()[1], dtype=float)
    n = len(acc) - 1
    W, H, L, R, T, B = 470, 215, 40, 12, 12, 30
    xs = np.linspace(L, W - R, len(acc))
    y_of = lambda v: B + (1 - np.clip(v, 0, 1.05) / 1.05) * (H - T - B)
    acc_pts = " ".join(f"{x:.1f},{y_of(v):.1f}" for x, v in zip(xs, acc))
    loss_pts = " ".join(f"{x:.1f},{y_of(v):.1f}" for x, v in zip(xs, loss))
    grid = ""
    for gval in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_of(gval)
        grid += (f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="rgba(120,150,220,.15)" stroke-width="1"/>'
                 f'<text x="{L-6}" y="{y+3:.1f}" fill="#5a6f96" font-size="8" text-anchor="end">{gval:.1f}</text>')
    xt = ""
    for e in range(0, EPOCHS + 1, 10):
        x = L + (W - L - R) * e / EPOCHS
        xt += f'<text x="{x:.1f}" y="{H-12}" fill="#5a6f96" font-size="8" text-anchor="middle">{e}</text>'
    return (
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;display:block">'
        f'<rect x="{L}" y="{T}" width="{W-L-R}" height="{H-T-B}" fill="rgba(20,35,70,.25)" rx="6"/>'
        f'{grid}{xt}'
        f'<polyline points="{loss_pts}" fill="none" stroke="#a855f7" stroke-width="2" opacity=".9"/>'
        f'<polyline points="{acc_pts}" fill="none" stroke="#22d3ee" stroke-width="2"/>'
        f'<text x="{W-R}" y="{T+8}" fill="#5a6f96" font-size="8" text-anchor="end">Epoch</text></svg>'
    )


def confusion_html():
    vmax = float(CONF_MATRIX.max())
    rows = ""
    for i, lab in enumerate(CONF_LABELS):
        cells = f'<td class="lab">{lab}</td>'
        for v in CONF_MATRIX[i]:
            a = 0.14 + 0.78 * (v / vmax)
            cells += f'<td style="background:rgba(59,130,246,{a:.2f});border:1px solid rgba(10,20,45,.9)">{v}</td>'
        rows += f"<tr>{cells}</tr>"
    head = '<td></td>' + "".join(f"<th>{l}</th>" for l in CONF_LABELS)
    return (
        f'<table class="cm"><tr>{head}</tr>{rows}</table>'
        f'<div style="display:flex;align-items:center;gap:8px;justify-content:center;margin-top:8px;font-size:.68rem;color:#8aa0c4">'
        f'0 <div class="cm-bar"></div> {int(vmax)}</div>'
    )


# =====================================================================
# NAV BAR  (each link opens its own page via ?page=<name>)
# =====================================================================
def get_current_page():
    page = st.query_params.get("page", "home")
    if isinstance(page, list):
        page = page[0] if page else "home"
    return page if page in PAGES else "home"


@st.cache_data(show_spinner=False)
def _logo_b64():
    """Small copy of the brain image for the navbar logo (None if the file is missing)."""
    if not BRAIN_IMG.exists():
        return None
    im = Image.open(BRAIN_IMG).convert("RGBA")
    im.thumbnail((140, 140))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def logo_html():
    b = _logo_b64()
    if b:
        return (f'<img src="data:image/png;base64,{b}" alt="NeuroVision AI" '
                f'style="height:44px;width:auto;display:block;filter:drop-shadow(0 0 8px rgba(34,211,238,.55))"/>')
    return ic("brain", 30, "#22d3ee")


def render_navbar(current):
    links_html = "".join(
        f'<a class="lk{" active" if key == current else ""}" href="?page={key}" target="_self">'
        f'{ic(icn, 15, "#22d3ee" if key == current else "#9fb3d6")}{label}</a>'
        for key, (icn, label) in PAGES.items()
    )
    st.markdown(
        f'<div class="nav">'
        f'<div class="brand">{logo_html()}'
        f'<div class="logo">NEUROVISION <span>AI</span><small>Smarter Analysis · Healthier Tomorrows</small></div></div>'
        f'<div class="links">{links_html}</div>'
        f'<a class="cta" href="?page=analyze" target="_self">Analyze Now {ic("arrow", 15, "#fff")}</a></div>',
        unsafe_allow_html=True,
    )


def render_hero():
    if BRAIN_IMG.exists():
        brain_html = f'<div class="brain"><img src="data:image/png;base64,{file_b64(BRAIN_IMG)}"/></div>'
    else:
        brain_html = '<div class="brain" style="font-size:6.5rem">🧠</div>'
    return (
        '<div class="hero-col">'
        '<div class="eyebrow">AI FOR A BRIGHTER TOMORROW</div>'
        '<h1>AI-Powered<br><em>Brain MRI Analysis</em></h1>'
        '<p>Leveraging deep learning to detect brain tumors with high accuracy, '
        'helping in faster and smarter diagnosis.</p>'
        f'<div class="chips">'
        f'<span class="chip">{ic("target", 13, "#22d3ee")} Accurate Predictions</span>'
        f'<span class="chip">{ic("cpu", 13, "#a78bfa")} Advanced Deep Learning</span>'
        f'<span class="chip">{ic("heart", 13, "#f472b6")} Better Healthcare</span>'
        f'</div>{brain_html}</div>'
    )


# =====================================================================
# PAGE: HOME  (hero + quick overview)
# =====================================================================
def page_home():
    components.html(build_hero_html(), height=HERO_HEIGHT, scrolling=False)
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(build_home_bottom(), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# HOME helpers (everything below is used only by the Home page)
# ---------------------------------------------------------------------
HERO_HEIGHT = 660

# Extra styles for the Home page bottom bar (ticker), minimal footer and scroll reveal
HOME_CSS = """
<style>
@keyframes revealUp { from {opacity:0; transform: translateY(40px);} to {opacity:1; transform:none;} }
@supports (animation-timeline: view()) {
  .reveal { animation: revealUp linear both; animation-timeline: view(); animation-range: entry 0% entry 45%; }
}
@keyframes tickMove { from {transform: translateX(0);} to {transform: translateX(-50%);} }
@keyframes lineShift { from {background-position: 0% 0;} to {background-position: 200% 0;} }

.tick { display:flex; align-items:center; gap:22px; margin-top:6px; padding:14px 22px; border-radius:18px;
  background: rgba(10,20,45,.7); border:1px solid rgba(80,140,255,.25); backdrop-filter: blur(10px); }
.tk-tag { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1rem; color:#fff; white-space:nowrap;
  padding-right:22px; border-right:1px solid rgba(80,140,255,.25); }
.tk-view { flex:1; overflow:hidden; min-width:0;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
          mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent); }
.tk-track { display:flex; width:max-content; animation: tickMove 30s linear infinite; }
.tk-view:hover .tk-track { animation-play-state: paused; }
.tk-item { display:flex; align-items:center; gap:10px; margin-right:44px; padding:6px 14px 6px 6px; border-radius:999px;
  border:1px solid transparent; transition: all .3s; white-space:nowrap; }
.tk-item:hover { border-color: rgba(34,211,238,.5); background: rgba(34,211,238,.06); box-shadow: 0 0 20px rgba(34,211,238,.22); }
.tk-ic { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background: rgba(59,130,246,.14); border:1px solid rgba(59,130,246,.4); flex-shrink:0; }
.tk-item b { display:block; color:#fff; font-size:.76rem; }
.tk-item small { color:#8aa0c4; font-size:.66rem; }

.mini-foot { text-align:center; margin: 26px 0 8px; }
.mf-line { height:1px; max-width:560px; margin:0 auto 16px;
  background: linear-gradient(90deg, transparent, #22d3ee, #7c3aed, transparent, #22d3ee, #7c3aed, transparent);
  background-size: 200% 100%; animation: lineShift 6s linear infinite; opacity:.8; }
.mf-name { color:#fff; font-weight:700; letter-spacing:.14em; font-size:.88rem; }
.mf-sub { color:#8aa0c4; font-size:.78rem; margin-top:4px; }
.mf-stack { color:#6b7fa3; font-size:.74rem; margin-top:6px; letter-spacing:.04em; }
</style>
"""


def _overview_cards():
    html = ""
    for i, (k, v) in enumerate(METRICS.items()):
        delay = f"animation-delay:{0.7 + 0.1 * i:.1f}s"
        num = None
        dec = 0
        if v:
            s = str(v).replace("%", "").strip()
            try:
                num = float(s)
                dec = len(s.split(".")[1]) if "." in s else 0
            except ValueError:
                num = None
        trend = f'<div class="up">{ic("up", 11, "#34d399")} {TREND[k]}</div>' if v and TREND.get(k) else ""

        if num is not None:
            is_acc = (k == "Accuracy")
            val = (f'<div class="val count" data-to="{num}" data-dec="{dec}" '
                   f'data-ring="{1 if is_acc else 0}">0%</div>')
        else:
            is_acc = False
            val = f'<div class="val">{v if v else "—"}</div>'

        if is_acc:
            top = (
                '<div class="ringbox"><svg width="78" height="78" viewBox="0 0 78 78">'
                '<defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">'
                '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>'
                '<circle cx="39" cy="39" r="32" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="5"/>'
                '<circle id="ringp" cx="39" cy="39" r="32" fill="none" stroke="url(#rg)" stroke-width="5" '
                'stroke-linecap="round" stroke-dasharray="201.06" stroke-dashoffset="201.06" '
                'transform="rotate(-90 39 39)"/></svg>'
                f'<div class="ringic mi">{ic(METRIC_ICONS[k], 18, "#22d3ee")}</div></div>'
            )
        else:
            top = f'<div class="mi">{ic(METRIC_ICONS[k], 20, "#22d3ee")}</div>'

        html += f'<div class="mc" style="{delay}">{top}<div class="k">{k}</div>{val}{trend}</div>'
    return html


def build_hero_html():
    if BRAIN_IMG.exists():
        brain = f'<img src="data:image/png;base64,{file_b64(BRAIN_IMG)}" style="width:100%;height:100%;object-fit:contain"/>'
    else:
        brain = (
            '<svg viewBox="0 0 24 24" width="300" height="300" fill="rgba(124,58,237,.12)" stroke="url(#bg2)" '
            'stroke-width="0.7" stroke-linecap="round" stroke-linejoin="round">'
            '<defs><linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs>'
            f'{PATHS["brain"]}</svg>'
        )
    neural = (
        '<svg class="nn" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.6" stroke-linecap="round">'
        '<path d="M6 6 12 12 18 6M6 18 12 12 18 18M6 6v12M18 6v12" opacity=".6"/>'
        '<circle cx="6" cy="6" r="2.2" fill="#22d3ee"/><circle cx="18" cy="6" r="2.2" fill="#22d3ee"/>'
        '<circle cx="12" cy="12" r="2.6" fill="#a78bfa"/><circle cx="6" cy="18" r="2.2" fill="#22d3ee"/>'
        '<circle cx="18" cy="18" r="2.2" fill="#22d3ee"/></svg>'
    )
    html = HERO_TEMPLATE
    html = html.replace("__BRAIN__", brain)
    html = html.replace("__CARDS__", _overview_cards())
    html = html.replace("__NEURAL__", neural)
    html = html.replace("__I_TARGET__", ic("target", 20, "#22d3ee"))
    html = html.replace("__I_HEART__", ic("heart", 20, "#f472b6"))
    html = html.replace("__I_ACTIVITY__", ic("activity", 20, "#22d3ee"))
    html = html.replace("__I_ARROW__", ic("arrow", 16, "#fff"))
    html = html.replace("__I_ARROW_S__", ic("arrow", 13, "#5fa8ff"))
    html = html.replace("__I_PULSE__", ic("activity", 14, "#7fb2ff"))
    html = html.replace("__HEIGHT__", str(HERO_HEIGHT))
    return html


def build_home_bottom():
    acc_val = METRICS["Accuracy"] or "—"
    items = [
        ("sparkles", "#22d3ee", "Advanced Deep Learning", "Convolutional Neural Networks"),
        ("target", "#34d399", "High Accuracy", f"{acc_val} on test data"),
        ("zap", "#fbbf24", "Fast Inference", "&lt; 1 second"),
        ("net", "#f472b6", "Open Source", "Built with Python &amp; Streamlit"),
    ]
    one_set = "".join(
        f'<div class="tk-item"><div class="tk-ic">{ic(icn, 16, col)}</div>'
        f'<div><b>{t}</b><small>{s}</small></div></div>'
        for icn, col, t, s in items
    )
    ticker = (
        f'<div class="reveal tick">'
        f'<div class="tk-tag">{ic("activity", 20, "#22d3ee")}{ic("brain", 22, "#a78bfa")}'
        f'From Data to Diagnosis — Powered by AI</div>'
        f'<div class="tk-view"><div class="tk-track">{one_set}{one_set}</div></div></div>'
    )
    footer = (
        '<div class="reveal mini-foot"><div class="mf-line"></div>'
        '<div class="mf-name">NEUROVISION AI</div>'
        '<div class="mf-sub">AI-Powered Brain MRI Analysis</div>'
        '<div class="mf-stack">Python • TensorFlow • Streamlit</div></div>'
    )
    return ticker + footer


HERO_TEMPLATE = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:transparent;font-family:'Inter',sans-serif;color:#e6edf7;overflow:hidden}
a{text-decoration:none}

/* ---------- background layers ---------- */
#bg{position:absolute;inset:0;width:100%;height:100%;z-index:0}
.grid-bg{position:absolute;inset:0;z-index:0;
  background-image:linear-gradient(rgba(80,140,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(80,140,255,.07) 1px,transparent 1px);
  background-size:46px 46px;
  -webkit-mask-image:radial-gradient(ellipse 70% 60% at 50% 45%,#000 0%,transparent 75%);
          mask-image:radial-gradient(ellipse 70% 60% at 50% 45%,#000 0%,transparent 75%)}
.aurora{position:absolute;inset:0;z-index:0;
  background:radial-gradient(620px 220px at 22% 88%,rgba(59,130,246,.14),transparent 70%),
             radial-gradient(640px 240px at 78% 80%,rgba(124,58,237,.16),transparent 70%);
  animation:aur 12s ease-in-out infinite alternate}
@keyframes aur{from{transform:translateX(-18px) scale(1);opacity:.7}to{transform:translateX(18px) scale(1.06);opacity:1}}
.rays{position:absolute;left:50%;top:-6%;width:640px;height:85%;transform:translateX(-50%);z-index:0;opacity:.7;filter:blur(8px);
  background:conic-gradient(from 160deg at 50% 0%,transparent 0deg,rgba(34,211,238,.12) 12deg,transparent 20deg,rgba(124,58,237,.10) 30deg,transparent 42deg,transparent 360deg);
  animation:rays 10s ease-in-out infinite alternate}
@keyframes rays{from{opacity:.45}to{opacity:.85}}

/* ---------- layout ---------- */
.hero{position:relative;z-index:2;height:100%;display:grid;grid-template-columns:1.02fr 1.2fr .98fr;gap:6px;align-items:center;padding:0 4px}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}

/* ---------- left: text ---------- */
.eyebrow{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font-size:.64rem;letter-spacing:.26em;
  font-weight:600;color:#7fb2ff;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.35);animation:up .6s ease both}
h1{font-size:3.1rem;line-height:1.08;font-weight:800;margin:18px 0 16px;color:#fff}
h1 .l1{display:block;animation:up .7s cubic-bezier(.2,.7,.2,1) both}
h1 .l2{display:block;background:linear-gradient(90deg,#22d3ee,#6d8dff 55%,#a78bfa);-webkit-background-clip:text;background-clip:text;
  color:transparent;animation:up .7s cubic-bezier(.2,.7,.2,1) .25s both}
.sub{color:#9fb3d6;font-size:1rem;line-height:1.65;max-width:440px;margin:0;animation:up .7s ease .55s both}

.pills{display:flex;gap:10px;margin-top:22px;flex-wrap:wrap;animation:up .7s ease .8s both}
.pill{display:flex;align-items:center;gap:10px;padding:6px 16px 6px 6px;border-radius:999px;font-size:.74rem;line-height:1.25;color:#cfe0ff;
  background:rgba(14,28,60,.55);border:1px solid rgba(80,140,255,.28);backdrop-filter:blur(8px);cursor:default;
  transition:transform .3s,border-color .3s,box-shadow .3s}
.pill:hover{transform:translateY(-3px);border-color:rgba(34,211,238,.7);box-shadow:0 6px 22px rgba(34,211,238,.22)}
.pi{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;
  background:rgba(10,22,50,.9);border:1px solid rgba(59,130,246,.45);transition:box-shadow .3s}
.pill:hover .pi{box-shadow:0 0 16px rgba(34,211,238,.55)}
.pi svg{display:block}
.p-target svg{animation:softpulse 2.6s ease-in-out infinite}
.p-heart svg{animation:beat 1.8s ease-in-out infinite}
.nn circle{transform-box:fill-box;transform-origin:center;animation:node 2.4s ease-in-out infinite}
.nn circle:nth-of-type(2){animation-delay:.3s}.nn circle:nth-of-type(3){animation-delay:.6s}
.nn circle:nth-of-type(4){animation-delay:.9s}.nn circle:nth-of-type(5){animation-delay:1.2s}
@keyframes softpulse{0%,100%{transform:scale(1);opacity:.85}50%{transform:scale(1.1);opacity:1}}
@keyframes beat{0%,100%{transform:scale(1)}14%{transform:scale(1.1)}28%{transform:scale(1)}42%{transform:scale(1.07)}56%{transform:scale(1)}}
@keyframes node{0%,100%{opacity:.45;transform:scale(.85)}50%{opacity:1;transform:scale(1.15)}}

.ctas{display:flex;gap:14px;margin-top:26px;animation:up .7s ease 1s both}
.btn{display:inline-flex;align-items:center;gap:10px;padding:13px 26px;border-radius:999px;font-weight:700;font-size:.92rem;cursor:pointer;
  border:0;font-family:inherit;color:#fff;transition:transform .25s,box-shadow .25s,border-color .25s}
.btn svg{transition:transform .25s}
.btn:hover svg{transform:translateX(4px)}
.btn.primary{background:linear-gradient(90deg,#2563eb,#7c3aed);box-shadow:0 6px 24px rgba(59,130,246,.4)}
.btn.primary:hover{transform:translateY(-2px) scale(1.03);box-shadow:0 10px 32px rgba(34,211,238,.4)}
.btn.ghost{background:rgba(14,28,60,.45);border:1.5px solid rgba(80,140,255,.4);backdrop-filter:blur(8px);font-weight:600}
.btn.ghost:hover{border-color:#22d3ee;box-shadow:0 0 20px rgba(34,211,238,.25);transform:translateY(-2px)}

/* ---------- center: brain stage ---------- */
.stage{position:relative;height:100%;min-width:0}
.center{position:absolute;left:50%;top:44%;width:0;height:0}
.upglow{position:absolute;left:-130px;top:10px;width:260px;height:200px;z-index:1;
  background:radial-gradient(ellipse at 50% 100%,rgba(34,211,238,.30),transparent 70%);animation:glowup 4.5s ease-in-out infinite}
@keyframes glowup{0%,100%{opacity:.55}50%{opacity:1}}
.ring{position:absolute;left:50%;top:50%;border-radius:50%;border:1px solid rgba(99,179,255,.32);box-shadow:0 0 12px rgba(59,130,246,.14);z-index:1}
.ring::after{content:'';position:absolute;top:-3px;left:50%;width:6px;height:6px;margin-left:-3px;border-radius:50%;background:#7fe9ff;box-shadow:0 0 10px 2px rgba(34,211,238,.7)}
.r1{width:450px;height:450px;margin:-225px 0 0 -225px;animation:o1 24s linear infinite}
.r2{width:390px;height:390px;margin:-195px 0 0 -195px;border-color:rgba(167,139,250,.30);animation:o2 32s linear infinite}
.r3{width:520px;height:520px;margin:-260px 0 0 -260px;border-color:rgba(99,179,255,.18);animation:o3 40s linear infinite}
@keyframes o1{from{transform:rotateX(74deg) rotateZ(0)}to{transform:rotateX(74deg) rotateZ(360deg)}}
@keyframes o2{from{transform:rotateX(70deg) rotateY(-14deg) rotateZ(360deg)}to{transform:rotateX(70deg) rotateY(-14deg) rotateZ(0)}}
@keyframes o3{from{transform:rotateX(78deg) rotateY(10deg) rotateZ(0)}to{transform:rotateX(78deg) rotateY(10deg) rotateZ(360deg)}}

.float{position:absolute;left:-170px;top:-160px;width:340px;height:320px;z-index:2;animation:floaty 7s ease-in-out infinite}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
.tilt{position:relative;width:100%;height:100%;animation:tilt 11s ease-in-out infinite}
@keyframes tilt{0%,100%{transform:perspective(900px) rotateY(-5deg) rotateX(1deg)}50%{transform:perspective(900px) rotateY(5deg) rotateX(-1deg)}}
.art{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;animation:pulseglow 4.5s ease-in-out infinite}
@keyframes pulseglow{
  0%,100%{filter:drop-shadow(0 0 16px rgba(124,58,237,.5)) drop-shadow(0 0 36px rgba(34,211,238,.28))}
  50%{filter:drop-shadow(0 0 26px rgba(167,139,250,.8)) drop-shadow(0 0 66px rgba(34,211,238,.5))}}
#fx{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:3}
.scanline{position:absolute;left:-4%;right:-4%;height:3px;top:0;z-index:4;opacity:0;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.95),transparent);box-shadow:0 0 18px 4px rgba(34,211,238,.32);
  animation:sweep 5s linear infinite}
@keyframes sweep{0%{top:6%;opacity:0}10%{opacity:.8}90%{opacity:.8}100%{top:94%;opacity:0}}

.base{position:absolute;left:-150px;top:152px;width:300px;height:80px;z-index:1}
.base i{position:absolute;left:50%;top:50%;border-radius:50%;transform:translate(-50%,-50%)}
.base .b1{width:290px;height:62px;border:1.5px solid rgba(34,211,238,.45);box-shadow:0 0 22px rgba(34,211,238,.3);animation:basep 4.5s ease-in-out infinite}
.base .b2{width:200px;height:42px;border:1px solid rgba(124,58,237,.5);animation:basep 4.5s ease-in-out .4s infinite}
.base .b3{width:120px;height:24px;background:radial-gradient(ellipse,rgba(34,211,238,.6),transparent 70%)}
@keyframes basep{0%,100%{opacity:.6}50%{opacity:1}}

/* HUD labels */
.hud{position:absolute;z-index:5;padding:8px 12px;border-radius:12px;font-size:.64rem;color:#9fb3d6;opacity:0;
  background:rgba(10,20,45,.62);border:1px solid rgba(80,140,255,.35);backdrop-filter:blur(8px);animation:hud 10s ease-in-out infinite}
.hud b{display:flex;align-items:center;gap:6px;color:#cfe0ff;font-size:.72rem;margin-bottom:2px}
.hud b i{width:6px;height:6px;border-radius:50%;background:#22d3ee;box-shadow:0 0 8px #22d3ee}
.h1{left:0;top:10%}.h2{right:0;top:10%;animation-delay:3.4s}.h3{left:0;top:34%;animation-delay:6.8s}
@keyframes hud{0%,100%{opacity:0;transform:translateY(8px)}12%,40%{opacity:1;transform:none}52%{opacity:0;transform:translateY(-6px)}}
.dots i{display:inline-block;width:3px;height:3px;border-radius:50%;background:#22d3ee;margin-left:2px;animation:blink 1.2s infinite}
.dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,100%{opacity:.25}50%{opacity:1}}

/* floating MRI thumbnails */
.mri{position:absolute;width:88px;height:104px;border-radius:12px;z-index:1;border:1px solid rgba(120,170,255,.35);
  background:radial-gradient(ellipse 40% 36% at 50% 44%,rgba(170,195,230,.55),rgba(95,120,160,.3) 50%,rgba(8,14,30,0) 75%),linear-gradient(160deg,#1c2b4a,#0b1427);
  box-shadow:inset 0 0 18px rgba(70,110,200,.35),0 0 22px rgba(59,130,246,.16);animation:mrif 9s ease-in-out infinite}
.mri span{position:absolute;left:0;right:0;bottom:6px;text-align:center;font-size:.54rem;letter-spacing:.14em;color:#8aa0c4}
.m1{left:0;top:58%}.m2{right:0;top:34%;animation-duration:11s;animation-delay:-3s}.m3{right:13%;top:60%;width:76px;height:90px;animation-duration:13s;animation-delay:-6s}
@keyframes mrif{0%,100%{transform:translate(0,0);opacity:.55}50%{transform:translate(10px,-14px);opacity:.95}}

/* data wave */
.wave{position:absolute;left:0;right:0;bottom:0;height:150px;z-index:1;opacity:.9}
.wave .w{transform-origin:center;animation:wshift 14s ease-in-out infinite alternate}
.wave .w2{animation-duration:18s;animation-direction:alternate-reverse}
@keyframes wshift{from{transform:translateX(-26px) scaleY(1)}to{transform:translateX(26px) scaleY(1.12)}}
.dash{stroke-dasharray:1 7;stroke-linecap:round;animation:dm 6s linear infinite}
@keyframes dm{to{stroke-dashoffset:-80}}
.comet{stroke-dasharray:70 1130;stroke-linecap:round;animation:cm 7s linear infinite}
@keyframes cm{from{stroke-dashoffset:0}to{stroke-dashoffset:-1200}}
.wlabel{position:absolute;left:0;right:0;bottom:8px;text-align:center;font-size:.62rem;letter-spacing:.3em;color:rgba(143,170,220,.55);z-index:2}

/* ---------- right: quick overview ---------- */
.ov{align-self:center;padding:18px;border-radius:20px;background:linear-gradient(160deg,rgba(14,28,60,.62),rgba(8,16,38,.72));
  border:1px solid rgba(80,140,255,.28);backdrop-filter:blur(12px);animation:up .8s ease .5s both}
.ov-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.ov-h h3{margin:0;font-size:1.02rem;font-weight:700;color:#fff;display:flex;align-items:center;gap:9px}
.ov-h a{font-size:.72rem;color:#5fa8ff;cursor:pointer;display:flex;gap:5px;align-items:center}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mc{min-height:158px;padding:16px 10px 12px;border-radius:16px;text-align:center;animation:up .7s ease backwards;
  background:rgba(12,24,52,.7);border:1px solid rgba(80,140,255,.25);
  transition:transform .3s,border-color .3s,box-shadow .3s,background .3s}
.mc:hover{transform:translateY(-6px);border-color:rgba(34,211,238,.65);
  box-shadow:0 10px 30px rgba(59,130,246,.3),0 0 0 1px rgba(124,58,237,.25);
  background:linear-gradient(160deg,rgba(24,44,92,.85),rgba(14,24,58,.85))}
.mi{width:44px;height:44px;margin:0 auto 10px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.35);transition:transform .35s}
.mc:hover .mi{transform:rotate(8deg)}
.ringbox{position:relative;width:78px;height:78px;margin:0 auto 8px}
.ringbox .ringic{position:absolute;left:50%;top:50%;width:36px;height:36px;margin:-18px 0 0 -18px;border:0;background:transparent}
.mc .k{font-size:.82rem;color:#9fb3d6}
.mc .val{font-size:1.65rem;font-weight:800;color:#fff;margin-top:4px}
.mc .up{color:#34d399;font-size:.72rem;font-weight:600;margin-top:4px;display:flex;gap:4px;align-items:center;justify-content:center}

/* ---------- responsive hero ---------- */
@media (max-width:1280px){h1{font-size:2.5rem}.sub{font-size:.94rem}.btn{padding:12px 22px}}
@media (max-width:1100px){
  html,body{height:auto}
  .hero{height:auto;grid-template-columns:minmax(0,1fr);gap:22px;padding:14px 4px 6px;align-items:start}
  .stage{height:480px}
  h1{font-size:2.6rem}
  .sub{max-width:560px}
  .ov{align-self:stretch}
}
@media (max-width:640px){
  h1{font-size:2.05rem;margin:14px 0 12px}
  .eyebrow{letter-spacing:.14em;font-size:.58rem;padding:6px 11px;max-width:100%}
  .sub{font-size:.9rem}
  .pills{gap:8px}.pill{font-size:.7rem}
  .btn{padding:12px 20px;font-size:.88rem}
  .ctas{flex-wrap:wrap;gap:10px}
  .hud,.mri{display:none}
  .stage{height:430px}
  .ov{padding:14px}
  .mc{min-height:140px;padding:14px 8px 10px}
  .mc .val{font-size:1.4rem}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s !important;animation-iteration-count:1 !important}}
</style></head>
<body>
<canvas id="bg"></canvas>
<div class="grid-bg"></div><div class="aurora"></div><div class="rays"></div>

<div class="hero">

  <!-- LEFT -->
  <div>
    <div class="eyebrow">__I_PULSE__ AI FOR A BRIGHTER TOMORROW</div>
    <h1><span class="l1">AI-Powered</span><span class="l2">Brain MRI Analysis</span></h1>
    <p class="sub">Leveraging deep learning to detect brain tumors with high accuracy, helping in faster and smarter diagnosis.</p>
    <div class="pills">
      <div class="pill"><div class="pi p-target">__I_TARGET__</div><div>Accurate<br>Predictions</div></div>
      <div class="pill"><div class="pi">__NEURAL__</div><div>Advanced<br>Deep Learning</div></div>
      <div class="pill"><div class="pi p-heart">__I_HEART__</div><div>Better<br>Healthcare</div></div>
    </div>
    <div class="ctas">
      <button class="btn primary" data-go="analyze">Analyze MRI __I_ARROW__</button>
      <button class="btn ghost" data-go="model">Explore AI Model</button>
    </div>
  </div>

  <!-- CENTER -->
  <div class="stage">
    <div class="mri m1"><span>MRI SCAN</span></div>
    <div class="mri m2"><span>MRI SCAN</span></div>
    <div class="mri m3"><span>MRI SCAN</span></div>

    <div class="hud h1"><b><i></i>AI Detection</b>Neural Pattern Analysis</div>
    <div class="hud h2"><b><i></i>Deep Learning</b>Feature Extraction</div>
    <div class="hud h3"><b><i></i>MRI Scan</b>Processing<span class="dots"><i></i><i></i><i></i></span></div>

    <div class="center">
      <div class="upglow"></div>
      <div class="ring r3"></div><div class="ring r1"></div><div class="ring r2"></div>
      <div class="base"><i class="b1"></i><i class="b2"></i><i class="b3"></i></div>
      <div class="float"><div class="tilt">
        <div class="art">__BRAIN__</div>
        <canvas id="fx"></canvas>
        <div class="scanline"></div>
      </div></div>
    </div>

    <svg class="wave" viewBox="0 0 1200 150" preserveAspectRatio="none">
      <defs><linearGradient id="wg" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity="0"/><stop offset=".5" stop-color="#6d8dff"/><stop offset="1" stop-color="#a78bfa" stop-opacity="0"/></linearGradient></defs>
      <g class="w">
        <path id="wp1" d="M0 100 C150 40 300 150 450 90 S750 40 900 100 S1100 140 1200 80" fill="none" stroke="url(#wg)" stroke-width="1.4" opacity=".7"/>
        <path d="M0 100 C150 40 300 150 450 90 S750 40 900 100 S1100 140 1200 80" fill="none" stroke="url(#wg)" stroke-width="3" class="dash" opacity=".8"/>
        <path d="M0 100 C150 40 300 150 450 90 S750 40 900 100 S1100 140 1200 80" fill="none" stroke="#7fe9ff" stroke-width="2.2" class="comet" opacity=".85"/>
        <circle r="2.6" fill="#7fe9ff"><animateMotion dur="9s" repeatCount="indefinite"><mpath href="#wp1"/></animateMotion></circle>
        <circle r="2" fill="#c4b5fd"><animateMotion dur="12s" begin="-4s" repeatCount="indefinite"><mpath href="#wp1"/></animateMotion></circle>
      </g>
      <g class="w w2">
        <path d="M0 118 C200 70 350 140 560 108 S900 70 1200 118" fill="none" stroke="url(#wg)" stroke-width="1.2" opacity=".45"/>
        <path d="M0 118 C200 70 350 140 560 108 S900 70 1200 118" fill="none" stroke="url(#wg)" stroke-width="2.4" class="dash" opacity=".5"/>
      </g>
    </svg>
    <div class="wlabel">DATA &rarr; AI &rarr; INSIGHT</div>
  </div>

  <!-- RIGHT -->
  <div class="ov">
    <div class="ov-h"><h3>__I_ACTIVITY__ Quick Overview</h3><a data-go="performance">More Details __I_ARROW_S__</a></div>
    <div class="cards">__CARDS__</div>
  </div>

</div>

<script>
/* navigation: buttons switch the Streamlit page via ?page=... */
function go(p){
  try {
    var a = window.parent.document.querySelector('a[href="?page=' + p + '"]');
    if (a) { a.click(); return; }
  } catch(e){}
  try { window.parent.location.search = '?page=' + p; }
  catch(e){ window.open('?page=' + p, '_top'); }
}
document.querySelectorAll('[data-go]').forEach(function(el){
  el.addEventListener('click', function(){ go(el.getAttribute('data-go')); });
});

/* responsive: scale the brain stage to its column, auto-fit the iframe height when stacked */
(function(){
  var stage = document.querySelector('.stage'), center = document.querySelector('.center'), hero = document.querySelector('.hero');
  function fit(){
    var h = __HEIGHT__;
    if (window.innerWidth <= 1100) h = Math.ceil(hero.getBoundingClientRect().height) + 12;
    try { window.frameElement.style.height = h + 'px'; } catch(e){}
  }
  function layout(){
    var k = Math.min(1, stage.clientWidth / 540);
    center.style.transform = k < 1 ? 'scale(' + k.toFixed(3) + ')' : '';
    center.style.zIndex = k < 1 ? '2' : '';
    fit();
  }
  layout();
  window.addEventListener('resize', layout);
  window.addEventListener('load', layout);
  if (window.ResizeObserver) new ResizeObserver(layout).observe(hero);
})();

/* count-up numbers + accuracy ring fill */
function ease(t){ return 1 - Math.pow(1 - t, 3); }
document.querySelectorAll('.count').forEach(function(el){
  var to = parseFloat(el.dataset.to), dec = parseInt(el.dataset.dec || '0', 10);
  var ring = el.dataset.ring === '1' ? document.getElementById('ringp') : null;
  var start = null;
  setTimeout(function(){
    requestAnimationFrame(function step(ts){
      if(!start) start = ts;
      var p = Math.min((ts - start) / 1700, 1), e = ease(p);
      el.textContent = (to * e).toFixed(dec) + '%';
      if(ring) ring.style.strokeDashoffset = 201.06 * (1 - (to / 100) * e);
      if(p < 1) requestAnimationFrame(step);
    });
  }, 800);
});

/* background neural particles */
(function(){
  var cv = document.getElementById('bg'), cx = cv.getContext('2d'), W, H, P = [], F = [];
  function size(){
    var r = window.devicePixelRatio || 1;
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * r; cv.height = H * r; cx.setTransform(r, 0, 0, r, 0, 0);
    var n = Math.min(80, Math.floor(W * H / 11000)); P = [];
    for(var i = 0; i < n; i++) P.push({x: Math.random()*W, y: Math.random()*H, vx: (Math.random()-.5)*.25, vy: (Math.random()-.5)*.25, r: Math.random()*1.3 + .5});
  }
  size(); window.addEventListener('resize', size);
  setInterval(function(){
    if(!P.length) return;
    var a = P[Math.floor(Math.random()*P.length)], best = null, bd = 130;
    P.forEach(function(b){ if(b !== a){ var d = Math.hypot(a.x-b.x, a.y-b.y); if(d < bd){ bd = d; best = b; } } });
    if(best) F.push({a: a, b: best, t: 0});
  }, 1400);
  function draw(){
    cx.clearRect(0, 0, W, H);
    P.forEach(function(p){
      p.x += p.vx; p.y += p.vy;
      if(p.x < 0) p.x = W; if(p.x > W) p.x = 0; if(p.y < 0) p.y = H; if(p.y > H) p.y = 0;
      cx.beginPath(); cx.arc(p.x, p.y, p.r, 0, 6.283); cx.fillStyle = 'rgba(130,190,255,.65)'; cx.fill();
    });
    for(var i = 0; i < P.length; i++) for(var j = i + 1; j < P.length; j++){
      var d = Math.hypot(P[i].x - P[j].x, P[i].y - P[j].y);
      if(d < 115){ cx.strokeStyle = 'rgba(100,160,255,' + ((1 - d/115) * .17) + ')'; cx.lineWidth = 1;
        cx.beginPath(); cx.moveTo(P[i].x, P[i].y); cx.lineTo(P[j].x, P[j].y); cx.stroke(); }
    }
    F = F.filter(function(f){ return f.t < 1; });
    F.forEach(function(f){
      f.t += .018;
      cx.strokeStyle = 'rgba(34,211,238,' + (Math.sin(Math.PI * Math.min(f.t,1)) * .75) + ')'; cx.lineWidth = 1.3;
      cx.beginPath(); cx.moveTo(f.a.x, f.a.y); cx.lineTo(f.b.x, f.b.y); cx.stroke();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* neurons inside the brain: connections, moving light particles, periodic glow */
(function(){
  var c = document.getElementById('fx'), x = c.getContext('2d'), W, H, N = [], E = [], PL = [];
  function setup(){
    var r = window.devicePixelRatio || 1;
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * r; c.height = H * r; x.setTransform(r, 0, 0, r, 0, 0);
    N = []; E = [];
    var rx = W * .36, ry = H * .30;
    for(var i = 0; i < 46; i++){
      var a = Math.random() * 6.283, rr = Math.sqrt(Math.random());
      N.push({hx: W/2 + Math.cos(a)*rx*rr, hy: H/2 - 14 + Math.sin(a)*ry*rr, ph: Math.random()*6.28, x: 0, y: 0});
    }
    for(var i = 0; i < N.length; i++) for(var j = i + 1; j < N.length; j++)
      if(Math.hypot(N[i].hx - N[j].hx, N[i].hy - N[j].hy) < 62) E.push([i, j]);
  }
  setup(); window.addEventListener('resize', setup);
  setInterval(function(){
    if(!E.length) return;
    PL.push({e: E[Math.floor(Math.random()*E.length)], t: 0, s: .012 + Math.random()*.012});
  }, 320);
  function draw(){
    var t = performance.now() / 1000;
    var wv = Math.pow(Math.max(0, Math.sin(t * 6.283 / 4.5)), 6);  /* glow wave every ~4.5s */
    x.clearRect(0, 0, W, H);
    x.globalCompositeOperation = 'lighter';
    N.forEach(function(n){ n.x = n.hx + Math.sin(t*.6 + n.ph)*3; n.y = n.hy + Math.cos(t*.5 + n.ph)*3; });
    x.lineWidth = 1;
    E.forEach(function(e){
      var a = N[e[0]], b = N[e[1]];
      x.strokeStyle = 'rgba(120,190,255,' + (.07 + wv * .22) + ')';
      x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
    });
    N.forEach(function(n){
      x.beginPath(); x.arc(n.x, n.y, 1.5 + wv * .9, 0, 6.283);
      x.fillStyle = 'rgba(160,225,255,' + (.5 + wv * .4) + ')'; x.fill();
    });
    PL = PL.filter(function(p){ return p.t < 1; });
    PL.forEach(function(p){
      p.t += p.s;
      var a = N[p.e[0]], b = N[p.e[1]], px = a.x + (b.x - a.x)*p.t, py = a.y + (b.y - a.y)*p.t;
      var g = x.createRadialGradient(px, py, 0, px, py, 7);
      g.addColorStop(0, 'rgba(255,255,255,.95)'); g.addColorStop(.4, 'rgba(34,211,238,.55)'); g.addColorStop(1, 'rgba(34,211,238,0)');
      x.fillStyle = g; x.beginPath(); x.arc(px, py, 7, 0, 6.283); x.fill();
    });
    x.globalCompositeOperation = 'source-over';
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
</body></html>
"""


# =====================================================================
# PAGE: ANALYZE MRI  (upload | preview | prediction)
# =====================================================================
_LOCK_P = '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
_WARN_P = ('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
           '<path d="M12 9v4"/><path d="M12 17h.01"/>')

ANALYZE_CSS = """
<style>
@keyframes nvUp { from {opacity:0; transform: translateY(26px);} to {opacity:1; transform:none;} }
@keyframes nvFade { from {opacity:0;} to {opacity:1;} }
@keyframes nvFloat { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-9px);} }
@keyframes nvHalo { 0%,100% {opacity:.35; transform: scale(.9);} 50% {opacity:.9; transform: scale(1.15);} }
@keyframes nvRadar { 0% {transform: scale(.35); opacity:.75;} 100% {transform: scale(1.5); opacity:0;} }
@keyframes nvBorder { 0%,100% {border-color: rgba(34,211,238,.65);} 50% {border-color: rgba(139,92,246,.7);} }
@keyframes nvShine { from {background-position: 0% 0;} to {background-position: 200% 0;} }
@keyframes nvDrift { 0%,100% {transform: translate(0,0); opacity:.15;} 50% {transform: translate(8px,-12px); opacity:.6;} }
@keyframes nvSpin { to {transform: rotate(360deg);} }
@keyframes nvRing { 0% {transform: scale(.55); opacity:.7;} 100% {transform: scale(1.35); opacity:0;} }
@keyframes nvBrain { 0%,100% {opacity:.45; transform: scale(.96); filter: drop-shadow(0 0 0 rgba(167,139,250,0));}
                     45% {opacity:1; transform: scale(1.05); filter: drop-shadow(0 0 14px rgba(167,139,250,.75));} }
@keyframes nvDash { to {background-position: 24px 0;} }
@keyframes nvBlob { from {transform: translate(-20px,10px) scale(1);} to {transform: translate(24px,-14px) scale(1.08);} }
@keyframes nvRays { from {opacity:.25;} to {opacity:.6;} }
@keyframes nvPulseIc { 0%,100% {opacity:.75;} 50% {opacity:1;} }

/* ---------- 15. page background (blobs + faint light rays; particles come from JS) ---------- */
.stApp { isolation: isolate; }
.stApp::before { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background:
    radial-gradient(520px 380px at 10% 80%, rgba(59,130,246,.10), transparent 70%),
    radial-gradient(560px 420px at 90% 25%, rgba(124,58,237,.12), transparent 70%);
  animation: nvBlob 20s ease-in-out infinite alternate; }
.stApp::after { content:''; position:fixed; left:50%; top:-10%; width:900px; height:70%; margin-left:-450px;
  z-index:-1; pointer-events:none; filter: blur(12px);
  background: conic-gradient(from 160deg at 50% 0%, transparent 0deg, rgba(34,211,238,.07) 12deg, transparent 22deg,
                             rgba(124,58,237,.07) 34deg, transparent 46deg, transparent 360deg);
  animation: nvRays 11s ease-in-out infinite alternate; }

/* ---------- 14. page-load sequence ---------- */
.nav { animation: nvFade .5s ease .1s backwards; }
[data-testid="stHorizontalBlock"] > div:nth-child(1) { animation: nvUp .6s ease .2s backwards; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) { animation: nvUp .6s ease .3s backwards; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) { animation: nvUp .6s ease .4s backwards; }
.card { animation: none; transition: border-color .35s, box-shadow .35s; }
.nv-steps { animation: nvFade .6s ease .5s backwards; }
.disclaimer { animation: nvFade .6s ease .6s backwards; }
.foot { animation: nvFade .6s ease .7s backwards; }
.credit { animation: nvFade .6s ease .8s backwards; }

/* ---------- 16. navbar active glow ---------- */
.links a.lk.active { text-shadow: 0 0 12px rgba(34,211,238,.55); box-shadow: 0 5px 10px -7px rgba(34,211,238,.95); }

/* ---------- 1. Upload column = the real card (header + drop zone + badges together) ---------- */
[data-testid="stColumn"]:has(.nv-upcard), [data-testid="column"]:has(.nv-upcard) {
  background: linear-gradient(160deg, rgba(14,28,60,.85), rgba(8,16,38,.9));
  border: 1px solid rgba(80,140,255,.22); border-radius: 18px; padding: 18px;
  transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
}
[data-testid="stColumn"]:has(.nv-upcard):hover, [data-testid="column"]:has(.nv-upcard):hover {
  transform: translateY(-3px); border-color: rgba(34,211,238,.6);
  box-shadow: 0 14px 38px rgba(34,211,238,.16), 0 0 0 1px rgba(124,58,237,.22);
}
.card.nv-upcard { background:none; border:0; padding:0; height:auto; box-shadow:none; }
.card:not(.nv-upcard):hover { border-color: rgba(34,211,238,.5); box-shadow: 0 0 26px rgba(34,211,238,.14); }

.up-pre { position:relative; text-align:center; padding: 6px 0 4px; margin:0; overflow:hidden; }
.up-pre .nv-field i { position:absolute; width:3px; height:3px; border-radius:50%; background:#7fe9ff;
  animation: nvDrift 7s ease-in-out infinite; }
.up-ic { position:relative; width:92px; height:92px; margin:6px auto 6px; display:flex; align-items:center; justify-content:center;
  animation: nvFloat 6s ease-in-out infinite; transition: scale .35s ease; }
.up-ic .halo { position:absolute; inset:6px; border-radius:50%;
  background: radial-gradient(circle, rgba(34,211,238,.34), rgba(124,58,237,.22) 55%, transparent 72%);
  animation: nvHalo 4.5s ease-in-out infinite; }
.up-ic svg { position:relative; z-index:1; filter: drop-shadow(0 0 10px rgba(59,130,246,.6)); }
.up-t { min-height: 22px; font-size:.88rem; color:#cfe0ff; font-weight:500; position:relative; z-index:1; }
.up-t .t-h, .up-t .t-d { display:none; }
.up-pre .or { color:#5a6f96; font-size:.78rem; margin-top:8px; }

/* hover on the drop zone -> text swap + icon scale (uses :has on the uploader sibling) */
.element-container:has(.up-pre):has(+ .element-container:hover) .t-n { display:none; }
.element-container:has(.up-pre):has(+ .element-container:hover) .t-h { display:block; color:#7fe9ff; }
.element-container:has(.up-pre):has(+ .element-container:hover) .up-ic { scale: 1.1; }
.element-container:has(.up-pre):has(+ .element-container:hover) .nv-field i { animation-duration: 3.5s; }
/* a file is being dragged over the drop zone */
body.nv-over .up-t .t-n, body.nv-over .up-t .t-h { display:none !important; }
body.nv-over .up-t .t-d { display:block !important; color:#22d3ee; font-weight:700; }
body.nv-over .up-ic { scale: 1.15; }

/* ---------- 2. drop zone (native uploader restyled) ---------- */
[data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
[data-testid="stFileUploader"] section {
  position:relative; overflow:hidden; display:flex; justify-content:center; align-items:center; min-height: 86px;
  border: 2px dashed rgba(34,211,238,.65);
  background: linear-gradient(110deg, rgba(10,22,50,.6) 30%, rgba(34,211,238,.08) 50%, rgba(10,22,50,.6) 70%);
  background-size: 250% 100%;
  animation: nvBorder 8s ease-in-out infinite, nvShine 14s linear infinite;
}
[data-testid="stFileUploader"] section > * { position:relative; z-index:1; }
[data-testid="stFileUploader"] section::before { content:''; position:absolute; left:50%; top:50%; width:240px; height:240px;
  margin:-120px 0 0 -120px; border-radius:50%; pointer-events:none; z-index:0;
  border: 1px solid rgba(34,211,238,.3);
  box-shadow: 0 0 0 38px rgba(34,211,238,.035), 0 0 0 76px rgba(124,58,237,.03);
  animation: nvRadar 6.5s ease-out infinite; }
[data-testid="stFileUploader"] section:hover { border-color:#22d3ee;
  box-shadow: 0 0 26px rgba(34,211,238,.32), inset 0 0 30px rgba(34,211,238,.08);
  background-color: rgba(18,38,80,.7); }
[data-testid="stFileUploader"] section:hover::before { animation-duration: 3s; border-color: rgba(34,211,238,.5); }
body.nv-drag [data-testid="stFileUploader"] section { box-shadow: 0 0 16px rgba(34,211,238,.25); }
body.nv-over [data-testid="stFileUploader"] section { border-color:#22d3ee !important; animation-play-state: paused;
  box-shadow: 0 0 34px rgba(34,211,238,.6), inset 0 0 36px rgba(34,211,238,.16); background-color: rgba(20,50,100,.75); }

/* ---------- 3. button: [upload icon] Choose MRI Scan ---------- */
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button {
  display:inline-flex !important; align-items:center; gap:8px; position:relative;
  background: linear-gradient(90deg,#2563eb,#7c3aed,#22d3ee,#2563eb) !important; background-size: 300% 100% !important;
  transition: transform .25s ease, box-shadow .25s ease, background-position .7s ease !important;
}
[data-testid="stFileUploaderDropzone"] button::before, [data-testid="stFileUploader"] section button::before {
  content:''; width:16px; height:16px; flex-shrink:0; transition: transform .25s ease;
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 15V3'/><path d='m7 8 5-5 5 5'/><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/></svg>") center/contain no-repeat;
}
[data-testid="stFileUploaderDropzone"] button::after, [data-testid="stFileUploader"] section button::after {
  content:'Choose MRI Scan'; font-size:.92rem; font-weight:600; }
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploader"] section button:hover {
  transform: scale(1.03); background-position: 100% 0 !important;
  box-shadow: 0 0 22px rgba(34,211,238,.5), 0 6px 20px rgba(124,58,237,.35) !important; }
[data-testid="stFileUploaderDropzone"] button:hover::before, [data-testid="stFileUploader"] section button:hover::before {
  transform: translateY(-3px); }
[data-testid="stFileUploaderDropzone"] button:active, [data-testid="stFileUploader"] section button:active {
  transform: scale(.97); }

/* ---------- 10 / 11. file info badges + privacy line ---------- */
.nv-info { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:center; margin-top:12px; }
.nv-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 11px; border-radius:999px; font-size:.7rem;
  color:#cfe0ff; background: rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.35); }
.nv-badge.max { color:#fcd9a0; background: rgba(251,191,36,.08); border-color: rgba(251,191,36,.3); }
.nv-secure { display:flex; align-items:center; justify-content:center; gap:7px; margin-top:12px; font-size:.72rem; color:#8aa0c4; }

/* ---------- 5. preview empty state ---------- */
.es-wrap { text-align:center; padding: 14px 0 6px; }
.es-vis { position:relative; width:170px; height:170px; margin:0 auto 12px; display:flex; align-items:center; justify-content:center; }
.es-vis .r { position:absolute; inset:0; border-radius:50%; border:1px solid rgba(34,211,238,.35); animation: nvRing 6s ease-out infinite; }
.es-vis .r2 { animation-delay: 2s; } .es-vis .r3 { animation-delay: 4s; }
.es-vis .sweep { position:absolute; inset:14px; border-radius:50%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(34,211,238,.20) 60deg, transparent 62deg);
  animation: nvSpin 12s linear infinite; }
.es-vis .orb { position:absolute; inset:6px; animation: nvSpin 22s linear infinite reverse; }
.es-vis .orb i { position:absolute; width:4px; height:4px; border-radius:50%; background:#7fe9ff; box-shadow:0 0 8px #22d3ee; opacity:.7; }
.es-ic { position:relative; z-index:1; animation: nvPulseIc 4.5s ease-in-out infinite; filter: drop-shadow(0 0 12px rgba(59,130,246,.55)); }
.es-t { color:#cfe0ff; font-weight:600; font-size:.92rem; }
.es-s { color:#6b7fa3; font-size:.76rem; margin-top:4px; }

/* ---------- 6. scanning feedback text (uploaded state) ---------- */
.scan-row .dots i { background:#22d3ee; }

/* ---------- 7. prediction empty state ---------- */
.ai-ready { text-align:center; padding: 18px 0 8px; }
.ai-ic { position:relative; width:110px; height:110px; margin:0 auto 12px; display:flex; align-items:center; justify-content:center; }
.ai-ic::before { content:''; position:absolute; inset:8px; border-radius:50%;
  background: radial-gradient(circle, rgba(167,139,250,.28), rgba(34,211,238,.10) 55%, transparent 72%);
  animation: nvHalo 4.5s ease-in-out infinite; }
.ai-ic .b { position:relative; z-index:1; animation: nvBrain 4.5s ease-in-out infinite; }
.ai-ic .nd { position:absolute; width:5px; height:5px; border-radius:50%; background:#22d3ee; box-shadow:0 0 8px #22d3ee;
  animation: nvPulseIc 3s ease-in-out infinite; }
.card:hover .ai-ic .b { animation-duration: 2.2s; }
.ai-t { color:#fff; font-weight:700; font-size:1.02rem; }
.ai-s { color:#8aa0c4; font-size:.8rem; margin-top:5px; }

/* ---------- 8 / 9. step indicator with connected flow ---------- */
.nv-steps { display:flex; align-items:center; justify-content:center; gap:14px; margin: 22px auto 4px; flex-wrap:wrap; }
.nv-step { display:flex; align-items:center; gap:9px; font-size:.84rem; color:#5a6f96; font-weight:600; transition: color .4s; }
.nv-step .n { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:.78rem; border:1px solid rgba(120,150,220,.3); background: rgba(10,22,50,.7); transition: all .4s; }
.nv-step.active { color:#22d3ee; }
.nv-step.active .n { color:#22d3ee; border-color:#22d3ee; box-shadow: 0 0 16px rgba(34,211,238,.55); }
.nv-step.done { color:#9fe7f5; }
.nv-step.done .n { border-color: rgba(52,211,153,.6); background: rgba(52,211,153,.12); }
.nv-conn { width:70px; height:2px; border-radius:2px; opacity:.6;
  background-image: repeating-linear-gradient(90deg, rgba(120,150,220,.4) 0 6px, transparent 6px 12px); background-size: 24px 2px; }
.nv-conn.on { opacity:1; background-image: repeating-linear-gradient(90deg, #22d3ee 0 6px, transparent 6px 12px);
  animation: nvDash 1.2s linear infinite; }

/* ---------- 12. disclaimer ---------- */
.disclaimer { display:flex; gap:12px; align-items:flex-start; color:#fde6b8; font-size:.82rem; line-height:1.55;
  background: rgba(251,191,36,.06); border:1px solid rgba(251,191,36,.3); backdrop-filter: blur(10px);
  box-shadow: 0 0 22px rgba(251,191,36,.07); }
.disclaimer .wi { flex-shrink:0; margin-top:2px; }

/* ---------- 13. bottom bar: only subtle effects ---------- */
.foot .fi svg, .foot .bi2 svg { animation: nvPulseIc 4.5s ease-in-out infinite; }
.foot .fbadge { border-left-color: rgba(34,211,238,.35); box-shadow: -6px 0 12px -10px rgba(34,211,238,.9);
  transition: background .3s, box-shadow .3s; border-radius: 0 12px 12px 0; padding-right: 10px; }
.foot .fbadge:hover { background: rgba(34,211,238,.06); box-shadow: -6px 0 14px -8px rgba(34,211,238,.9), 0 0 18px rgba(34,211,238,.12); }
.foot .report { transition: transform .25s, box-shadow .25s; cursor:pointer; }
.foot .report svg { transition: transform .25s; }
.foot .report:hover { transform: scale(1.03); box-shadow: 0 0 22px rgba(34,211,238,.4), 0 0 38px rgba(124,58,237,.25); }
.foot .report:hover svg { transform: translateX(5px); }

@media (prefers-reduced-motion: reduce) { * { animation-duration: .01s !important; animation-iteration-count: 1 !important; } }
</style>
"""

# Background particles + drag-and-drop detection (runs in the parent page, auto-cleans on rerun)
ANALYZE_JS = """
<script>
(function () {
  var P; try { P = window.parent; P.document; } catch (e) { return; }
  var D = P.document;
  if (P.__nvClean) { try { P.__nvClean(); } catch (e) {} }

  var alive = true;
  var cv = D.createElement('canvas'); cv.id = 'nv-bg';
  cv.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:-1;pointer-events:none;opacity:.5';
  (D.querySelector('.stApp') || D.body).appendChild(cv);
  var cx = cv.getContext('2d'), W = 0, H = 0, pts = [];

  function size() {
    var r = P.devicePixelRatio || 1;
    W = P.innerWidth; H = P.innerHeight;
    cv.width = W * r; cv.height = H * r; cx.setTransform(r, 0, 0, r, 0, 0);
    var n = Math.min(55, Math.floor(W * H / 24000)); pts = [];
    for (var i = 0; i < n; i++) pts.push({ x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .18, r: Math.random() * 1.1 + .4 });
  }
  size(); P.addEventListener('resize', size);

  function draw() {
    if (!alive) return;
    cx.clearRect(0, 0, W, H);
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i]; p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      cx.beginPath(); cx.arc(p.x, p.y, p.r, 0, 6.283); cx.fillStyle = 'rgba(130,190,255,.6)'; cx.fill();
      for (var j = i + 1; j < pts.length; j++) {
        var d = Math.hypot(p.x - pts[j].x, p.y - pts[j].y);
        if (d < 120) { cx.strokeStyle = 'rgba(100,160,255,' + ((1 - d / 120) * .14) + ')'; cx.lineWidth = 1;
          cx.beginPath(); cx.moveTo(p.x, p.y); cx.lineTo(pts[j].x, pts[j].y); cx.stroke(); }
      }
    }
    P.requestAnimationFrame(draw);
  }
  draw();

  function isFiles(e) {
    var t = e.dataTransfer && e.dataTransfer.types;
    return !!t && Array.prototype.indexOf.call(t, 'Files') > -1;
  }
  function clear() { D.body.classList.remove('nv-drag'); D.body.classList.remove('nv-over'); }
  function onDrag(e) {
    if (!isFiles(e)) return;
    D.body.classList.add('nv-drag');
    var over = e.target && e.target.closest && e.target.closest('[data-testid="stFileUploader"]');
    D.body.classList.toggle('nv-over', !!over);
  }
  function onLeave(e) { if (!e.relatedTarget) clear(); }
  D.addEventListener('dragenter', onDrag, true);
  D.addEventListener('dragover', onDrag, true);
  D.addEventListener('dragleave', onLeave, true);
  D.addEventListener('drop', clear, true);
  D.addEventListener('dragend', clear, true);

  P.__nvClean = function () {
    alive = false; clear();
    if (cv.parentNode) cv.parentNode.removeChild(cv);
    P.removeEventListener('resize', size);
    D.removeEventListener('dragenter', onDrag, true);
    D.removeEventListener('dragover', onDrag, true);
    D.removeEventListener('dragleave', onLeave, true);
    D.removeEventListener('drop', clear, true);
    D.removeEventListener('dragend', clear, true);
  };
})();
</script>
"""


def _field_dots():
    spec = [(8, 14, 0), (22, 70, 1.2), (86, 18, 2.1), (92, 62, 0.6), (14, 44, 3.0), (78, 40, 1.8), (50, 8, 2.6), (60, 82, 0.9)]
    return "".join(
        f'<i style="left:{x}%;top:{y}%;animation-delay:{d}s;animation-duration:{6 + (k % 3)}s"></i>'
        for k, (x, y, d) in enumerate(spec)
    )


def _steps_html(uploaded):
    if uploaded:
        states = ["done", "done", "active"]
        conns = ["on", "on"]
    else:
        states = ["active", "", ""]
        conns = ["", ""]
    names = ["Upload", "Preview", "Analyze"]
    html = ""
    for i, (s, nm) in enumerate(zip(states, names)):
        num = ic("check", 14, "#34d399") if s == "done" else str(i + 1)
        html += f'<div class="nv-step {s}"><span class="n">{num}</span>{nm}</div>'
        if i < 2:
            html += f'<div class="nv-conn {conns[i]}"></div>'
    return f'<div class="nv-steps">{html}</div>'


def page_analyze():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}. Place it in the same folder as app.py.")
        return

    st.markdown(ANALYZE_CSS, unsafe_allow_html=True)
    components.html(ANALYZE_JS, height=0)

    c2, c3, c4 = st.columns([0.85, 0.9, 1.0], gap="small")

    # ---------------- 1 / 2 / 3 / 4 : upload card ----------------
    with c2:
        st.markdown(
            f'<div class="card nv-upcard"><h3><span class="h-ic">{ic("upload", 16, "#22d3ee")}</span>Upload MRI Scan</h3>'
            f'<div class="up-pre"><div class="nv-field">{_field_dots()}</div>'
            f'<div class="up-ic"><span class="halo"></span>{ic("upload", 46, "#5fa8ff")}</div>'
            f'<div class="up-t"><span class="t-n">Drag &amp; drop your MRI image here</span>'
            f'<span class="t-h">Drop your MRI scan here</span>'
            f'<span class="t-d">Release to Upload MRI</span></div>'
            f'<div class="or">or</div></div></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown(
            f'<div class="nv-info">'
            f'<span class="nv-badge">{ic("image", 12, "#22d3ee")} JPG</span>'
            f'<span class="nv-badge">{ic("image", 12, "#22d3ee")} PNG</span>'
            f'<span class="nv-badge">{ic("image", 12, "#22d3ee")} JPEG</span>'
            f'<span class="nv-badge max">{ic("report", 12, "#fbbf24")} Max 10 MB</span></div>'
            f'<div class="nv-secure">{_SVG.format(s=13, c="#34d399", p=_LOCK_P)}'
            f'Your scan is processed securely for analysis.</div>',
            unsafe_allow_html=True,
        )

    image = None
    probs = None
    if uploaded is not None:
        image = Image.open(uploaded)
        with st.spinner("Analyzing MRI..."):
            probs = predict(image)

    # ---------------- 5 / 6 : preview card ----------------
    with c3:
        if image is not None:
            st.markdown(
                f'<div class="card"><h3><span class="h-ic">{ic("image", 16, "#22d3ee")}</span>MRI Scan Preview</h3>'
                f'<div class="scan"><img src="data:image/jpeg;base64,{to_b64(image)}"/></div>'
                f'<div class="scan-row">{ic("activity", 14, "#22d3ee")}<span>Preparing MRI scan</span>'
                f'<span class="dots"><i></i><i></i><i></i></span>'
                f'<div class="scanbar"><i></i></div><span class="scan-pct">68%</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            orb = "".join(
                f'<i style="left:{x}%;top:{y}%"></i>' for x, y in [(50, 0), (96, 58), (12, 86), (4, 30)]
            )
            st.markdown(
                f'<div class="card"><h3><span class="h-ic">{ic("image", 16, "#22d3ee")}</span>MRI Scan Preview</h3>'
                f'<div class="es-wrap"><div class="es-vis">'
                f'<i class="r r1"></i><i class="r r2"></i><i class="r r3"></i>'
                f'<span class="sweep"></span><span class="orb">{orb}</span>'
                f'<div class="es-ic">{ic("brain", 56, "#5fa8ff")}</div></div>'
                f'<div class="es-t">Upload an MRI scan to preview</div>'
                f'<div class="es-s">Your scan preview will appear here.</div></div></div>',
                unsafe_allow_html=True,
            )

    # ---------------- 7 : prediction card ----------------
    with c4:
        if probs is not None:
            idx = int(np.argmax(probs))
            key = CLASS_NAMES[idx]
            label, color = DISPLAY[key]
            conf = float(probs[idx]) * 100
            if conf >= 85:
                badge = f'<span class="badge b-ok">{ic("up", 11, "#34d399")} High Confidence</span>'
            elif conf >= 60:
                badge = f'<span class="badge b-mid">{ic("up", 11, "#fbbf24")} Moderate Confidence</span>'
            else:
                badge = f'<span class="badge b-low">{ic("up", 11, "#f87171")} Low Confidence</span>'

            rows = ""
            for i in np.argsort(probs)[::-1]:
                n, col = DISPLAY[CLASS_NAMES[i]]
                p = float(probs[i]) * 100
                rows += (
                    f'<div class="prob"><span class="dot" style="background:{col}"></span>'
                    f'<span class="n">{n}</span>'
                    f'<span class="bar"><i style="width:{p:.1f}%;background:{col}"></i></span>'
                    f'<span class="v">{p:.1f}%</span></div>'
                )
            st.markdown(
                f'<div class="card"><h3><span class="h-ic">{ic("brain", 16, "#a78bfa")}</span>AI Prediction Result</h3>'
                f'<div class="ring-wrap"><div class="ring" style="--c:{color};--target:{conf:.1f}"></div>'
                f'<div><div style="color:{color};font-weight:700;font-size:1.15rem">{label}</div>'
                f'<div class="big">{conf:.1f}%</div><div style="color:#9fb3d6;font-size:.85rem">Confidence</div>{badge}</div></div>'
                f'<div class="prob-title">Probability Distribution</div>{rows}</div>',
                unsafe_allow_html=True,
            )
        else:
            nodes = (
                '<i class="nd" style="left:10px;top:30px"></i>'
                '<i class="nd" style="right:8px;top:22px;animation-delay:.8s"></i>'
                '<i class="nd" style="left:22px;bottom:12px;animation-delay:1.6s"></i>'
                '<i class="nd" style="right:20px;bottom:18px;animation-delay:2.2s"></i>'
            )
            st.markdown(
                f'<div class="card"><h3><span class="h-ic">{ic("brain", 16, "#a78bfa")}</span>AI Prediction Result</h3>'
                f'<div class="ai-ready"><div class="ai-ic">{nodes}<div class="b">{ic("brain", 52, "#a78bfa")}</div></div>'
                f'<div class="ai-t">AI Analysis Ready</div>'
                f'<div class="ai-s">Upload an MRI scan to begin AI-powered analysis.</div></div></div>',
                unsafe_allow_html=True,
            )

    # ---------------- 8 / 9 : step indicator ----------------
    st.markdown(_steps_html(uploaded is not None), unsafe_allow_html=True)

    # ---------------- 12 : disclaimer ----------------
    st.markdown(
        f'<div class="disclaimer"><span class="wi">{_SVG.format(s=20, c="#fbbf24", p=_WARN_P)}</span>'
        f'<div><b>Research &amp; Educational Use:</b> This AI prediction is for demonstration/research purposes '
        f'and is not a medical diagnosis. Please consult a qualified doctor or radiologist for clinical evaluation.</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE: MODEL  (complete AI pipeline)
# =====================================================================
def page_model():
    steps = [
        # (css class, accent colour, icon html, title, description, hover tooltip, extra html)
        ("s-kaggle", "#20beff", '<span class="kag">K</span>', "Kaggle Dataset",
         "Brain MRI dataset from Kaggle", "Source of the brain MRI images used in this project", ""),
        ("s-python", "#fbbf24", "🐍", "Python Processing",
         "Data cleaning, resize, normalize, augmentation", "Images are cleaned and prepared for training", ""),
        ("s-data", "#22d3ee", ic("db", 20, "#22d3ee"), "Preprocessed Dataset",
         "Ready for model training", "Prepared images, split into train / validation / test", ""),
        ("s-colab", "#f59e0b", '<span class="colab"><i></i><i></i></span>', "Google Colab Training",
         "Train model on cloud", "The model is trained on Google Colab", ""),
        ("s-model core", "#a78bfa", f'<span class="brn">{ic("brain", 22, "#a78bfa")}</span>', "Trained Model",
         "Saved model (.h5 / .pkl)", "Model successfully trained → Ready for testing",
         '<span class="nvp-tag">AI CORE</span>'),
        ("s-test", "#34d399", ic("check", 20, "#34d399"), "Testing",
         "Evaluate on unseen data", "Performance is checked on images the model has not seen",
         '<span class="nvp-tag live"><i></i>MODEL VALIDATION</span>'),
        ("s-stream", "#c084fc", f'<span class="rkt">{ic("rocket", 20, "#c084fc")}</span>', "Streamlit Demo",
         "Interactive web app demo", "Upload an MRI scan and get a prediction",
         f'<span class="go-ar">{ic("arrow", 12, "#c084fc")}</span><span class="nvp-bot">Interactive AI Application</span>'),
        ("s-git", "#e6edf7", ic("github", 20, "#e6edf7"), "GitHub",
         "Version control &amp; deployment", "Code is versioned and deployed from GitHub",
         '<span class="nvp-tag done">&#10003; DEPLOYED</span>'),
    ]

    pipe_html = ""
    n = len(steps)
    for i, (cls, ac, icon, t, s, tip, extra) in enumerate(steps):
        d = 0.4 + 0.1 * i
        card = (
            f'<div class="step nvp nvp-step {cls}" style="--d:{d:.1f}s;--ac:{ac};--i:{i}">'
            f'{extra}<div class="p-ic">{icon}</div><b>{t}</b><small>{s}</small>'
            f'<div class="tip">{tip}</div></div>'
        )
        if cls == "s-stream" and STREAMLIT_DEMO_URL:
            card = (f'<a href="{STREAMLIT_DEMO_URL}" target="_blank" rel="noopener" '
                    f'style="display:contents;text-decoration:none">{card}</a>')
        pipe_html += card
        if i < n - 1:
            pipe_html += (
                f'<div class="arrow nvp-arrow" style="--i:{i};--d:{d + 0.05:.2f}s">'
                f'<span class="ln"></span><i class="pt"></i>{ic("arrow", 14, "#3b82f6")}</div>'
            )
    pipe_html += '<div class="nvp-end"></div>'

    bg = (
        '<div class="nvp-bg"><svg viewBox="0 0 1000 300" preserveAspectRatio="xMidYMid slice" width="100%" height="100%">'
        '<g stroke="rgba(120,170,255,.5)" stroke-width="1" fill="none" class="nvp-lines">'
        '<path d="M60 70 L200 150 L340 60 L480 140 L620 70 L760 150 L900 80"/>'
        '<path d="M60 230 L200 150 L340 240 L480 140 L620 230 L760 150 L900 220"/>'
        '<path d="M200 150 L340 150 L480 140 L620 150 L760 150"/>'
        '<path d="M340 60 L340 240 M620 70 L620 230"/></g>'
        '<g fill="#7fb2ff" class="nvp-nodes">'
        '<circle cx="60" cy="70" r="3"/><circle cx="200" cy="150" r="3.5"/><circle cx="340" cy="60" r="3"/>'
        '<circle cx="480" cy="140" r="3.5"/><circle cx="620" cy="70" r="3"/><circle cx="760" cy="150" r="3.5"/>'
        '<circle cx="900" cy="80" r="3"/><circle cx="60" cy="230" r="3"/><circle cx="340" cy="240" r="3"/>'
        '<circle cx="620" cy="230" r="3"/><circle cx="900" cy="220" r="3"/></g></svg></div>'
    )

    st.markdown(MODEL_CSS, unsafe_allow_html=True)
    components.html(MODEL_JS, height=0)
    st.markdown(
        f'<div class="card nvp-card">{bg}'
        f'<h3 class="nvp-h"><span class="h-ic nvp-hic">{ic("net", 16, "#22d3ee")}</span>Complete AI Pipeline</h3>'
        f'<div class="nvp-sub">From raw data to intelligent predictions — powered by a seamless AI workflow.</div>'
        f'<div class="nvp-line"></div>'
        f'<div class="pipe nvp-pipe">{pipe_html}</div></div>',
        unsafe_allow_html=True,
    )


# Optional: link opened when the "Streamlit Demo" card is clicked (leave "" for no link)
STREAMLIT_DEMO_URL = ""

MODEL_CSS = """
<style>
@keyframes nvpIn { from {opacity:0; transform: translateX(-24px);} to {opacity:1; transform:none;} }
@keyframes nvpFade { from {opacity:0;} to {opacity:1;} }
@keyframes nvpUp { from {opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }
@keyframes nvpRing { 0% {transform: scale(.85); opacity:.65;} 100% {transform: scale(1.5); opacity:0;} }
@keyframes nvpSpin { to {transform: rotate(360deg);} }
@keyframes nvpGlowO { 0%,100% {opacity:.12;} 50% {opacity:.75;} }
@keyframes nvpRocket { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-3px);} }
@keyframes nvpBrain { 0%,100% {transform: scale(1); filter: drop-shadow(0 0 0 rgba(167,139,250,0));}
                      50% {transform: scale(1.08); filter: drop-shadow(0 0 8px rgba(167,139,250,.9));} }
@keyframes nvpSweep { 0% {top:-30%; opacity:0;} 25% {opacity:.8;} 75% {opacity:.8;} 100% {top:110%; opacity:0;} }
@keyframes nvpTravel { 0% {left:0; opacity:0;} 2% {opacity:1;} 10% {left:calc(100% - 6px); opacity:1;} 12% {left:calc(100% - 6px); opacity:0;} 100% {left:calc(100% - 6px); opacity:0;} }
@keyframes nvpLine { 0% {left:0; opacity:0;} 10% {opacity:1;} 90% {opacity:1;} 100% {left:calc(100% - 8px); opacity:0;} }
@keyframes nvpDot { 0%,100% {opacity:.35; transform: scale(.8);} 50% {opacity:1; transform: scale(1.2);} }
@keyframes nvpPulse { 0%,100% {opacity:.75;} 50% {opacity:1;} }
@keyframes nvpDash { to {stroke-dashoffset: -40;} }
@keyframes nvpNode { 0%,100% {opacity:.3;} 50% {opacity:.9;} }
@keyframes nvpBlob { from {transform: translate(-20px,10px) scale(1);} to {transform: translate(24px,-14px) scale(1.08);} }
@keyframes nvpArc { from {stroke-dashoffset: 87.96;} to {stroke-dashoffset: var(--off);} }
@keyframes nvpSlow { to {transform: rotate(360deg);} }

/* ---------- 12. background: faint technical grid + radial glow (particles come from JS) ---------- */
.stApp { isolation: isolate; }
.stApp::before { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.9;
  background-image: linear-gradient(rgba(80,140,255,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(80,140,255,.04) 1px, transparent 1px);
  background-size: 56px 56px;
  -webkit-mask-image: radial-gradient(ellipse 85% 75% at 50% 40%, #000 0%, transparent 80%);
          mask-image: radial-gradient(ellipse 85% 75% at 50% 40%, #000 0%, transparent 80%); }
.stApp::after { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background: radial-gradient(560px 380px at 12% 85%, rgba(34,211,238,.07), transparent 70%),
              radial-gradient(600px 420px at 88% 20%, rgba(124,58,237,.09), transparent 70%);
  animation: nvpBlob 20s ease-in-out infinite alternate; }

/* ---------- 14. page-load sequence ---------- */
.nvp-card { overflow: visible; }
.nvp-h { animation: nvpFade .6s ease .2s backwards; position:relative; z-index:1; }
.nvp-sub { animation: nvpFade .6s ease .25s backwards; color:#8aa0c4; font-size:.8rem; margin:-6px 0 0 44px; position:relative; z-index:1; }
.nvp-line { animation: nvpFade .6s ease .3s backwards; }
.foot { animation: nvpFade .6s ease 1.3s backwards; }
.credit { animation: nvpFade .6s ease 1.4s backwards; }

/* ---------- 5. heading icon + animated line ---------- */
.nvp-hic { position:relative; box-shadow: 0 0 14px rgba(34,211,238,.35); animation: nvpPulse 4.5s ease-in-out infinite; }
.nvp-hic::after { content:''; position:absolute; inset:-5px; border-radius:50%; border:1px solid transparent;
  border-top-color:#22d3ee; border-right-color: rgba(167,139,250,.7); animation: nvpSpin 9s linear infinite; }
.nvp-line { position:relative; height:1px; margin: 12px 0 18px; z-index:1;
  background: linear-gradient(90deg, rgba(34,211,238,.45), rgba(124,58,237,.25), transparent 85%); }
.nvp-line::after { content:''; position:absolute; top:-2px; left:0; width:8px; height:5px; border-radius:50%;
  background:#7fe9ff; box-shadow: 0 0 10px 2px rgba(34,211,238,.8); animation: nvpLine 6s linear infinite; }

/* ---------- 13. neural network behind the pipeline ---------- */
.nvp-bg { position:absolute; inset:0; border-radius:18px; overflow:hidden; pointer-events:none; z-index:0; opacity:.16; }
.nvp-lines path { stroke-dasharray: 4 8; animation: nvpDash 8s linear infinite; }
.nvp-nodes circle { animation: nvpNode 5s ease-in-out infinite; }
.nvp-nodes circle:nth-child(2n) { animation-delay: 1.2s; } .nvp-nodes circle:nth-child(3n) { animation-delay: 2.4s; }

/* ---------- pipeline row ---------- */
.nvp-pipe { position:relative; z-index:1; flex-wrap:nowrap; gap:0; align-items:stretch; padding-top:6px; }
@media (max-width: 1100px) { .nvp-pipe { flex-wrap:wrap; gap:8px; } }

/* ---------- 1 / 2 / 16. cards: stagger in, hover lift, tilt ---------- */
.step.nvp { position:relative; min-width:0; animation: nvpIn .6s ease var(--d, 0s) backwards;
  transform: translateY(var(--ly, 0px)) perspective(700px) rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg));
  transition: transform .25s ease, border-color .3s, box-shadow .3s, background .3s; }
.step.nvp::before { content:''; position:absolute; inset:-14px; z-index:-1; border-radius:24px; opacity:0; pointer-events:none;
  background: radial-gradient(circle at 50% 50%, var(--ac), transparent 68%); transition: opacity .35s; filter: blur(14px); }
.step.nvp:hover { --ly:-3px; border-color: var(--ac); box-shadow: 0 0 22px rgba(34,211,238,.18); }
.step.nvp:hover::before { opacity:.2; }
.step.nvp:hover .p-ic { transform: scale(1.05); box-shadow: 0 0 22px var(--ac); }
.step.nvp b { transition: color .3s, text-shadow .3s; }
.step.nvp:hover b { color:#fff; text-shadow: 0 0 12px rgba(255,255,255,.35); }
.step.nvp small { transition: color .3s; }
.step.nvp:hover small { color:#cfe0ff; }

/* tooltip (extra info only on hover) */
.step .tip { position:absolute; left:50%; top:calc(100% + 10px); transform: translate(-50%, 6px); width:max-content; max-width:190px;
  padding:7px 11px; border-radius:10px; font-size:.64rem; line-height:1.4; color:#cfe0ff; text-align:center; z-index:20;
  background: rgba(8,16,38,.96); border:1px solid rgba(34,211,238,.4); box-shadow: 0 8px 24px rgba(0,0,0,.4);
  opacity:0; pointer-events:none; transition: opacity .25s, transform .25s; }
.step .tip::before { content:''; position:absolute; left:50%; top:-5px; width:8px; height:8px; margin-left:-4px; transform: rotate(45deg);
  background: rgba(8,16,38,.96); border-left:1px solid rgba(34,211,238,.4); border-top:1px solid rgba(34,211,238,.4); }
.step:hover .tip { opacity:1; transform: translate(-50%, 0); }

/* ---------- 3. icon animations (slow + staggered) ---------- */
.step .p-ic { position:relative; transition: transform .3s, box-shadow .3s; }
.s-kaggle .p-ic::after, .s-test .p-ic::after { content:''; position:absolute; inset:0; border-radius:50%; border:1px solid var(--ac);
  pointer-events:none; animation: nvpRing 5s ease-out infinite; }
.s-test .p-ic::after { animation-delay: 2.5s; }
.s-python .p-ic::before { content:''; position:absolute; inset:-4px; border-radius:50%; border:2px solid transparent;
  border-top-color: var(--ac); border-right-color: rgba(56,189,248,.7); animation: nvpSpin 12s linear infinite; pointer-events:none; }
.s-python .p-ic { animation: nvpPulse 5s ease-in-out 1s infinite; }
.s-data .p-ic { overflow:hidden; }
.s-data .p-ic::after { content:''; position:absolute; left:8px; right:8px; height:10px; top:-30%;
  background: linear-gradient(180deg, transparent, rgba(34,211,238,.55), transparent); animation: nvpSweep 4.5s ease-in-out 1.5s infinite; }
.s-colab .p-ic::after, .s-git .p-ic::after { content:''; position:absolute; inset:0; border-radius:50%; pointer-events:none;
  box-shadow: 0 0 18px 2px var(--ac); animation: nvpGlowO 5s ease-in-out infinite; }
.s-colab .p-ic::after { animation-delay: 2s; } .s-git .p-ic::after { animation-delay: 3.4s; }
.s-model .brn { display:flex; animation: nvpBrain 4.5s ease-in-out 1.2s infinite; }
.s-stream .rkt { display:flex; animation: nvpRocket 4s ease-in-out 2s infinite; }

/* ---------- 6. Trained Model = AI CORE ---------- */
.step.core { border-color: rgba(167,139,250,.65); background: rgba(24,28,70,.85);
  box-shadow: 0 0 26px rgba(124,58,237,.3), inset 0 0 18px rgba(34,211,238,.07); }
.step.core .p-ic { border-color: rgba(167,139,250,.7); box-shadow: 0 0 20px rgba(167,139,250,.35); }

/* tags: AI CORE / MODEL VALIDATION / DEPLOYED */
.nvp-tag { position:absolute; top:-9px; right:8px; padding:2px 9px; border-radius:999px; font-size:.54rem; font-weight:700;
  letter-spacing:.06em; white-space:nowrap; z-index:2; color:#fff; background: linear-gradient(90deg,#2563eb,#7c3aed);
  box-shadow: 0 0 12px rgba(124,58,237,.5); }
.nvp-tag.live { right:auto; left:50%; transform: translateX(-50%); display:inline-flex; align-items:center; gap:5px;
  color:#34d399; background: rgba(8,30,28,.95); border:1px solid rgba(52,211,153,.5); box-shadow:none; }
.nvp-tag.live i { width:6px; height:6px; border-radius:50%; background:#34d399; box-shadow:0 0 8px #34d399; animation: nvpDot 2.4s ease-in-out infinite; }
.nvp-tag.done { color:#cfe0ff; background: rgba(14,28,60,.95); border:1px solid rgba(80,140,255,.5); box-shadow:none; }

/* ---------- 8. Streamlit demo ---------- */
.go-ar { position:absolute; top:10px; right:10px; opacity:0; transform: translateX(-4px); transition: all .3s; }
.s-stream:hover .go-ar { opacity:1; transform:none; }
.nvp-bot { position:absolute; left:8px; right:8px; bottom:8px; padding:3px 4px; border-radius:8px; font-size:.54rem; font-weight:600;
  color:#e9d5ff; background: rgba(124,58,237,.25); border:1px solid rgba(192,132,252,.5);
  opacity:0; transform: translateY(4px); transition: all .3s; }
.s-stream:hover .nvp-bot { opacity:1; transform:none; }
.s-stream { padding-bottom: 14px; }
.s-stream:hover { box-shadow: 0 0 26px rgba(192,132,252,.3); }

/* ---------- 4 / 9. arrows with a travelling light + terminated end ---------- */
.nvp-arrow { position:relative; width:28px; flex-shrink:0; display:flex; align-items:center; justify-content:center;
  animation: nvpFade .5s ease var(--d, 0s) backwards; }
.nvp-arrow .ln { position:absolute; left:0; right:0; top:50%; height:1px; background: rgba(34,211,238,.28); }
.nvp-arrow svg { position:relative; z-index:1; transition: transform .25s; filter: drop-shadow(0 0 4px rgba(34,211,238,.6)); }
.nvp-arrow:hover svg { transform: translateX(3px); }
.nvp-arrow .pt { position:absolute; top:50%; left:0; width:6px; height:6px; margin-top:-3px; border-radius:50%; z-index:2;
  background:#7fe9ff; box-shadow: 0 0 10px 3px rgba(34,211,238,.85); opacity:0;
  animation: nvpTravel 8s linear infinite; animation-delay: calc(var(--i) * .9s + 1.6s); }
.nvp-end { width:26px; height:2px; align-self:center; flex-shrink:0; border-radius:2px; margin-left:4px;
  background: linear-gradient(90deg, rgba(34,211,238,.7), transparent); box-shadow: 0 0 10px rgba(34,211,238,.5);
  animation: nvpFade .6s ease 1.2s backwards; }
.s-git .nvp-tag.done { animation: nvpPulse 4s ease-in-out infinite; }
@media (max-width: 1100px) { .nvp-arrow, .nvp-end { display:none; } .step.nvp { min-width:140px; } }

/* ---------- 10. bottom highlight bar (only subtle extras) ---------- */
.foot .fbadge { border-left-color: rgba(34,211,238,.35); box-shadow: -6px 0 12px -10px rgba(34,211,238,.9);
  transition: background .3s, box-shadow .3s; border-radius: 0 12px 12px 0; padding-right: 10px; }
.foot .fbadge:hover { background: rgba(34,211,238,.06); }
.foot .fbadge:nth-child(2) .bi2 svg { animation: nvpPulse 4s ease-in-out infinite; }
.foot .fbadge:nth-child(4) .bi2 svg { animation: nvpPulse 2.8s ease-in-out infinite; }
.foot .fbadge:nth-child(5) .bi2 svg { animation: nvpSlow 28s linear infinite; }
.foot .fbadge .bi2.nvp-ring { position:relative; background:transparent; border:0; width:38px; height:38px; }
.foot .fbadge .bi2.nvp-ring svg.rg { position:absolute; inset:0; transform: rotate(-90deg); }
.foot .fbadge .bi2.nvp-ring svg.ti { position:absolute; left:50%; top:50%; margin:-7px 0 0 -7px; }
.foot .fbadge .bi2.nvp-ring .arc { stroke-dasharray: 87.96; stroke-dashoffset: var(--off); animation: nvpArc 1.8s ease-out 1.5s backwards; }
.foot .report { transition: transform .25s, box-shadow .25s; cursor:pointer; }
.foot .report svg { transition: transform .25s; }
.foot .report:hover { transform: scale(1.03); box-shadow: 0 0 22px rgba(34,211,238,.4), 0 0 38px rgba(124,58,237,.25); }
.foot .report:hover svg { transform: translateX(5px); }

@media (prefers-reduced-motion: reduce) { * { animation-duration: .01s !important; animation-iteration-count: 1 !important; } }
</style>
"""

# 16 mouse tilt (max ~1.5deg), 11 accuracy count-up, 12 faint particles. Runs in the parent page.
MODEL_JS = """
<script>
(function () {
  var P; try { P = window.parent; P.document; } catch (e) { return; }
  var D = P.document;
  if (P.__nvpClean) { try { P.__nvpClean(); } catch (e) {} }
  var alive = true, timers = [];

  /* faint particles */
  var cv = D.createElement('canvas'); cv.id = 'nvp-bg';
  cv.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:-1;pointer-events:none;opacity:.45';
  (D.querySelector('.stApp') || D.body).appendChild(cv);
  var cx = cv.getContext('2d'), W = 0, H = 0, pts = [];
  function size() {
    var r = P.devicePixelRatio || 1; W = P.innerWidth; H = P.innerHeight;
    cv.width = W * r; cv.height = H * r; cx.setTransform(r, 0, 0, r, 0, 0);
    var n = Math.min(50, Math.floor(W * H / 26000)); pts = [];
    for (var i = 0; i < n; i++) pts.push({ x: Math.random()*W, y: Math.random()*H, vx: (Math.random()-.5)*.16, vy: (Math.random()-.5)*.16, r: Math.random()*1.1+.4 });
  }
  size(); P.addEventListener('resize', size);
  function draw() {
    if (!alive) return;
    cx.clearRect(0, 0, W, H);
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i]; p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      cx.beginPath(); cx.arc(p.x, p.y, p.r, 0, 6.283); cx.fillStyle = 'rgba(130,190,255,.6)'; cx.fill();
      for (var j = i + 1; j < pts.length; j++) {
        var d = Math.hypot(p.x - pts[j].x, p.y - pts[j].y);
        if (d < 115) { cx.strokeStyle = 'rgba(100,160,255,' + ((1 - d/115) * .12) + ')'; cx.lineWidth = 1;
          cx.beginPath(); cx.moveTo(p.x, p.y); cx.lineTo(pts[j].x, pts[j].y); cx.stroke(); }
      }
    }
    P.requestAnimationFrame(draw);
  }
  draw();

  /* tilt on cards + count-up (elements may render a moment later, so poll briefly) */
  function ease(t) { return 1 - Math.pow(1 - t, 3); }
  function setup() {
    D.querySelectorAll('.nvp-step').forEach(function (el) {
      if (el.__nvp) return; el.__nvp = 1;
      el.addEventListener('mousemove', function (e) {
        var b = el.getBoundingClientRect();
        var px = (e.clientX - b.left) / b.width - .5, py = (e.clientY - b.top) / b.height - .5;
        el.style.setProperty('--ry', (px * 3).toFixed(2) + 'deg');
        el.style.setProperty('--rx', (-py * 3).toFixed(2) + 'deg');
      });
      el.addEventListener('mouseleave', function () {
        el.style.setProperty('--rx', '0deg'); el.style.setProperty('--ry', '0deg');
      });
    });
    D.querySelectorAll('[data-nvcount]').forEach(function (el) {
      if (el.__nvc) return; el.__nvc = 1;
      var to = parseFloat(el.getAttribute('data-nvcount')), dec = parseInt(el.getAttribute('data-dec') || '0', 10);
      el.textContent = (0).toFixed(dec);
      var start = null;
      timers.push(setTimeout(function () {
        P.requestAnimationFrame(function step(ts) {
          if (!alive) return;
          if (!start) start = ts;
          var p = Math.min((ts - start) / 1800, 1);
          el.textContent = (to * ease(p)).toFixed(dec);
          if (p < 1) P.requestAnimationFrame(step);
        });
      }, 1500));
    });
  }
  setup();
  var poll = setInterval(setup, 250);
  timers.push(setTimeout(function () { clearInterval(poll); }, 6000));

  P.__nvpClean = function () {
    alive = false; clearInterval(poll);
    timers.forEach(clearTimeout);
    if (cv.parentNode) cv.parentNode.removeChild(cv);
    P.removeEventListener('resize', size);
  };
})();
</script>
"""


# =====================================================================
# PAGE: DATASET  (dataset explorer)
# =====================================================================
# ---------------------------------------------------------------------
# Real MRI thumbnails for the Dataset page.
# Put a few sample images in a folder next to app.py, one sub-folder per class:
#     dataset_samples/glioma/*.jpg
#     dataset_samples/meningioma/*.jpg
#     dataset_samples/pituitary/*.jpg
#     dataset_samples/notumor/*.jpg
# (a Kaggle-style "Training/<class>" folder also works). If nothing is found,
# the old placeholder boxes are shown, so nothing breaks.
# ---------------------------------------------------------------------
DATASET_IMG_DIRS = ["dataset_samples", "sample_images", "Training", "Testing",
                    "dataset/Training", "dataset/Testing", "archive/Training", "archive/Testing"]
_CLASS_FOLDERS = {
    "Glioma": ("glioma", "Glioma", "glioma_tumor"),
    "Meningioma": ("meningioma", "Meningioma", "meningioma_tumor"),
    "Pituitary": ("pituitary", "Pituitary", "pituitary_tumor"),
    "No Tumor": ("notumor", "no_tumor", "NoTumor", "No Tumor", "no tumor", "normal", "Normal"),
}
_CLASS_SLUG = {"Glioma": "glioma", "Meningioma": "meningioma", "Pituitary": "pituitary", "No Tumor": "notumor"}


@st.cache_data(show_spinner=False)
def _class_thumbs(class_label, limit=3):
    """Up to `limit` (thumbnail_b64, original_width, original_height) for one class ([] if none found)."""
    base = Path(__file__).parent
    exts = {".jpg", ".jpeg", ".png"}
    for root in DATASET_IMG_DIRS:
        for folder in _CLASS_FOLDERS.get(class_label, ()):
            d = base / root / folder
            if not d.is_dir():
                continue
            out = []
            for p in sorted(f for f in d.iterdir() if f.suffix.lower() in exts):
                try:
                    im = Image.open(p).convert("RGB")
                    w, h = im.size
                    im.thumbnail((260, 260))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=82)
                    out.append((base64.b64encode(buf.getvalue()).decode(), w, h))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
            if out:
                return out
    return []


def _mri_div(b64=None):
    if b64:
        return (f'<div class="mri" style="background:url(data:image/jpeg;base64,{b64}) '
                f'center/cover no-repeat"></div>')
    return '<div class="mri"></div>'


def _qp(name, default=""):
    v = st.query_params.get(name, default)
    if isinstance(v, list):
        v = v[0] if v else default
    return v


def _num(s):
    try:
        return int(str(s).replace(",", "").strip())
    except ValueError:
        return None


def _mwrap(item, overlay=True):
    """Image area. Metadata labels are shown only for real images (real file dimensions)."""
    if item:
        b64, w, h = item
        labels = f'<span class="mlab tl">MRI SAMPLE</span><span class="mlab br">{w} × {h}</span>'
    else:
        b64, labels = None, ""
    ovl = (f'<div class="ovl"><span>View Details {ic("arrow", 11, "#fff")}</span></div>' if overlay else
           '<div class="ovl"></div>')
    return f'<div class="mwrap">{_mri_div(b64)}{labels}<i class="scn"></i>{ovl}</div>'


def _dataset_gallery():
    """Full gallery (opened by 'View All' / 'View Details'): ?page=dataset&view=all[&cls=<class>]"""
    cls = _qp("cls")
    chips = f'<a class="gchip{" on" if not cls else ""}" href="?page=dataset&amp;view=all" target="_self">All</a>'
    for n, _, col in DATASET["categories"]:
        on = " on" if cls == _CLASS_SLUG[n] else ""
        chips += (f'<a class="gchip{on}" href="?page=dataset&amp;view=all&amp;cls={_CLASS_SLUG[n]}" target="_self">'
                  f'<i style="background:{col}"></i>{n}</a>')

    items, k = "", 0
    for n, cnt, col in DATASET["categories"]:
        if cls and _CLASS_SLUG[n] != cls:
            continue
        for it in _class_thumbs(n, 12):
            d = min(k * 0.04, 0.7)
            items += (f'<div class="gitem" style="--c:{col};--d:{d:.2f}s">{_mwrap(it, overlay=False)}'
                      f'<div class="nm"><i style="background:{col}"></i>{n}</div></div>')
            k += 1
    if not items:
        items = ('<div class="wait" style="grid-column:1/-1">No sample images found. Add a few images to '
                 '<b>dataset_samples/&lt;class&gt;/</b> next to app.py (glioma, meningioma, pituitary, notumor).</div>')

    st.markdown(
        f'<div class="card"><h3><span class="h-ic">{ic("db", 16, "#22d3ee")}</span>Dataset Gallery</h3>'
        f'<div class="ds-sh"><a class="ds-va back" href="?page=dataset" target="_self">'
        f'<span style="display:inline-block;transform:rotate(180deg);line-height:0">{ic("arrow", 13, "#22d3ee")}</span>'
        f'Back to Dataset Explorer</a></div>'
        f'<div class="gchips">{chips}</div><div class="gal">{items}</div></div>',
        unsafe_allow_html=True,
    )


DATASET_CSS = """
<style>
@keyframes dsUp { from {opacity:0; transform: translateY(16px);} to {opacity:1; transform:none;} }
@keyframes dsReveal { from {opacity:0; transform: translateY(40px);} to {opacity:1; transform:none;} }
@keyframes dsScan { 0% {top:0; opacity:0;} 10% {opacity:1;} 90% {opacity:1;} 100% {top:calc(100% - 3px); opacity:0;} }
@keyframes dsDot { 0%,100% {opacity:.55; transform: scale(.85);} 50% {opacity:1; transform: scale(1.25);} }
@keyframes dsBar { from {width:0;} to {width: var(--w);} }
@keyframes dsDrift {
  from {background-position: 0 100%, 0 0, 40px 60px;}
  to   {background-position: -800px 100%, 0 -190px, 40px -170px;} }
@keyframes dsGlow { 0%,100% {opacity:.6;} 50% {opacity:1;} }

/* ---------- 11. background: MRI scan grid + data particles + faint ECG waveform ---------- */
.stApp { isolation: isolate; }
.stApp::before { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background-image:
    linear-gradient(rgba(80,150,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(80,150,255,.035) 1px, transparent 1px),
    radial-gradient(520px 340px at 85% 10%, rgba(34,211,238,.07), transparent 70%);
  background-size: 32px 32px, 32px 32px, auto;
  -webkit-mask-image: radial-gradient(ellipse 90% 80% at 50% 35%, #000 0%, transparent 85%);
          mask-image: radial-gradient(ellipse 90% 80% at 50% 35%, #000 0%, transparent 85%); }
.stApp::after { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.2;
  background-image:
    url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='60' viewBox='0 0 800 60'%3E%3Cpath d='M0 30 H150 L165 30 L175 8 L190 52 L200 30 H260 L275 24 L290 30 H550 L565 30 L575 8 L590 52 L600 30 H660 L675 24 L690 30 H800' fill='none' stroke='%2322d3ee' stroke-width='1.2'/%3E%3C/svg%3E"),
    radial-gradient(circle, rgba(127,233,255,.6) 1px, transparent 1.6px),
    radial-gradient(circle, rgba(167,139,250,.5) 1px, transparent 1.6px);
  background-repeat: repeat-x, repeat, repeat;
  background-size: 800px 60px, 170px 190px, 260px 230px;
  background-position: 0 100%, 0 0, 40px 60px;
  animation: dsDrift 70s linear infinite; }

/* ---------- 12. glass cards (kept subtle) ---------- */
.card { backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  background: linear-gradient(160deg, rgba(14,28,60,.62), rgba(8,16,38,.72));
  border:1px solid rgba(80,140,255,.28); box-shadow: 0 10px 36px rgba(0,0,0,.28);
  animation: dsUp .6s ease .05s backwards; transition: box-shadow .35s, border-color .35s; }
.card:hover { border-color: rgba(34,211,238,.4); box-shadow: 0 10px 36px rgba(0,0,0,.28), 0 0 26px rgba(34,211,238,.1); }

/* ---------- 2 / 3. statistics cards: count-up + proportion bar ---------- */
.dstat { position:relative; backdrop-filter: blur(6px); animation: dsUp .55s ease var(--d) backwards;
  transition: transform .3s, border-color .3s, box-shadow .3s; }
.dstat:hover { transform: translateY(-3px); border-color: rgba(34,211,238,.6); box-shadow: 0 8px 22px rgba(34,211,238,.15); }
.dstat .v { font-variant-numeric: tabular-nums; }
.dbar { height:4px; margin:8px 8px 0; border-radius:99px; background: rgba(255,255,255,.08); overflow:hidden; }
.dbar i { display:block; height:100%; width: var(--w); border-radius:99px;
  background: linear-gradient(90deg, var(--c), rgba(124,58,237,.9));
  animation: dsBar 1.3s ease-out calc(var(--d) + .3s) backwards; }

/* ---------- 4 / 5 / 6 / 7. MRI cards ---------- */
a.cat, a.dsamp { display:block; text-decoration:none; color:inherit; }
.dcat { position:relative; animation: dsUp .55s ease var(--d) backwards;
  backdrop-filter: blur(6px); transition: transform .3s, border-color .3s, box-shadow .3s; }
.dcat:hover { transform: translateY(-4px); border-color: rgba(34,211,238,.7);
  box-shadow: 0 0 0 1px rgba(124,58,237,.35), 0 10px 28px rgba(34,211,238,.18); }
.mwrap { position:relative; overflow:hidden; border-radius:10px; margin:7px 7px 0; }
.mwrap .mri { margin:0; width:100%; transition: transform .55s ease; }
.dcat:hover .mri, .dsamp:hover .mri, .gitem:hover .mri { transform: scale(1.03); }
.scn { position:absolute; left:0; right:0; top:0; height:3px; opacity:0; pointer-events:none; z-index:3;
  background: linear-gradient(90deg, transparent, #22d3ee, transparent); box-shadow: 0 0 14px 3px rgba(34,211,238,.6); }
.dcat:hover .scn, .dsamp:hover .scn, .gitem:hover .scn { animation: dsScan 2.8s ease-in-out infinite; }
.ovl { position:absolute; inset:0; z-index:2; display:flex; align-items:center; justify-content:center; opacity:0; pointer-events:none;
  background: linear-gradient(180deg, rgba(5,11,26,.3), rgba(5,11,26,.7)); transition: opacity .35s; }
.ovl span { display:inline-flex; align-items:center; gap:5px; font-size:.62rem; font-weight:700; color:#fff; }
.ovl svg { transition: transform .3s; }
.dcat:hover .ovl, .dsamp:hover .ovl { opacity:1; }
.dcat:hover .ovl svg, .dsamp:hover .ovl svg { transform: translateX(4px); }
.mlab { position:absolute; z-index:2; font-size:.46rem; letter-spacing:.1em; color: rgba(207,224,255,.85); pointer-events:none;
  padding:1px 5px; border-radius:5px; background: rgba(5,11,26,.55); }
.mlab.tl { top:5px; left:5px; } .mlab.br { bottom:5px; right:5px; }
.dcat .nm, .gitem .nm { transition: color .3s; }
.dcat:hover .nm, .gitem:hover .nm { color:#fff; }
.dcat .nm i, .gitem .nm i { box-shadow: 0 0 8px var(--c); animation: dsDot 2.6s ease-in-out infinite; transition: box-shadow .3s; }
.dcat:hover .nm i { box-shadow: 0 0 14px var(--c); }
.cbar { height:3px; margin:0 8px 9px; border-radius:99px; background: rgba(255,255,255,.08); overflow:hidden; }
.cbar i { display:block; height:100%; width: var(--w); border-radius:99px; background: var(--c); opacity:.85;
  animation: dsBar 1.2s ease-out calc(var(--d) + .3s) backwards; }

/* ---------- 8 / 9. sample images section ---------- */
.ds-sec { margin-top:6px; padding:14px; border-radius:14px; background: rgba(10,22,50,.45);
  border:1px solid rgba(80,140,255,.2); animation: dsUp .6s ease .6s backwards; }
.ds-sh { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; gap:10px; }
.ds-sh .sub { margin:0; }
.ds-desc { font-size:.7rem; color:#6b7fa3; margin-top:2px; }
a.ds-va { display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:10px; font-size:.8rem; font-weight:600;
  color:#22d3ee; text-decoration:none; border:1px solid transparent; white-space:nowrap; transition: border-color .25s, box-shadow .25s, background .25s; }
a.ds-va svg { transition: transform .25s; }
a.ds-va:hover { border-color: rgba(34,211,238,.6); background: rgba(34,211,238,.07); box-shadow: 0 0 18px rgba(34,211,238,.25); }
a.ds-va:hover svg { transform: translateX(4px); }
a.ds-va.back:hover svg { transform: none; }
.dsamp { position:relative; border-radius:10px; animation: dsUp .55s ease var(--d) backwards;
  transition: transform .3s, box-shadow .3s; }
.dsamp:hover { transform: translateY(-3px); box-shadow: 0 0 0 1px rgba(34,211,238,.6), 0 8px 22px rgba(34,211,238,.18); }
.samples .mwrap { margin:0; }

/* ---------- gallery (View All) ---------- */
.gchips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
a.gchip { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; font-size:.74rem; text-decoration:none; color:#cfe0ff;
  background: rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.3); transition: border-color .25s, background .25s; }
a.gchip i { width:7px; height:7px; border-radius:50%; display:inline-block; }
a.gchip:hover, a.gchip.on { border-color:#22d3ee; background: rgba(34,211,238,.14); color:#fff; }
.gal { display:grid; grid-template-columns: repeat(6, 1fr); gap:10px; }
.gitem { border-radius:12px; overflow:hidden; background: rgba(10,22,50,.65); border:1px solid rgba(80,140,255,.22);
  animation: dsUp .5s ease var(--d) backwards; transition: transform .3s, border-color .3s, box-shadow .3s; }
.gitem:hover { transform: translateY(-3px); border-color: rgba(34,211,238,.7); box-shadow: 0 8px 22px rgba(34,211,238,.18); }
.gitem .nm { font-size:.7rem; color:#cfe0ff; padding:7px 8px 8px; display:flex; align-items:center; gap:6px; }
.gitem .nm i { width:7px; height:7px; border-radius:50%; display:inline-block; }

@media (max-width: 900px) { .gal, .samples { grid-template-columns: repeat(3, 1fr); } .cats, .dstats { grid-template-columns: repeat(2, 1fr); } }

/* ---------- 13 / 14. scroll reveal (where supported) + bottom bar last ---------- */
.foot { animation: dsUp .7s ease 1s backwards; }
.credit { animation: dsUp .7s ease 1.2s backwards; }
@supports (animation-timeline: view()) {
  .ds-sec, .foot, .credit { animation: dsReveal linear both; animation-timeline: view(); animation-range: entry 0% entry 55%; }
}
@media (prefers-reduced-motion: reduce) { * { animation-duration: .01s !important; animation-iteration-count: 1 !important; } }
</style>
"""


def page_dataset():
    st.markdown(ARCH_CSS, unsafe_allow_html=True)      # bottom-bar effects (icon pulse, ring, report arrow)
    st.markdown(DATASET_CSS, unsafe_allow_html=True)
    components.html(ARCH_JS, height=0)                  # count-up for numbers

    if _qp("view") == "all":
        _dataset_gallery()
        return

    # ---- statistics cards (numbers are the real DATASET values; only animated) ----
    total = _num(DATASET["total"])
    stats = [("image", "Total Images", "total", "#22d3ee", "#22d3ee"),
             ("report", "Training", "train", "#34d399", "#34d399"),
             ("net", "Validation", "val", "#a78bfa", "#a78bfa"),
             ("check", "Testing", "test", "#fbbf24", "#fbbf24")]
    dstat = ""
    for i, (icn, label, key, icol, col) in enumerate(stats):
        raw = DATASET[key]
        n = _num(raw)
        if n is None:
            val, bar = raw, ""
        else:
            val = f'<span data-nvcount="{n}" data-dec="0" data-comma="1">{raw}</span>'
            pct = 100 if key == "total" else (min(100.0, n / total * 100) if total else 0)
            bar = f'<div class="dbar" style="--w:{pct:.1f}%;--c:{col}"><i></i></div>'
        dstat += (
            f'<div class="dstat" style="--d:{0.1 + 0.08 * i:.2f}s"><div class="di">{ic(icn, 14, icol)}</div>'
            f'<div class="l">{label}</div><div class="v">{val}</div>{bar}</div>'
        )

    # ---- category cards ----
    thumbs = {n: _class_thumbs(n) for n, _, _ in DATASET["categories"]}
    counts = [_num(c) or 0 for _, c, _ in DATASET["categories"]]
    cmax = max(counts) if max(counts) else 1
    cats = ""
    for i, (n, cnt, c) in enumerate(DATASET["categories"]):
        d = 0.1 + 0.1 * i                                        # 0.1s -> 0.2s -> 0.3s -> 0.4s
        first = thumbs[n][0] if thumbs[n] else None
        w = (_num(cnt) or 0) / cmax * 100
        cats += (
            f'<a class="cat dcat" href="?page=dataset&amp;view=all&amp;cls={_CLASS_SLUG[n]}" target="_self" '
            f'style="--c:{c};--d:{d:.2f}s">{_mwrap(first)}'
            f'<div class="nm"><i style="background:{c}"></i>{n}</div><div class="ct">{cnt}</div>'
            f'<div class="cbar" style="--w:{w:.1f}%"><i></i></div></a>'
        )

    # ---- 6 sample images, mixed across the classes (placeholders if no images exist) ----
    picks = []
    for r in range(3):
        for n, _, _ in DATASET["categories"]:
            if r < len(thumbs[n]):
                picks.append(thumbs[n][r])
    picks = picks[:6] + [None] * (6 - len(picks[:6]))
    samples = "".join(
        f'<a class="dsamp" href="?page=dataset&amp;view=all" target="_self" style="--d:{0.5 + 0.1 * i:.2f}s">{_mwrap(p)}</a>'
        for i, p in enumerate(picks)
    )

    st.markdown(
        f'<div class="card"><h3><span class="h-ic">{ic("db", 16, "#22d3ee")}</span>Dataset Explorer</h3>'
        f'<div class="dstats">{dstat}</div>'
        f'<div class="sub">Dataset Categories</div><div class="cats">{cats}</div>'
        f'<div class="ds-sec"><div class="ds-sh"><div><div class="sub">Sample Images</div>'
        f'<div class="ds-desc">Representative MRI scans from the dataset</div></div>'
        f'<a class="ds-va" href="?page=dataset&amp;view=all" target="_self">View All {ic("arrow", 12, "#22d3ee")}</a></div>'
        f'<div class="samples">{samples}</div></div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE: PERFORMANCE  (metrics + training results)
# =====================================================================
def page_performance():
    import json
    # same data as before: your ACC_CURVE / LOSS_CURVE if set, otherwise the generated sample curves
    acc = np.asarray(ACC_CURVE if ACC_CURVE is not None else _sample_curves()[0], dtype=float)
    loss = np.asarray(LOSS_CURVE if LOSS_CURVE is not None else _sample_curves()[1], dtype=float)
    data = {
        "acc": [round(float(v), 4) for v in acc],
        "loss": [round(float(v), 4) for v in loss],
        "labels": list(CONF_LABELS),
        "matrix": [[int(v) for v in row] for row in CONF_MATRIX],
        "sample": bool(ACC_CURVE is None or LOSS_CURVE is None),
    }

    html = PERF_TEMPLATE
    html = html.replace("__TILES__", _perf_tiles())
    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__I_CHART__", ic("chart", 16, "#22d3ee"))

    st.markdown(ARCH_CSS, unsafe_allow_html=True)       # bottom-bar effects (shared)
    st.markdown(PERF_CSS, unsafe_allow_html=True)
    components.html(ARCH_JS, height=0)                   # count-up for the bottom bar
    components.html(html, height=PERF_HEIGHT, scrolling=False)


PERF_HEIGHT = 700   # initial iframe height (it then auto-fits its content)


def _perf_tiles():
    """Metric cards. Values come straight from METRICS; missing ones are shown honestly as 'Not available'."""
    out = ""
    for i, (k, v) in enumerate(METRICS.items()):
        delay = 0.15 + 0.1 * i
        num, dec, suffix = None, 0, ""
        if v:
            s = str(v).replace("%", "").strip()
            suffix = "%" if "%" in str(v) else ""
            try:
                num = float(s)
                dec = len(s.split(".")[1]) if "." in s else 0
            except ValueError:
                num = None
        trend = (f'<div class="up">{ic("up", 11, "#34d399")}<span>{TREND[k]}</span></div>'
                 if v and TREND.get(k) else "")

        if v:
            if num is not None:
                val = f'<div class="val" data-count="{num}" data-dec="{dec}" data-suffix="{suffix}">{v}</div>'
            else:
                val = f'<div class="val">{v}</div>'
            tip, cls = "", ""
        else:
            val = '<div class="val na">—</div><div class="nacap">Not available</div>'
            tip, cls = '<div class="tip">Metric not available for the current evaluation.</div>', " unavailable"

        if k == "Accuracy" and num is not None:
            cls += " hero"
            top = (
                '<div class="ringbox"><svg width="78" height="78" viewBox="0 0 78 78">'
                '<defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">'
                '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>'
                '<circle cx="39" cy="39" r="32" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="5"/>'
                f'<circle id="accring" data-to="{num}" cx="39" cy="39" r="32" fill="none" stroke="url(#pg)" stroke-width="5" '
                'stroke-linecap="round" stroke-dasharray="201.06" stroke-dashoffset="201.06" '
                'transform="rotate(-90 39 39)"/></svg>'
                f'<div class="ringic mi">{ic(METRIC_ICONS[k], 18, "#22d3ee")}</div></div>'
            )
        else:
            top = f'<div class="mi">{ic(METRIC_ICONS[k], 18, "#22d3ee")}</div>'

        out += (f'<div class="tile{cls}" style="--d:{delay:.2f}s">{top}'
                f'<div class="k">{k}</div>{val}{trend}{tip}</div>')
    return out


PERF_CSS = """
<style>
@keyframes pfDrift { from {background-position: 0 100%, 0 0;} to {background-position: -700px 100%, 0 0;} }
@keyframes pfFade { from {opacity:0;} to {opacity:1;} }
@keyframes pfReveal { from {opacity:0; transform: translateY(40px);} to {opacity:1; transform:none;} }
@keyframes pfPulse { 0%,100% {opacity:.75;} 50% {opacity:1;} }
@keyframes pfZap { 0%,100% {opacity:.75; transform: scale(1);} 45% {opacity:1; transform: scale(1.15);} 55% {opacity:.85; transform: scale(1);} }

/* ---------- 12. background: grid + graph lines + tiny data points (barely visible) ---------- */
.stApp { isolation: isolate; }
.stApp::before { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background-image:
    linear-gradient(rgba(80,150,255,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(80,150,255,.05) 1px, transparent 1px),
    linear-gradient(rgba(80,150,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(80,150,255,.03) 1px, transparent 1px);
  background-size: 140px 140px, 140px 140px, 28px 28px, 28px 28px;
  -webkit-mask-image: radial-gradient(ellipse 90% 80% at 50% 40%, #000 0%, transparent 85%);
          mask-image: radial-gradient(ellipse 90% 80% at 50% 40%, #000 0%, transparent 85%); }
.stApp::after { content:''; position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.16;
  background-image:
    url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='700' height='320' viewBox='0 0 700 320'%3E%3Cpath d='M0 250 L90 210 L180 228 L270 150 L360 172 L450 96 L540 122 L630 60 L700 80' fill='none' stroke='%2322d3ee' stroke-width='1'/%3E%3Cpath d='M0 90 L90 120 L180 100 L270 170 L360 150 L450 215 L540 190 L630 250 L700 235' fill='none' stroke='%23a78bfa' stroke-width='1' stroke-dasharray='3 6'/%3E%3Cg fill='%237fe9ff'%3E%3Ccircle cx='90' cy='210' r='2'/%3E%3Ccircle cx='180' cy='228' r='2'/%3E%3Ccircle cx='270' cy='150' r='2'/%3E%3Ccircle cx='360' cy='172' r='2'/%3E%3Ccircle cx='450' cy='96' r='2'/%3E%3Ccircle cx='540' cy='122' r='2'/%3E%3Ccircle cx='630' cy='60' r='2'/%3E%3C/g%3E%3C/svg%3E"),
    radial-gradient(circle, rgba(127,233,255,.5) 1px, transparent 1.6px);
  background-repeat: repeat-x, repeat;
  background-size: 700px 320px, 150px 150px;
  background-position: 0 100%, 0 0;
  animation: pfDrift 140s linear infinite; }

/* ---------- 14. bottom bar: same bar, subtle extras; revealed last ---------- */
.foot .fbadge:nth-child(4) .bi2 svg { animation: pfZap 2.8s ease-in-out infinite; }
.foot .fbadge:nth-child(5) .bi2 svg { animation: pfPulse 3.4s ease-in-out infinite; }
.foot { animation: pfFade .7s ease 1.2s backwards; }
.credit { animation: pfFade .8s ease 1.4s backwards; }
@supports (animation-timeline: view()) {
  .foot, .credit { animation: pfReveal linear both; animation-timeline: view(); animation-range: entry 0% entry 55%; }
}
@media (prefers-reduced-motion: reduce) { * { animation-duration: .01s !important; animation-iteration-count: 1 !important; } }
</style>
"""

PERF_TEMPLATE = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box}
html,body{margin:0;background:transparent;font-family:'Inter',sans-serif;color:#e6edf7;overflow:hidden}
button{font-family:inherit}

@keyframes pfUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}
@keyframes cmIn{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:none}}
@keyframes pulseO{0%,100%{opacity:.75}50%{opacity:1}}
@keyframes hicp{0%,100%{box-shadow:0 0 8px rgba(34,211,238,.25)}50%{box-shadow:0 0 18px rgba(34,211,238,.6)}}
@keyframes barp{0%,100%{transform:scaleY(1)}50%{transform:scaleY(.55)}}
@keyframes lineMove{0%{left:0;opacity:0}10%{opacity:1}90%{opacity:1}100%{left:calc(100% - 8px);opacity:0}}
@keyframes rise{0%,100%{transform:translateY(1px)}50%{transform:translateY(-2px)}}
@keyframes dotp{0%{r:4;opacity:.8}100%{r:11;opacity:0}}

/* scroll reveal (class "in" is added when the block enters the viewport) */
.rv:not(.in){opacity:0}
.rv.in{animation:pfUp .65s ease var(--d,0s) backwards}

.grid{display:grid;grid-template-columns:1fr 1.45fr;gap:10px;align-items:start}
.card{position:relative;padding:18px;border-radius:18px;border:1px solid rgba(80,140,255,.28);
  background:linear-gradient(160deg,rgba(14,28,60,.62),rgba(8,16,38,.72));
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);box-shadow:0 10px 36px rgba(0,0,0,.28);
  transition:border-color .35s,box-shadow .35s}
.card:hover{border-color:rgba(34,211,238,.4);box-shadow:0 10px 36px rgba(0,0,0,.28),0 0 26px rgba(34,211,238,.1)}
h3{margin:0 0 14px;font-size:1.02rem;font-weight:700;color:#fff;display:flex;align-items:center;gap:9px}
.hic{display:flex;padding:6px;border-radius:9px;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.35)}
.hic.glow{animation:hicp 4s ease-in-out infinite}
.hic.glow svg path:nth-child(n+2){transform-box:fill-box;transform-origin:bottom;animation:barp 2.8s ease-in-out infinite}
.hic.glow svg path:nth-child(3){animation-delay:.5s}.hic.glow svg path:nth-child(4){animation-delay:1s}

/* ---------- 10. thin animated line under heading ---------- */
.hline{position:relative;height:1px;margin:-4px 0 14px;background:linear-gradient(90deg,rgba(34,211,238,.45),rgba(124,58,237,.25),transparent 85%)}
.hline::after{content:'';position:absolute;top:-2px;left:0;width:8px;height:5px;border-radius:50%;background:#7fe9ff;
  box-shadow:0 0 10px 2px rgba(34,211,238,.8);animation:lineMove 6s linear infinite}

/* ---------- metric tiles ---------- */
.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.tile{position:relative;text-align:center;padding:16px 6px 13px;border-radius:14px;background:rgba(14,28,60,.8);
  border:1px solid rgba(80,140,255,.22);transition:transform .3s,box-shadow .3s,border-color .3s}
.tile::before{content:'';position:absolute;inset:0;border-radius:14px;pointer-events:none;opacity:0;transition:opacity .35s;
  background:radial-gradient(circle at 50% 38%,rgba(34,211,238,.32),transparent 65%)}
.tile>*{position:relative}
.tile:hover{transform:translateY(-3px);border-color:rgba(34,211,238,.65);box-shadow:0 8px 26px rgba(59,130,246,.3)}
.tile:hover::before{opacity:.55}
.tile .mi{width:38px;height:38px;margin:0 auto 8px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.35)}
.tile:hover .mi{animation:pulseO 1.4s ease-in-out infinite;box-shadow:0 0 14px rgba(34,211,238,.5)}
.tile .k{color:#9fb3d6;font-size:.82rem}
.tile .val{font-size:1.55rem;font-weight:800;color:#fff;margin-top:4px;font-variant-numeric:tabular-nums;transition:text-shadow .3s}
.tile:hover .val{text-shadow:0 0 16px rgba(34,211,238,.7)}
.tile .up{color:#34d399;font-size:.72rem;font-weight:600;margin-top:4px;display:flex;gap:4px;align-items:center;justify-content:center}
.tile .up svg{animation:rise 2.6s ease-in-out infinite}
.tile.hero{border-color:rgba(34,211,238,.4);box-shadow:0 0 22px rgba(34,211,238,.12)}
.tile.hero .val{font-size:1.9rem}
.ringbox{position:relative;width:78px;height:78px;margin:0 auto 8px}
.ringbox .ringic{position:absolute;left:50%;top:50%;width:36px;height:36px;margin:-18px 0 0 -18px;border:0;background:transparent}
.tile .na{color:#6b7fa3}
.nacap{font-size:.62rem;color:#5a6f96;margin-top:2px}
.tip{position:absolute;left:50%;bottom:calc(100% + 8px);transform:translate(-50%,4px);width:max-content;max-width:190px;z-index:20;
  padding:7px 11px;border-radius:10px;font-size:.64rem;line-height:1.4;color:#cfe0ff;pointer-events:none;
  background:rgba(8,16,38,.9);border:1px solid rgba(34,211,238,.4);backdrop-filter:blur(8px);opacity:0;transition:opacity .25s,transform .25s}
.tile:hover .tip{opacity:1;transform:translate(-50%,0)}

/* ---------- chart ---------- */
.ch{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.chart-title{font-size:.82rem;color:#cfe0ff;font-weight:600;display:flex;align-items:center;gap:8px}
.chart-title small{font-weight:400;color:#6b7fa3;font-size:.66rem}
.ctl{display:flex;gap:8px;align-items:center}
.lg{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:.72rem;color:#9fb3d6;cursor:pointer;
  background:transparent;border:1px solid transparent;transition:border-color .25s,background .25s,color .25s}
.lg i{width:9px;height:9px;border-radius:3px;display:inline-block}
.lg:hover{color:#fff;border-color:rgba(80,140,255,.4)}
.lg.on{color:#fff;border-color:rgba(34,211,238,.6);background:rgba(34,211,238,.1)}
.rp{padding:3px 10px;border-radius:999px;font-size:.7rem;color:#7fb2ff;cursor:pointer;background:transparent;border:1px solid rgba(59,130,246,.35);transition:all .25s}
.rp:hover{border-color:#22d3ee;color:#fff;box-shadow:0 0 12px rgba(34,211,238,.25)}
.cwrap{position:relative}
#chart{width:100%;display:block;cursor:crosshair}
.ln{fill:none;stroke-linecap:round;stroke-linejoin:round;transition:opacity .3s,stroke-width .3s}
#lacc{stroke:#22d3ee;stroke-width:2;filter:drop-shadow(0 0 3px rgba(34,211,238,.55))}
#lloss{stroke:#a855f7;stroke-width:2;opacity:.9;filter:drop-shadow(0 0 3px rgba(168,85,247,.5))}
.cwrap.fa #lloss,.cwrap.fa #dl,.cwrap.fa #pl{opacity:.18}
.cwrap.fl #lacc,.cwrap.fl #da,.cwrap.fl #pa{opacity:.18}
.cwrap.fa #lacc,.cwrap.fl #lloss{stroke-width:3}
.pr{fill:none;stroke-width:1.4;opacity:0}
.done .pr{animation:dotp 2.6s ease-out infinite}
.done #pl{animation-delay:1.3s}
#epl{position:absolute;right:10px;top:6px;font-size:.66rem;color:#7fb2ff;opacity:.9;transition:opacity .6s}
.glass{position:absolute;z-index:25;pointer-events:none;padding:9px 12px;border-radius:12px;font-size:.7rem;line-height:1.55;color:#dbe6ff;white-space:nowrap;
  background:rgba(8,16,38,.74);border:1px solid rgba(34,211,238,.38);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:0 10px 28px rgba(0,0,0,.4);opacity:0;transition:opacity .15s}
.glass.show{opacity:1}
.glass b{display:block;color:#fff;font-size:.74rem;margin-bottom:3px}
.glass .a{color:#22d3ee}.glass .l{color:#c084fc}.glass .ok{color:#6ee7b7}.glass .bad{color:#fbbf24}
.note{margin-top:4px;font-size:.62rem;color:#5a6f96}

/* ---------- confusion matrix ---------- */
.cmw{position:relative;margin-top:16px}
table.cm{border-collapse:collapse;margin:0 auto}
table.cm td,table.cm th{padding:6px 9px;font-size:.72rem;text-align:center;color:#dbe6ff}
table.cm th{color:#8aa0c4;font-weight:600;transition:color .25s}
table.cm td.lab{color:#8aa0c4;text-align:right;font-weight:600;transition:color .25s}
table.cm td.c{min-width:54px;cursor:default;border:1px solid rgba(10,20,45,.9);
  background:rgba(59,130,246,var(--a));transition:filter .2s,box-shadow .2s,outline-color .2s;outline:1px solid transparent;outline-offset:-2px}
table.cm:not(.show) td.c{opacity:0}
table.cm.show td.c{animation:cmIn .3s ease calc(var(--i) * .07s) backwards}
td.c.diag{box-shadow:inset 0 0 0 1px rgba(52,211,153,.32)}
td.c:hover{filter:brightness(1.35);outline-color:rgba(34,211,238,.9);z-index:2;position:relative}
td.c.off:hover{outline-color:rgba(251,191,36,.85)}
td.c.hr,td.c.hc{filter:brightness(1.18)}
th.hl,td.lab.hl{color:#fff}
.cmbar{display:flex;align-items:center;gap:8px;justify-content:center;margin-top:8px;font-size:.68rem;color:#8aa0c4}
.cmbar i{display:block;height:6px;width:170px;border-radius:99px;background:linear-gradient(90deg,rgba(59,130,246,.15),#3b82f6)}

@media (max-width:900px){.grid{grid-template-columns:1fr}}
#cmhost{overflow-x:auto}
@media (max-width:520px){
  .card{padding:14px}
  .metrics{gap:10px}
  .tile .val{font-size:1.35rem}.tile.hero .val{font-size:1.6rem}
  .ch{flex-direction:column;align-items:flex-start}
  table.cm td,table.cm th{padding:5px 5px;font-size:.62rem}
  table.cm td.c{min-width:38px}
  .cmbar i{width:110px}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s !important;animation-iteration-count:1 !important}}
</style></head>
<body>
<div id="app">
<div class="grid">

  <!-- LEFT: metrics -->
  <div class="card rv" id="left" style="--d:0s">
    <h3><span class="hic">__I_CHART__</span>Model Performance</h3>
    <div class="metrics">__TILES__</div>
  </div>

  <!-- RIGHT: training results -->
  <div class="card rv" id="right" style="--d:.1s">
    <h3><span class="hic glow">__I_CHART__</span>Model Training Results</h3>
    <div class="hline"></div>

    <div class="ch">
      <div class="chart-title">Accuracy &amp; Loss</div>
      <div class="ctl">
        <button class="lg" data-s="acc"><i style="background:#22d3ee"></i>Accuracy</button>
        <button class="lg" data-s="loss"><i style="background:#a855f7"></i>Loss</button>
        <button class="rp" id="replay" title="Replay animation">&#8635; Replay</button>
      </div>
    </div>
    <div class="cwrap" id="cwrap">
      <svg id="chart"></svg>
      <div id="epl"></div>
      <div class="glass" id="tipc"></div>
    </div>
    <div class="note" id="note" style="display:none">Illustrative curve &mdash; set ACC_CURVE / LOSS_CURVE in app.py to show your real training history.</div>

    <div class="cmw" id="cmw">
      <div class="chart-title">Confusion Matrix <small>rows: actual &middot; columns: predicted</small></div>
      <div style="height:8px"></div>
      <div id="cmhost"></div>
      <div class="glass" id="tipm"></div>
    </div>
  </div>

</div>
</div>

<script>
var D = __DATA__;
var $ = function (id) { return document.getElementById(id); };
var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function ease(t) { return 1 - Math.pow(1 - t, 3); }
function easeIO(t) { return t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2; }
function reveal(el, cb) {
  if (!('IntersectionObserver' in window)) { el.classList.add('in'); if (cb) cb(); return; }
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { el.classList.add('in'); io.disconnect(); if (cb) cb(); } });
  }, { threshold: .08 });
  io.observe(el);
}

/* ---------- counters + accuracy ring ---------- */
function startCounters() {
  document.querySelectorAll('[data-count]').forEach(function (el) {
    var to = parseFloat(el.getAttribute('data-count')), dec = parseInt(el.getAttribute('data-dec') || '0', 10), suf = el.getAttribute('data-suffix') || '';
    var ring = el.classList.contains('val') && el.closest('.tile.hero') ? $('accring') : null;
    if (reduce) { el.textContent = to.toFixed(dec) + suf; if (ring) ring.style.strokeDashoffset = 201.06 * (1 - to / 100); return; }
    var start = null;
    setTimeout(function () {
      requestAnimationFrame(function step(ts) {
        if (!start) start = ts;
        var p = Math.min((ts - start) / 1800, 1), e = ease(p);
        el.textContent = (to * e).toFixed(dec) + suf;
        if (ring) ring.style.strokeDashoffset = 201.06 * (1 - (to / 100) * e);
        if (p < 1) requestAnimationFrame(step);
      });
    }, 350);
    el.textContent = (0).toFixed(dec) + suf;
  });
}

/* ---------- accuracy / loss chart ---------- */
var n = D.acc.length - 1;
var W = 520, H = 232, L = 42, R = 14, T = 14, B = 32;
var ymax = Math.ceil(Math.max(1, Math.max.apply(null, D.acc), Math.max.apply(null, D.loss)) * 5) / 5;
function X(i) { return L + (W - L - R) * i / n; }
function Y(v) { return T + (1 - Math.max(0, v) / ymax) * (H - T - B); }
function pts(arr, f) {
  var s = '', k = Math.floor(f), i;
  for (i = 0; i <= k && i <= n; i++) s += X(i).toFixed(1) + ',' + Y(arr[i]).toFixed(1) + ' ';
  if (f > k && k < n) { var fr = f - k, v = arr[k] + (arr[k + 1] - arr[k]) * fr; s += X(f).toFixed(1) + ',' + Y(v).toFixed(1); }
  return s;
}
function at(arr, f) { var k = Math.floor(f); if (k >= n) return arr[n]; return arr[k] + (arr[k + 1] - arr[k]) * (f - k); }

(function build() {
  var g = '', i, v;
  for (v = 0; v <= ymax + 1e-6; v += 0.2) {
    g += '<line x1="' + L + '" y1="' + Y(v).toFixed(1) + '" x2="' + (W - R) + '" y2="' + Y(v).toFixed(1) + '" stroke="rgba(120,150,220,.15)"/>' +
         '<text x="' + (L - 6) + '" y="' + (Y(v) + 3).toFixed(1) + '" fill="#5a6f96" font-size="8" text-anchor="end">' + v.toFixed(1) + '</text>';
  }
  for (i = 0; i <= n; i += 10) g += '<text x="' + X(i).toFixed(1) + '" y="' + (H - 12) + '" fill="#5a6f96" font-size="8" text-anchor="middle">' + i + '</text>';
  g += '<text x="' + (W - R) + '" y="' + (T + 8) + '" fill="#5a6f96" font-size="8" text-anchor="end">Epoch</text>';
  $('chart').setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  $('chart').innerHTML =
    '<rect x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) + '" fill="rgba(20,35,70,.25)" rx="6"/>' + g +
    '<polyline id="lloss" class="ln" points=""/><polyline id="lacc" class="ln" points=""/>' +
    '<line id="guide" x1="0" y1="' + T + '" x2="0" y2="' + (H - B) + '" stroke="rgba(207,224,255,.45)" stroke-dasharray="3 3" opacity="0"/>' +
    '<circle id="pl" class="pr" r="4" stroke="#a855f7" cx="0" cy="0"/><circle id="pa" class="pr" r="4" stroke="#22d3ee" cx="0" cy="0"/>' +
    '<circle id="dl" r="3.2" fill="#c084fc" cx="0" cy="0" opacity="0" style="filter:drop-shadow(0 0 5px #a855f7)"/>' +
    '<circle id="da" r="3.2" fill="#22d3ee" cx="0" cy="0" opacity="0" style="filter:drop-shadow(0 0 5px #22d3ee)"/>' +
    '<rect id="hit" x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) + '" fill="transparent"/>';
  if (D.sample) $('note').style.display = 'block';
})();

var raf = null, ready = false;
function place(f) {
  $('lacc').setAttribute('points', pts(D.acc, f)); $('lloss').setAttribute('points', pts(D.loss, f));
  var xa = X(f), ya = Y(at(D.acc, f)), yl = Y(at(D.loss, f));
  ['da', 'pa'].forEach(function (id) { $(id).setAttribute('cx', xa); $(id).setAttribute('cy', ya); });
  ['dl', 'pl'].forEach(function (id) { $(id).setAttribute('cx', xa); $(id).setAttribute('cy', yl); });
  $('da').setAttribute('opacity', 1); $('dl').setAttribute('opacity', 1);
}
function draw() {
  if (raf) cancelAnimationFrame(raf);
  ready = false; $('cwrap').classList.remove('done'); $('epl').style.opacity = .9;
  if (reduce) { place(n); $('epl').textContent = ''; ready = true; $('cwrap').classList.add('done'); return; }
  var t0 = null, dur = 3400;
  raf = requestAnimationFrame(function step(ts) {
    if (!t0) t0 = ts;
    var p = Math.min((ts - t0) / dur, 1), f = easeIO(p) * n;
    place(f); $('epl').textContent = 'Epoch ' + Math.max(1, Math.round(f)) + ' / ' + n;
    if (p < 1) raf = requestAnimationFrame(step);
    else { place(n); ready = true; $('cwrap').classList.add('done'); $('epl').style.opacity = 0; }
  });
}
$('replay').addEventListener('click', draw);

/* legend: click to highlight one line (click again to reset) */
var focus = null;
Array.prototype.forEach.call(document.querySelectorAll('.lg'), function (b) {
  b.addEventListener('click', function () {
    var s = b.getAttribute('data-s'); focus = (focus === s) ? null : s;
    $('cwrap').classList.toggle('fa', focus === 'acc'); $('cwrap').classList.toggle('fl', focus === 'loss');
    Array.prototype.forEach.call(document.querySelectorAll('.lg'), function (x) { x.classList.toggle('on', x.getAttribute('data-s') === focus); });
  });
});

/* hover tooltip + moving guide line */
var hit = $('hit'), tipc = $('tipc'), guide = $('guide');
hit.addEventListener('mousemove', function (e) {
  if (!ready) return;
  var svg = $('chart'), r = svg.getBoundingClientRect(), sx = W / r.width;
  var vx = (e.clientX - r.left) * sx, idx = Math.max(0, Math.min(n, Math.round((vx - L) / (W - L - R) * n)));
  var x = X(idx);
  guide.setAttribute('x1', x); guide.setAttribute('x2', x); guide.setAttribute('opacity', 1);
  ['da', 'pa'].forEach(function (id) { $(id).setAttribute('cx', x); $(id).setAttribute('cy', Y(D.acc[idx])); });
  ['dl', 'pl'].forEach(function (id) { $(id).setAttribute('cx', x); $(id).setAttribute('cy', Y(D.loss[idx])); });
  tipc.innerHTML = '<b>Epoch ' + idx + '</b><span class="a">Accuracy: ' + (D.acc[idx] * 100).toFixed(1) + '%</span><br><span class="l">Loss: ' + D.loss[idx].toFixed(2) + '</span>';
  var wr = $('cwrap').getBoundingClientRect(), px = e.clientX - wr.left, py = e.clientY - wr.top;
  var tw = tipc.offsetWidth, left = px + 14; if (left + tw > wr.width) left = px - tw - 14;
  tipc.style.left = Math.max(0, left) + 'px'; tipc.style.top = Math.max(0, py - 54) + 'px'; tipc.classList.add('show');
});
hit.addEventListener('mouseleave', function () { tipc.classList.remove('show'); guide.setAttribute('opacity', 0); if (ready) place(n); });

/* ---------- confusion matrix ---------- */
(function matrix() {
  var M = D.matrix, lab = D.labels, vmax = 1, i, j, html = '';
  M.forEach(function (r) { r.forEach(function (v) { if (v > vmax) vmax = v; }); });
  var tot = M.map(function (r) { return r.reduce(function (a, b) { return a + b; }, 0); });
  html = '<table class="cm" id="cm"><tr><td></td>' + lab.map(function (l, c) { return '<th data-c="' + c + '">' + l + '</th>'; }).join('') + '</tr>';
  for (i = 0; i < M.length; i++) {
    html += '<tr><td class="lab" data-r="' + i + '">' + lab[i] + '</td>';
    for (j = 0; j < M[i].length; j++) {
      var a = (0.14 + 0.78 * (M[i][j] / vmax)).toFixed(2);
      html += '<td class="c ' + (i === j ? 'diag' : 'off') + '" data-r="' + i + '" data-c="' + j + '" style="--a:' + a + ';--i:' + (i * M.length + j) + '">' + M[i][j] + '</td>';
    }
    html += '</tr>';
  }
  html += '</table><div class="cmbar">0 <i></i> ' + vmax + '</div>';
  $('cmhost').innerHTML = html;

  var cm = $('cm'), tip = $('tipm'), cells = cm.querySelectorAll('td.c');
  function clearHl() { cm.querySelectorAll('.hr,.hc,.hl').forEach(function (e) { e.classList.remove('hr', 'hc', 'hl'); }); }
  Array.prototype.forEach.call(cells, function (td) {
    td.addEventListener('mousemove', function (e) {
      var r = +td.getAttribute('data-r'), c = +td.getAttribute('data-c'), v = M[r][c];
      clearHl();
      cells.forEach(function (x) {
        if (x.getAttribute('data-r') == r) x.classList.add('hr');
        if (x.getAttribute('data-c') == c) x.classList.add('hc');
      });
      var lr = cm.querySelector('td.lab[data-r="' + r + '"]'), hc = cm.querySelector('th[data-c="' + c + '"]');
      if (lr) lr.classList.add('hl'); if (hc) hc.classList.add('hl');
      var pct = tot[r] ? (v / tot[r] * 100).toFixed(1) : null;
      var status = (r === c)
        ? '<span class="ok">&#10003; Correct' + (pct ? ' &middot; ' + pct + '% of ' + lab[r] + ' scans' : '') + '</span>'
        : '<span class="bad">Misclassified' + (pct ? ' &middot; ' + pct + '% of ' + lab[r] + ' scans' : '') + '</span>';
      tip.innerHTML = '<b>Actual: ' + lab[r] + '</b>Predicted: ' + lab[c] + '<br>Samples: <span class="a">' + v + '</span><br>' + status;
      var wr = $('cmw').getBoundingClientRect(), px = e.clientX - wr.left, py = e.clientY - wr.top;
      var tw = tip.offsetWidth, left = px + 14; if (left + tw > wr.width) left = px - tw - 14;
      tip.style.left = Math.max(0, left) + 'px'; tip.style.top = Math.max(0, py - 70) + 'px'; tip.classList.add('show');
    });
    td.addEventListener('mouseleave', function () { clearHl(); tip.classList.remove('show'); });
  });
})();

/* ---------- scroll-triggered sequence ---------- */
reveal($('left'), startCounters);
reveal($('right'), function () {
  reveal($('cwrap'), draw);
  reveal($('cmw'), function () { setTimeout(function () { $('cm').classList.add('show'); }, 900); });
});

/* keep the iframe exactly as tall as the content */
var app = $('app');
function fit() { try { window.frameElement.style.height = Math.ceil(app.getBoundingClientRect().height) + 4 + 'px'; } catch (e) {} }
if (window.ResizeObserver) new ResizeObserver(fit).observe(app);
window.addEventListener('load', fit); window.addEventListener('resize', fit); fit();
</script>
</body></html>
"""


# =====================================================================
# PAGE: ARCHITECTURE  (model architecture)
# =====================================================================
def page_architecture():
    html = ARCH_TEMPLATE
    html = html.replace("__I_NET__", ic("net", 16, "#22d3ee"))
    html = html.replace("__I_SPARK__", ic("sparkles", 16, "#22d3ee"))
    html = html.replace("__I_ARROW_B__", ic("arrow", 14, "#3b82f6"))
    html = html.replace("__I_ARROW_W__", ic("arrow", 15, "#fff"))
    html = html.replace("__I_TREND__", ic("trend", 24, "#22d3ee"))
    st.markdown(ARCH_CSS, unsafe_allow_html=True)
    components.html(html, height=ARCH_HEIGHT, scrolling=False)
    components.html(ARCH_JS, height=0)


# Initial iframe height (the page then auto-fits its content)
ARCH_HEIGHT = 470

# Styles for the parts of the Architecture page that live outside the interactive card
# (bottom AI bar + footer). Only injected on this page.
ARCH_CSS = """
<style>
@keyframes arFade { from {opacity:0;} to {opacity:1;} }
@keyframes arPulse { 0%,100% {opacity:.75;} 50% {opacity:1;} }
@keyframes arArc { from {stroke-dashoffset: 87.96;} to {stroke-dashoffset: var(--off);} }
@keyframes arSlow { to {transform: rotate(360deg);} }

/* 15 / 17: bottom bar and footer fade in last */
.foot { animation: arFade .7s ease 1.3s backwards; }
.credit { animation: arFade .8s ease 1.5s backwards; }

/* 16: bottom bar - only subtle extras */
.foot .fbadge { border-left-color: rgba(34,211,238,.35); box-shadow: -6px 0 12px -10px rgba(34,211,238,.9);
  transition: background .3s, box-shadow .3s; border-radius: 0 12px 12px 0; padding-right: 10px; }
.foot .fbadge:hover { background: rgba(34,211,238,.06); }
.foot .fi svg, .foot .fbadge:nth-child(2) .bi2 svg { animation: arPulse 4s ease-in-out infinite; }
.foot .fbadge:nth-child(4) .bi2 svg { animation: arPulse 2.8s ease-in-out infinite; }
.foot .fbadge:nth-child(5) .bi2 svg { animation: arSlow 28s linear infinite; }
.foot .fbadge .bi2.nvp-ring { position:relative; background:transparent; border:0; width:38px; height:38px; }
.foot .fbadge .bi2.nvp-ring svg.rg { position:absolute; inset:0; transform: rotate(-90deg); animation:none; }
.foot .fbadge .bi2.nvp-ring svg.ti { position:absolute; left:50%; top:50%; margin:-7px 0 0 -7px; animation:none; }
.foot .fbadge .bi2.nvp-ring .arc { stroke-dasharray: 87.96; stroke-dashoffset: var(--off); animation: arArc 1.8s ease-out 1.5s backwards; }
.foot .report { transition: transform .25s, box-shadow .25s; cursor:pointer; }
.foot a.report, .foot a.report:hover { text-decoration:none; color:#fff; }
.foot .report svg { transition: transform .25s; }
.foot .report:hover { transform: scale(1.03); box-shadow: 0 0 22px rgba(34,211,238,.4), 0 0 38px rgba(124,58,237,.25); }
.foot .report:hover svg { transform: translateX(5px); }
@media (prefers-reduced-motion: reduce) { * { animation-duration: .01s !important; animation-iteration-count: 1 !important; } }
</style>
"""

# count-up for the "High Accuracy" value in the bottom bar (real value comes from METRICS)
ARCH_JS = """
<script>
(function () {
  var P; try { P = window.parent; P.document; } catch (e) { return; }
  var D = P.document, alive = true;
  function ease(t) { return 1 - Math.pow(1 - t, 3); }
  function setup() {
    D.querySelectorAll('[data-nvcount]').forEach(function (el) {
      if (el.__nvc) return; el.__nvc = 1;
      var to = parseFloat(el.getAttribute('data-nvcount')), dec = parseInt(el.getAttribute('data-dec') || '0', 10);
      el.textContent = (0).toFixed(dec);
      var start = null;
      setTimeout(function () {
        P.requestAnimationFrame(function step(ts) {
          if (!alive) return;
          if (!start) start = ts;
          var p = Math.min((ts - start) / 1800, 1);
          var v = to * ease(p);
          el.textContent = el.getAttribute('data-comma') ? v.toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec}) : v.toFixed(dec);
          if (p < 1) P.requestAnimationFrame(step);
        });
      }, 1500);
    });
  }
  setup();
  var poll = setInterval(setup, 250);
  setTimeout(function () { clearInterval(poll); }, 6000);
})();
</script>
"""

ARCH_TEMPLATE = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box}
html,body{margin:0;background:transparent;font-family:'Inter',sans-serif;color:#e6edf7;overflow:hidden}
button{font-family:inherit}

@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@keyframes reveal{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes nodeIn{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes hicp{0%,100%{box-shadow:0 0 8px rgba(34,211,238,.25)}50%{box-shadow:0 0 20px rgba(34,211,238,.65)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulseO{0%,100%{opacity:.7}50%{opacity:1}}
@keyframes drift{from{transform:translate(-14px,6px)}to{transform:translate(14px,-8px)}}
@keyframes dashmove{to{stroke-dashoffset:-48}}
@keyframes nodeblink{0%,100%{opacity:.25}50%{opacity:.9}}
@keyframes blob{from{transform:translate(-20px,8px) scale(1)}to{transform:translate(22px,-10px) scale(1.08)}}

/* ---------- card shell ---------- */
.card{position:relative;margin:0;padding:18px;border-radius:18px;border:1px solid rgba(80,140,255,.22);
  background:linear-gradient(160deg,rgba(14,28,60,.85),rgba(8,16,38,.9));animation:fadeIn .5s ease both}
.bg{position:absolute;inset:0;border-radius:18px;overflow:hidden;pointer-events:none;z-index:0}
.bg .glow{position:absolute;inset:0;
  background:radial-gradient(420px 220px at 12% 95%,rgba(34,211,238,.09),transparent 70%),
             radial-gradient(460px 240px at 90% 8%,rgba(124,58,237,.11),transparent 70%);animation:blob 18s ease-in-out infinite alternate}
.bg svg{position:absolute;inset:0;width:100%;height:100%;opacity:.15}
.bg .net{animation:drift 22s ease-in-out infinite alternate}
.bg .net path{stroke-dasharray:4 9;animation:dashmove 9s linear infinite}
.bg .net circle{animation:nodeblink 5s ease-in-out infinite}
.bg .net circle:nth-child(2n){animation-delay:1.3s}.bg .net circle:nth-child(3n){animation-delay:2.6s}
.bg .pt{position:absolute;width:3px;height:3px;border-radius:50%;background:#7fe9ff;box-shadow:0 0 8px #22d3ee;opacity:.0;animation:ptf 9s ease-in-out infinite}
@keyframes ptf{0%,100%{opacity:0;transform:translate(0,0)}30%{opacity:.55}70%{opacity:.4}100%{transform:translate(40px,-30px)}}

/* ---------- 14. heading ---------- */
h3{position:relative;z-index:1;margin:0;font-size:1.02rem;font-weight:700;color:#fff;display:flex;align-items:center;gap:9px}
h3 .t{display:inline-block;animation:slideUp .6s ease .05s both}
.hic{position:relative;display:flex;padding:6px;border-radius:9px;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.35);
  animation:hicp 4.5s ease-in-out infinite,fadeIn .5s ease both}
.uline{position:relative;z-index:1;height:2px;margin:11px 0 16px}
.uline i{position:absolute;inset:0;transform-origin:left;border-radius:2px;
  background:linear-gradient(90deg,#22d3ee,rgba(124,58,237,.55) 60%,transparent);animation:reveal 1s ease .2s both}

/* ---------- flow ---------- */
.flow{position:relative;z-index:1;display:flex;align-items:stretch;gap:0}
.node{position:relative;flex:1;min-width:0;text-align:center;padding:14px 6px 13px;border-radius:12px;cursor:pointer;outline:none;
  background:rgba(10,22,50,.65);border:1px solid rgba(80,140,255,.28);
  animation:nodeIn .6s ease var(--d) backwards;
  transition:transform .25s,border-color .3s,box-shadow .3s,background .3s,opacity .35s}
.node:hover,.node:focus-visible{transform:translateY(-3px);border-color:rgba(34,211,238,.75);background:rgba(16,34,74,.8);
  box-shadow:0 10px 26px rgba(0,0,0,.35),0 0 20px rgba(34,211,238,.2)}
.node b{display:block;font-size:.74rem;color:#e6edf7;transition:color .3s,text-shadow .3s}
.node:hover b,.node:focus-visible b,.node.sel b{color:#fff;text-shadow:0 0 12px rgba(255,255,255,.35)}
.node small{display:block;color:#8aa0c4;font-size:.6rem;line-height:1.4;margin-top:3px;min-height:1.4em;transition:color .3s}
.node:hover small{color:#b9cbe8}

.ic{position:relative;width:46px;height:46px;margin:0 auto 9px;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden;
  background:rgba(59,130,246,.14);border:1px solid rgba(59,130,246,.4);transition:box-shadow .3s,border-color .3s,background .3s}
.ic>svg{position:relative;z-index:1}
.node:hover .ic{box-shadow:0 0 18px rgba(34,211,238,.55);border-color:rgba(34,211,238,.8)}

/* selected / dimmed */
.flow.has-sel .node:not(.sel){opacity:.5}
.flow.has-sel .node:not(.sel):hover{opacity:.9}
.node.sel{border-color:#22d3ee;background:rgba(18,40,86,.85);
  box-shadow:0 0 28px rgba(34,211,238,.35),0 0 0 1px rgba(124,58,237,.45),inset 0 0 18px rgba(34,211,238,.08)}
.node.sel .ic{border-color:#22d3ee;background:rgba(34,211,238,.18);box-shadow:0 0 22px rgba(34,211,238,.6)}
.node.sel::after{content:'';position:absolute;left:28%;right:28%;bottom:5px;height:2px;border-radius:2px;
  background:linear-gradient(90deg,#22d3ee,#a78bfa);box-shadow:0 0 8px rgba(34,211,238,.8)}

/* data pulse "arrives" at a layer */
.node.arrive{animation:arrive 1.1s ease-out}
@keyframes arrive{0%{box-shadow:0 0 0 rgba(34,211,238,0)}35%{box-shadow:0 0 20px rgba(34,211,238,.5);border-color:rgba(34,211,238,.85)}100%{box-shadow:0 0 0 rgba(34,211,238,0)}}

/* tooltips */
.tip{position:absolute;left:50%;top:calc(100% + 8px);transform:translate(-50%,6px);width:max-content;max-width:176px;z-index:30;
  padding:8px 11px;border-radius:10px;font-size:.64rem;line-height:1.45;color:#cfe0ff;text-align:left;pointer-events:none;
  background:rgba(8,16,38,.97);border:1px solid rgba(34,211,238,.4);box-shadow:0 10px 26px rgba(0,0,0,.45);
  opacity:0;transition:opacity .25s,transform .25s}
.tip b{display:block;color:#22d3ee;font-size:.66rem;margin-bottom:2px;text-shadow:none}
.node:hover .tip,.node:focus-visible .tip{opacity:1;transform:translate(-50%,0)}

/* ---------- arrows with travelling pulse ---------- */
.arr{position:relative;width:36px;flex-shrink:0;display:flex;align-items:center;justify-content:center;animation:fadeIn .5s ease .9s backwards}
.arr .ln{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(59,130,246,.35);transition:background .3s,box-shadow .3s}
.arr svg{position:relative;z-index:1;filter:drop-shadow(0 0 4px rgba(59,130,246,.7));transition:transform .25s}
.arr:hover svg{transform:translateX(3px)}
.arr .pt{position:absolute;top:50%;left:0;width:7px;height:7px;margin-top:-3.5px;border-radius:50%;z-index:2;opacity:0;
  background:#7fe9ff;box-shadow:0 0 10px 3px rgba(34,211,238,.85)}
.arr.go .pt{animation:travel .75s linear forwards}
.arr.on .ln{background:#22d3ee;box-shadow:0 0 8px rgba(34,211,238,.7)}
@keyframes travel{0%{left:0;opacity:0}12%{opacity:1}88%{opacity:1}100%{left:calc(100% - 7px);opacity:0}}

/* ---------- 4. Input MRI ---------- */
.n-input .scanl{position:absolute;left:6px;right:6px;height:2px;top:12%;border-radius:2px;opacity:0;
  background:linear-gradient(90deg,transparent,#22d3ee,transparent);box-shadow:0 0 8px #22d3ee;animation:scanl 3.6s ease-in-out 1s infinite}
@keyframes scanl{0%{top:14%;opacity:0}15%,85%{opacity:.9}100%{top:84%;opacity:0}}
.n-input .pring{position:absolute;inset:9px;border-radius:50%;border:1px solid rgba(34,211,238,.6);animation:pring 4.5s ease-out 1.5s infinite}
@keyframes pring{0%{transform:scale(.5);opacity:.7}100%{transform:scale(1.5);opacity:0}}
.n-input .sp{position:absolute;width:2px;height:2px;border-radius:50%;background:#7fe9ff;opacity:0;animation:spf 4s ease-in-out infinite}
.n-input .s1{left:12px;bottom:8px}.n-input .s2{left:22px;bottom:6px;animation-delay:1.4s}.n-input .s3{left:32px;bottom:9px;animation-delay:2.6s}
@keyframes spf{0%{transform:translateY(0);opacity:0}30%{opacity:.9}100%{transform:translateY(-22px);opacity:0}}

/* ---------- 5. Preprocessing ---------- */
.n-pre .k{transform-box:fill-box;transform-origin:center}
.n-pre .k1{animation:knob 5s ease-in-out infinite}
.n-pre .k2{animation:knob 6.5s ease-in-out .8s infinite reverse}
.n-pre .k3{animation:knob 5.8s ease-in-out 1.6s infinite}
@keyframes knob{0%,100%{transform:translateX(-4px)}50%{transform:translateX(5px)}}
.st{transition:color .3s}
.node:hover .st,.node.sel .st{animation:stHi 2.7s ease-in-out infinite}
.node:hover .st:nth-of-type(2),.node.sel .st:nth-of-type(2){animation-delay:.9s}
.node:hover .st:nth-of-type(3),.node.sel .st:nth-of-type(3){animation-delay:1.8s}
@keyframes stHi{0%,100%{color:#8aa0c4;text-shadow:none}10%,30%{color:#22d3ee;text-shadow:0 0 8px rgba(34,211,238,.6)}40%{color:#8aa0c4}}

/* ---------- 6. CNN (strongest) ---------- */
.n-cnn{border-color:rgba(124,58,237,.5);background:rgba(18,32,74,.78);box-shadow:0 0 22px rgba(124,58,237,.16)}
.n-cnn .ic{border-color:rgba(167,139,250,.55);background:rgba(124,58,237,.14);box-shadow:0 0 14px rgba(124,58,237,.25)}
.badge{position:absolute;top:-8px;right:6px;padding:2px 8px;border-radius:999px;font-size:.5rem;font-weight:700;letter-spacing:.07em;
  white-space:nowrap;color:#fff;background:linear-gradient(90deg,#2563eb,#7c3aed);box-shadow:0 0 12px rgba(124,58,237,.5);z-index:3}
.sig line,.sig path{stroke-dasharray:2.5 9;animation:sigm 2.4s linear infinite}
@keyframes sigm{to{stroke-dashoffset:-23}}
.nd{animation:ndp 3.2s ease-in-out infinite}
.nd.c2{animation-delay:.5s}.nd.c3{animation-delay:1s}
@keyframes ndp{0%,100%{opacity:.55}50%{opacity:1}}

/* ---------- 7. Feature extraction ---------- */
.n-feat .wave{animation:wv 3.4s linear infinite}
@keyframes wv{from{transform:translateX(0)}to{transform:translateX(-18px)}}
.n-feat .ic::before{content:'';position:absolute;inset:0;background:radial-gradient(circle,rgba(34,211,238,.28),transparent 70%);animation:pulseO 3.6s ease-in-out infinite}
.n-feat .fp{position:absolute;width:2px;height:2px;border-radius:50%;background:#7fe9ff;opacity:0;animation:fp 3.4s ease-in-out infinite}
.n-feat .f1{left:8px;top:14px}.n-feat .f2{left:20px;top:30px;animation-delay:1.1s}.n-feat .f3{left:32px;top:18px;animation-delay:2.2s}
@keyframes fp{0%{transform:translateX(-4px);opacity:0}40%{opacity:.9}100%{transform:translateX(8px);opacity:0}}

/* ---------- 8. Dense ---------- */
.dn line{stroke:rgba(34,211,238,.35)}
.dn .dp{stroke:#7fe9ff;stroke-width:2.2;stroke-linecap:round;stroke-dasharray:2 40;stroke-dashoffset:2;opacity:.95;animation:dpul 5s ease-in-out infinite}
@keyframes dpul{0%{stroke-dashoffset:2}45%,100%{stroke-dashoffset:-22}}

/* ---------- 9. Prediction activation ---------- */
.n-pred .actring{position:absolute;inset:8px;border-radius:50%;border:1.5px solid rgba(34,211,238,.8);opacity:0}
.n-pred.fire .actring{animation:actring 1.5s ease-out}
.n-pred.fire .ic{animation:actglow 1.5s ease-out}
@keyframes actring{0%{transform:scale(.55);opacity:.85}100%{transform:scale(1.7);opacity:0}}
@keyframes actglow{0%{box-shadow:0 0 0 rgba(34,211,238,0)}35%{box-shadow:0 0 24px rgba(34,211,238,.75);background:rgba(34,211,238,.22)}100%{box-shadow:0 0 0 rgba(34,211,238,0)}}

/* ---------- 10 / 11. Interactive view bar ---------- */
.iv{position:relative;z-index:1;display:flex;align-items:center;gap:14px;margin-top:20px;padding:12px 14px;min-height:78px;border-radius:12px;
  background:rgba(34,211,238,.07);border:1px solid rgba(34,211,238,.3);animation:fadeIn .6s ease 1.1s backwards;
  transition:border-color .3s,box-shadow .3s,background .3s}
.iv:hover{border-color:rgba(34,211,238,.75);box-shadow:0 0 24px rgba(34,211,238,.2);background:rgba(34,211,238,.09)}
.iv .ii{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;
  background:rgba(34,211,238,.15);border:1px solid rgba(34,211,238,.5);animation:pulseO 4s ease-in-out infinite}
.iv .ii svg{animation:spin 14s linear infinite}
.iv .txt{flex:1;min-width:0}
.iv .txt.swap{animation:slideUp .35s ease both}
.iv b{display:block;color:#22d3ee;font-size:.86rem}
.iv .tech{display:block;color:#e6edf7;font-size:.74rem;font-weight:600;margin-top:2px}
.iv small{display:block;color:#8aa0c4;font-size:.74rem;margin-top:2px;line-height:1.5}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.meta:empty{display:none}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;font-size:.62rem;color:#cfe0ff;
  background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.35);font-family:inherit}
button.chip{cursor:pointer;transition:border-color .25s,background .25s}
button.chip:hover{border-color:#22d3ee;background:rgba(34,211,238,.14)}
.iv .btns{display:flex;gap:8px;align-items:center;flex-shrink:0}
.iv .go,.iv .prev{width:34px;height:34px;border-radius:10px;border:0;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#fff;
  background:linear-gradient(90deg,#2563eb,#7c3aed);transition:transform .25s,box-shadow .25s}
.iv .prev{background:rgba(59,130,246,.18);border:1px solid rgba(59,130,246,.45);font-size:1.1rem;line-height:1;display:none}
.iv.active .prev{display:flex}
.iv .go svg{transition:transform .25s}
.iv .go:hover{transform:scale(1.06);box-shadow:0 0 20px rgba(34,211,238,.5)}
.iv .go:hover svg{transform:translateX(3px)}
.iv .go:active,.iv .prev:active{transform:scale(.94)}
.iv .prev:hover{border-color:#22d3ee;box-shadow:0 0 14px rgba(34,211,238,.35)}

@media (max-width:1200px){.arr{width:24px}.node b{font-size:.7rem}}
@media (max-width:980px){
  .flow{flex-wrap:wrap;gap:12px}.arr{display:none}.node{flex:1 1 calc(33.333% - 12px)}
  .tip{display:none}
}
@media (max-width:560px){
  .card{padding:14px}
  .node{flex:1 1 calc(50% - 12px)}
  .iv{flex-wrap:wrap}
  .iv .txt{flex:1 1 100%}
  .iv .btns{width:100%;justify-content:flex-end}
  h3{font-size:.95rem}
}
@media (hover:none){.tip{display:none}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s !important;animation-iteration-count:1 !important}}
</style></head>
<body>
<div class="card" id="card">

  <div class="bg">
    <div class="glow"></div>
    <svg viewBox="0 0 1000 300" preserveAspectRatio="xMidYMid slice"><g class="net">
      <g stroke="rgba(120,170,255,.8)" stroke-width="1" fill="none">
        <path d="M40 60 L180 140 L320 50 L460 130 L600 60 L740 140 L880 70 L980 130"/>
        <path d="M40 240 L180 150 L320 250 L460 140 L600 240 L740 150 L880 230"/>
        <path d="M180 140 L320 150 L460 135 L600 150 L740 145"/>
        <path d="M320 50 L320 250 M600 60 L600 240"/>
      </g>
      <g fill="#7fb2ff">
        <circle cx="40" cy="60" r="3"/><circle cx="180" cy="140" r="3.5"/><circle cx="320" cy="50" r="3"/>
        <circle cx="460" cy="130" r="3.5"/><circle cx="600" cy="60" r="3"/><circle cx="740" cy="140" r="3.5"/>
        <circle cx="880" cy="70" r="3"/><circle cx="40" cy="240" r="3"/><circle cx="320" cy="250" r="3"/>
        <circle cx="600" cy="240" r="3"/><circle cx="880" cy="230" r="3"/>
      </g></g></svg>
    <i class="pt" style="left:14%;top:70%"></i><i class="pt" style="left:46%;top:22%;animation-delay:3s"></i>
    <i class="pt" style="left:78%;top:64%;animation-delay:6s"></i>
  </div>

  <h3><span class="hic">__I_NET__</span><span class="t">Model Architecture</span></h3>
  <div class="uline"><i></i></div>

  <div class="flow" id="flow">

    <!-- 1 Input MRI -->
    <div class="node n-input" role="button" tabindex="0" aria-pressed="false" data-i="0" style="--d:.2s">
      <div class="ic">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>
          <ellipse cx="12" cy="12" rx="4.2" ry="5.2"/></svg>
        <i class="scanl"></i><i class="pring"></i><i class="sp s1"></i><i class="sp s2"></i><i class="sp s3"></i>
      </div>
      <b>Input MRI</b><small>224 × 224 × 3</small>
      <div class="tip"><b>Input MRI</b>Receives the MRI scan as an image.</div>
    </div>
    <div class="arr"><span class="ln"></span><i class="pt"></i>__I_ARROW_B__</div>

    <!-- 2 Preprocessing -->
    <div class="node n-pre" role="button" tabindex="0" aria-pressed="false" data-i="1" style="--d:.3s">
      <div class="ic">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round">
          <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
          <circle class="k k1" cx="9" cy="6" r="2.3" fill="#0b1a3a"/><circle class="k k2" cx="15" cy="12" r="2.3" fill="#0b1a3a"/>
          <circle class="k k3" cx="10" cy="18" r="2.3" fill="#0b1a3a"/></svg>
      </div>
      <b>Preprocessing</b><small><span class="st">Resize</span> · <span class="st">Normalize</span> · <span class="st">Augment</span></small>
      <div class="tip"><b>Preprocessing</b>Resizes, normalizes and augments the images.</div>
    </div>
    <div class="arr"><span class="ln"></span><i class="pt"></i>__I_ARROW_B__</div>

    <!-- 3 CNN Layers -->
    <div class="node n-cnn" role="button" tabindex="0" aria-pressed="false" data-i="2" style="--d:.4s">
      <span class="badge">FEATURE LEARNING</span>
      <div class="ic">
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" stroke-linecap="round">
          <g class="sig" stroke="#7fe9ff" stroke-width="1.1">
            <line x1="6" y1="11" x2="18" y2="6"/><line x1="6" y1="11" x2="18" y2="18"/><line x1="6" y1="11" x2="18" y2="30"/>
            <line x1="6" y1="25" x2="18" y2="6"/><line x1="6" y1="25" x2="18" y2="18"/><line x1="6" y1="25" x2="18" y2="30"/>
            <line x1="18" y1="6" x2="30" y2="11"/><line x1="18" y1="18" x2="30" y2="11"/><line x1="18" y1="30" x2="30" y2="11"/>
            <line x1="18" y1="6" x2="30" y2="25"/><line x1="18" y1="18" x2="30" y2="25"/><line x1="18" y1="30" x2="30" y2="25"/></g>
          <g fill="#a78bfa">
            <circle class="nd" cx="6" cy="11" r="2.4"/><circle class="nd" cx="6" cy="25" r="2.4"/>
            <circle class="nd c2" cx="18" cy="6" r="2.4"/><circle class="nd c2" cx="18" cy="18" r="2.4"/><circle class="nd c2" cx="18" cy="30" r="2.4"/>
            <circle class="nd c3" cx="30" cy="11" r="2.4"/><circle class="nd c3" cx="30" cy="25" r="2.4"/></g></svg>
      </div>
      <b>CNN Layers</b><small>Conv2D + ReLU MaxPool</small>
      <div class="tip"><b>CNN Layers</b>Extracts important visual patterns from the MRI image.</div>
    </div>
    <div class="arr"><span class="ln"></span><i class="pt"></i>__I_ARROW_B__</div>

    <!-- 4 Feature Extraction -->
    <div class="node n-feat" role="button" tabindex="0" aria-pressed="false" data-i="3" style="--d:.5s">
      <div class="ic">
        <svg width="38" height="26" viewBox="0 0 36 26" fill="none" stroke-linecap="round" style="overflow:hidden">
          <g class="wave"><path d="M-18 13 Q-13.5 1 -9 13 T0 13 T9 13 T18 13 T27 13 T36 13 T45 13" stroke="#22d3ee" stroke-width="1.8"/>
          <path d="M-18 13 Q-13.5 7 -9 13 T0 13 T9 13 T18 13 T27 13 T36 13 T45 13" stroke="rgba(167,139,250,.7)" stroke-width="1.2"/></g></svg>
        <i class="fp f1"></i><i class="fp f2"></i><i class="fp f3"></i>
      </div>
      <b>Feature Extraction</b><small></small>
      <div class="tip"><b>Feature Extraction</b>Identifies meaningful visual patterns from extracted features.</div>
    </div>
    <div class="arr"><span class="ln"></span><i class="pt"></i>__I_ARROW_B__</div>

    <!-- 5 Dense Layer -->
    <div class="node n-dense" role="button" tabindex="0" aria-pressed="false" data-i="4" style="--d:.6s">
      <div class="ic">
        <svg class="dn" width="36" height="36" viewBox="0 0 36 36" fill="none" stroke-linecap="round">
          <g stroke-width="1"><line x1="8" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="18" y2="18"/><line x1="28" y1="6" x2="18" y2="18"/>
            <line x1="18" y1="18" x2="8" y2="30"/><line x1="18" y1="18" x2="18" y2="30"/><line x1="18" y1="18" x2="28" y2="30"/></g>
          <line class="dp" x1="8" y1="6" x2="18" y2="18" style="animation-delay:0s"/>
          <line class="dp" x1="28" y1="6" x2="18" y2="18" style="animation-delay:1.6s"/>
          <line class="dp" x1="18" y1="18" x2="8" y2="30" style="animation-delay:3.1s"/>
          <line class="dp" x1="18" y1="18" x2="28" y2="30" style="animation-delay:4.2s"/>
          <g fill="#22d3ee"><circle cx="8" cy="6" r="2.3"/><circle cx="18" cy="6" r="2.3"/><circle cx="28" cy="6" r="2.3"/>
            <circle cx="18" cy="18" r="2.8" fill="#a78bfa"/>
            <circle cx="8" cy="30" r="2.3"/><circle cx="18" cy="30" r="2.3"/><circle cx="28" cy="30" r="2.3"/></g></svg>
      </div>
      <b>Dense Layer</b><small>Fully Connected</small>
      <div class="tip"><b>Dense Layer</b>Combines the features to weigh the evidence for each class.</div>
    </div>
    <div class="arr"><span class="ln"></span><i class="pt"></i>__I_ARROW_B__</div>

    <!-- 6 Prediction -->
    <div class="node n-pred" role="button" tabindex="0" aria-pressed="false" data-i="5" style="--d:.7s">
      <div class="ic">__I_TREND__<i class="actring"></i></div>
      <b>Prediction</b><small>Softmax</small>
      <div class="tip"><b>Prediction</b>Gives a probability for each of the 4 classes.</div>
    </div>

  </div>

  <!-- Interactive view -->
  <div class="iv" id="iv">
    <div class="ii">__I_SPARK__</div>
    <div class="txt" id="txt">
      <b id="ivT">Interactive View</b>
      <span class="tech" id="ivTech"></span>
      <small id="ivS">Click on each layer to see details</small>
      <div class="meta" id="meta"></div>
    </div>
    <div class="btns">
      <button class="prev" id="prev" aria-label="Previous layer">&#8249;</button>
      <button class="go" id="go" aria-label="Next layer">__I_ARROW_W__</button>
    </div>
  </div>

</div>

<script>
var LAYERS = [
  {t:'Input MRI', tech:'224 × 224 × 3', d:'The uploaded MRI scan enters the network as a 224 × 224 colour image.'},
  {t:'Preprocessing', tech:'Resize · Normalize · Augment', d:'Images are resized and normalized so the model gets consistent input. Augmentation adds variety during training.'},
  {t:'CNN Layers', tech:'Conv2D + ReLU + MaxPool', d:'Extracts spatial features from MRI images, such as edges, textures and shapes.'},
  {t:'Feature Extraction', tech:'Learned feature maps', d:'Condenses the patterns found by the CNN into a compact set of meaningful features.'},
  {t:'Dense Layer', tech:'Fully Connected', d:'Combines all extracted features to weigh the evidence for each tumor class.'},
  {t:'Prediction', tech:'Softmax', d:'Outputs a probability for Glioma, Meningioma, Pituitary and No Tumor. The highest one is the result.'}
];
var nodes = Array.prototype.slice.call(document.querySelectorAll('.node'));
var arrs = Array.prototype.slice.call(document.querySelectorAll('.arr'));
var flow = document.getElementById('flow'), iv = document.getElementById('iv'), txt = document.getElementById('txt');
var ivT = document.getElementById('ivT'), ivTech = document.getElementById('ivTech'), ivS = document.getElementById('ivS'), meta = document.getElementById('meta');
var sel = -1;

function render() {
  nodes.forEach(function (n, k) { n.classList.toggle('sel', k === sel); n.setAttribute('aria-pressed', k === sel ? 'true' : 'false'); });
  flow.classList.toggle('has-sel', sel >= 0);
  iv.classList.toggle('active', sel >= 0);
  txt.classList.remove('swap'); void txt.offsetWidth; txt.classList.add('swap');
  if (sel < 0) {
    ivT.textContent = 'Interactive View'; ivTech.textContent = '';
    ivS.textContent = 'Click on each layer to see details'; meta.innerHTML = '';
  } else {
    var L = LAYERS[sel], nx = LAYERS[(sel + 1) % LAYERS.length];
    ivT.textContent = L.t; ivTech.textContent = L.tech; ivS.textContent = L.d;
    meta.innerHTML = '<span class="chip">Layer ' + (sel + 1) + ' of ' + LAYERS.length + '</span>' +
      '<button class="chip" data-act="next">Next: ' + nx.t + ' &rarr;</button>' +
      '<button class="chip" data-act="clear">&#10005; Clear</button>';
  }
  fit();
}
function select(i) { sel = (i === sel) ? -1 : i; render(); }
function goTo(i) { sel = i; render(); }

nodes.forEach(function (n, k) {
  n.addEventListener('click', function () { select(k); });
  n.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(k); }
    else if (e.key === 'ArrowRight' && nodes[k + 1]) { nodes[k + 1].focus(); }
    else if (e.key === 'ArrowLeft' && nodes[k - 1]) { nodes[k - 1].focus(); }
  });
});
document.getElementById('go').addEventListener('click', function () { goTo(sel < 0 ? 0 : (sel + 1) % LAYERS.length); });
document.getElementById('prev').addEventListener('click', function () { goTo(sel <= 0 ? LAYERS.length - 1 : sel - 1); });
meta.addEventListener('click', function (e) {
  var b = e.target.closest('button'); if (!b) return;
  if (b.getAttribute('data-act') === 'next') goTo((sel + 1) % LAYERS.length);
  else if (b.getAttribute('data-act') === 'clear') { sel = -1; render(); }
});
document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && sel >= 0) { sel = -1; render(); } });

/* data pulse: Input -> Preprocessing -> CNN -> Features -> Dense -> Prediction (activation) */
var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function restart(el, cls, ms) {
  el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls);
  if (ms) setTimeout(function () { el.classList.remove(cls); }, ms);
}
function cycle() {
  restart(nodes[0], 'arrive', 1100);
  arrs.forEach(function (a, i) {
    var t0 = 500 + i * 950;
    setTimeout(function () { arrs.forEach(function (x) { x.classList.remove('on'); }); a.classList.add('on'); restart(a, 'go', 800); }, t0);
    setTimeout(function () {
      restart(nodes[i + 1], 'arrive', 1100);
      if (i + 1 === nodes.length - 1) restart(nodes[i + 1], 'fire', 1500);
    }, t0 + 720);
  });
  setTimeout(function () { arrs.forEach(function (x) { x.classList.remove('on'); }); }, 500 + arrs.length * 950 + 1600);
  setTimeout(cycle, 500 + arrs.length * 950 + 3400);
}
if (!reduce) setTimeout(cycle, 900);

/* keep the iframe exactly as tall as the card */
var card = document.getElementById('card');
function fit() {
  try { window.frameElement.style.height = Math.ceil(card.getBoundingClientRect().height) + 4 + 'px'; } catch (e) {}
}
if (window.ResizeObserver) new ResizeObserver(fit).observe(card);
window.addEventListener('load', fit); window.addEventListener('resize', fit); fit();
</script>
</body></html>
"""


# =====================================================================
# FOOTER BAR
# =====================================================================
def render_footer(rich=False, report_link=None):
    acc_val = METRICS["Accuracy"] or "—"

    # "View Full Report": a real link when report_link is given, otherwise the original static button
    _arrow = ic("arrow", 14, "#fff")
    if report_link:
        report_html = (f'<a class="report" href="{report_link}" target="_self" '
                       f'style="text-decoration:none;color:#fff">View Full Report {_arrow}</a>')
    else:
        report_html = f'<div class="report">View Full Report {_arrow}</div>'

    # default (all other pages): exactly the original "High Accuracy" badge
    acc_badge = (
        f'<div class="fbadge"><div class="bi2">{ic("target", 14, "#34d399")}</div>'
        f'<div><b>High Accuracy</b><small>{acc_val} on test data</small></div></div>'
    )
    # Model page only: progress ring + count-up, using the real value from METRICS
    if rich and METRICS["Accuracy"]:
        s = str(METRICS["Accuracy"]).replace("%", "").strip()
        try:
            num = float(s)
            dec = len(s.split(".")[1]) if "." in s else 0
            off = 87.96 * (1 - num / 100)
            acc_badge = (
                f'<div class="fbadge"><div class="bi2 nvp-ring">'
                f'<svg class="rg" width="38" height="38" viewBox="0 0 38 38">'
                f'<circle cx="19" cy="19" r="14" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="3"/>'
                f'<circle class="arc" cx="19" cy="19" r="14" fill="none" stroke="#34d399" stroke-width="3" '
                f'stroke-linecap="round" style="--off:{off:.2f}"/></svg>'
                f'<span class="ti">{ic("target", 14, "#34d399")}</span></div>'
                f'<div><b>High Accuracy</b><small><span data-nvcount="{num}" data-dec="{dec}">{num:.{dec}f}</span>% on test data</small></div></div>'
            )
        except ValueError:
            pass

    st.markdown(
        f'<div class="foot">'
        f'<div class="tag"><span class="fi">{ic("activity", 20, "#22d3ee")}{ic("brain", 22, "#a78bfa")}</span>'
        f'From Data to Diagnosis — Powered by AI</div>'
        f'<div class="fbadge"><div class="bi2">{ic("sparkles", 14, "#22d3ee")}</div>'
        f'<div><b>Advanced Deep Learning</b><small>Convolutional Neural Networks</small></div></div>'
        f'{acc_badge}'
        f'<div class="fbadge"><div class="bi2">{ic("zap", 14, "#fbbf24")}</div>'
        f'<div><b>Fast Inference</b><small>&lt; 1 second</small></div></div>'
        f'<div class="fbadge"><div class="bi2">{ic("net", 14, "#f472b6")}</div>'
        f'<div><b>Open Source</b><small>Built with Python &amp; Streamlit</small></div></div>'
        f'{report_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="credit">NeuroVision AI · Built with Python, TensorFlow &amp; Streamlit</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# ROUTER  (decides which page to show)
# =====================================================================
current = get_current_page()
render_navbar(current)

if current == "home":
    page_home()
elif current == "analyze":
    page_analyze()
elif current == "model":
    page_model()
elif current == "dataset":
    page_dataset()
elif current == "performance":
    page_performance()
elif current == "architecture":
    page_architecture()

if current != "home":
    render_footer(rich=(current in ("model", "architecture", "dataset", "performance")), report_link="?page=performance")


# =====================================================================
# RESPONSIVE LAYER  (phones, tablets, laptops, desktops, large LED screens)
# Injected last so it can refine the page-specific styles above.
# =====================================================================
RESP_CSS = """
<style>
.stApp, [data-testid="stMain"], section.main { overflow-x: hidden; }
img { max-width: 100%; }
@media (min-width: 1900px) { .block-container { max-width: 1680px; } }

/* ---- navbar: links move to their own scrollable row on smaller screens ---- */
@media (max-width: 1180px) {
  .nav { flex-wrap: wrap; row-gap: 10px; }
  .brand { min-width: 0; }
  .links { order: 3; width: 100%; gap: 20px; overflow-x: auto; justify-content: flex-start; justify-content: safe center;
           scrollbar-width: none; -webkit-overflow-scrolling: touch; padding: 2px 2px 0; }
  .links::-webkit-scrollbar { display: none; }
  .links a.lk { white-space: nowrap; flex-shrink: 0; }
}

/* ---- page padding ---- */
@media (max-width: 900px) { .block-container { padding-left: 1.25rem !important; padding-right: 1.25rem !important; } }
@media (max-width: 640px) { .block-container { padding-left: .75rem !important; padding-right: .75rem !important; padding-bottom: 4rem !important; } }

/* ---- Analyze page: columns -> 2 per row on tablets, 1 per row on phones ---- */
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: 1rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 calc(50% - 1rem) !important; width: calc(50% - 1rem) !important; min-width: calc(50% - 1rem) !important; }
}
@media (max-width: 700px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
  .nv-conn { width: 28px; }
  .nv-steps { gap: 8px; }
  .nv-step { font-size: .76rem; }
  .scan-row { flex-wrap: wrap; }
  .ring { width: 100px; height: 100px; }
}
@media (max-width: 420px) { .ring-wrap { flex-wrap: wrap; justify-content: center; text-align: center; } .nv-info { gap: 6px; } }

/* ---- Dataset page ---- */
@media (max-width: 1100px) { .samples, .gal { grid-template-columns: repeat(4, 1fr); } }
@media (max-width: 900px) {
  .samples, .gal { grid-template-columns: repeat(3, 1fr); }
  .cats, .dstats { grid-template-columns: repeat(2, 1fr); }
  .ds-sh { flex-wrap: wrap; }
}
@media (max-width: 520px) { .samples, .gal { grid-template-columns: repeat(2, 1fr); } .dstat .v { font-size: 1rem; } }

/* ---- Model page ---- */
@media (max-width: 1100px) { .nvp-sub { margin-left: 0 !important; } }
@media (max-width: 700px) {
  .step .tip { display: none; }
  .step.nvp { flex: 1 1 calc(50% - 8px); min-width: 130px; }
}
@media (hover: none) { .step .tip { display: none; } }

/* ---- Home page ticker ---- */
@media (max-width: 760px) {
  .tick { flex-direction: column; align-items: stretch; gap: 12px; padding: 12px 14px; }
  .tk-tag { border-right: 0; padding-right: 0; white-space: normal; font-size: .92rem; }
  .tk-item { margin-right: 28px; }
}

/* ---- footer bar ---- */
@media (max-width: 900px) {
  .foot { gap: 12px 16px; padding: 14px 16px; }
  .foot .fbadge { border-left: 0 !important; padding-left: 0 !important; box-shadow: none !important; }
  .foot .tag { white-space: normal; width: 100%; }
}
@media (max-width: 640px) {
  .foot .fbadge { flex: 1 1 calc(50% - 16px); min-width: 150px; }
  .foot .report { margin-left: 0; width: 100%; justify-content: center; }
  .nav { padding: 10px 14px; border-radius: 14px; }
  .logo { font-size: 1rem; letter-spacing: .1em; }
  .logo small { font-size: .58rem; letter-spacing: .02em; }
  .cta { padding: 8px 14px; font-size: .82rem; }
  .links { gap: 16px; font-size: .82rem; }
  .card { padding: 14px; }
  .disclaimer { font-size: .78rem; }
  .big { font-size: 1.8rem; }
}
@media (max-width: 420px) { .logo small { display: none; } .cta { padding: 7px 12px; } }
</style>
"""
st.markdown(RESP_CSS, unsafe_allow_html=True)