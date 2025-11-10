# app.py
import io
import unicodedata
from typing import Dict

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---- Opcional: si usas torch.cuda, puedes setear device = "cuda" ----
import torch

# ------------------------------
# CONFIG GENERAL
# ------------------------------
st.set_page_config(page_title="MediScan AI", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; color: #343a40; }
        .block-container { max-width: 820px; }
        h1, h2, h3 { color: #007bff; text-align: center; font-weight: 600; }
        [data-testid="stImage"] img { border-radius: 8px; border: 1px solid #e9ecef; }
        .metric-card{
            border:1px solid #e9ecef; border-radius:12px; padding:16px; background:#fff;
            box-shadow:0 1px 4px rgba(0,0,0,.06); text-align:center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>MediScan AI</h1>", unsafe_allow_html=True)
st.caption("Sube una imagen: 1) CLIP valida que sea ecografía → 2) YOLO identifica el tipo → 3) YOLO específico clasifica la categoría.")

# ==============================
# CLIP: verificador de ECOGRAFÍA
# ==============================
try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception as e:
    st.error(f"No se pudo importar CLIP. Instala con: pip install git+https://github.com/openai/CLIP.git\nDetalle: {e}")
    st.stop()

DEVICE = "cpu"  # cambia a "cuda" si tienes GPU y torch reconoce cuda

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

clip_model, clip_preprocess = load_clip_model()

ULTRASOUND_TEXTS = [
    "an ultrasound image",
    "a medical ultrasound scan",
    "a liver ultrasound",
    "a kidney ultrasound",
    "a breast ultrasound",
    "a medical imaging scan",
    # negativos para contraste
    "a normal photo",
    "a cat",
    "a dog",
    "a landscape",
]

@st.cache_data
def _clip_tokens():
    return clip.tokenize(ULTRASOUND_TEXTS).to(DEVICE)

def is_ultrasound_image(pil_img: Image.Image, threshold: float = 0.45) -> bool:
    """
    Retorna True si CLIP considera que la imagen se parece más a prompts de ecografía.
    Umbral simple sobre la probabilidad softmax combinada de los prompts 'ultrasound'.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    image_input = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    text_tokens = _clip_tokens()

    with torch.no_grad():
        img_feat = clip_model.encode_image(image_input)
        txt_feat = clip_model.encode_text(text_tokens)
        # similitudes normalizadas → softmax
        probs = (img_feat @ txt_feat.T).softmax(dim=-1).squeeze(0)  # [N_TEXT]

    # pesos: suma de probabilidades de los prompts pro-ultrasonido (índices 0..5)
    prob_ultra = float(probs[:6].sum().item())
    return prob_ultra >= threshold

# ==============================
# YOLO: identificación + modelos
# ==============================
TYPE_MODEL_PATH = "modelo_identificacion_eco_best.pt"
ORGAN_MODELS: Dict[str, str] = {
    "higado": "best_fibrosis_y11s.pt",
    "rinon": "kidney_normal_stone_best.pt",  # rinon/riñon
    "mamaria": "mamarias_best.pt",
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()

@st.cache_resource
def load_yolo(path: str) -> YOLO:
    return YOLO(path)

def predict_top1(model: YOLO, pil_image: Image.Image):
    res = model(pil_image)[0]
    if not hasattr(res, "probs") or res.probs is None:
        raise RuntimeError("El modelo no devolvió probabilidades de clasificación (probs).")
    idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    name = model.names[idx] if hasattr(model, "names") else str(idx)
    return idx, name, conf

def map_type_name(raw_name: str) -> str:
    n = _strip_accents(raw_name)
    if any(k in n for k in ["higado", "liver"]): return "higado"
    if any(k in n for k in ["rinon", "riñon", "kidney"]): return "rinon"
    if any(k in n for k in ["mamaria", "mama", "breast"]): return "mamaria"
    return n  # desconocido tal cual (sin acentos)

# Carga modelos cacheados
try:
    type_model = load_yolo(TYPE_MODEL_PATH)
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo de identificación: {TYPE_MODEL_PATH}\nDetalle: {e}")
    st.stop()

organ_loaded: Dict[str, YOLO] = {}
for k, p in ORGAN_MODELS.items():
    try:
        organ_loaded[k] = load_yolo(p)
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo '{k}': {p}\nDetalle: {e}")

# ==============================
# UI
# ==============================
st.markdown("### Cargar imagen")
uploaded = st.file_uploader("Formatos: JPG, PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    st.image(img, caption="Vista previa", use_container_width=True)

    if st.button("Analizar imagen", type="primary"):
        with st.spinner("Verificando con CLIP si es ecografía…"):
            if not is_ultrasound_image(img):
                st.warning("⚠️ La imagen no parece ser una **ecografía** según CLIP. Sube una ecografía válida.")
                st.stop()

        with st.spinner("Identificando tipo de ecografía…"):
            try:
                _, raw_type, type_conf = predict_top1(type_model, img)
                organ_key = map_type_name(raw_type)
            except Exception as e:
                st.error(f"Error al identificar tipo de ecografía: {e}")
                st.stop()

        mdl = organ_loaded.get(organ_key)
        if mdl is None:
            st.error(
                f"No hay modelo específico cargado para el tipo detectado: **{organ_key}** "
                f"(clase original: '{raw_type}'). Revisa nombres y rutas."
            )
            st.stop()

        with st.spinner("Clasificando categoría específica…"):
            try:
                _, diag_name, diag_conf = predict_top1(mdl, img)
            except Exception as e:
                st.error(f"Error al clasificar con el modelo específico ({organ_key}): {e}")
                st.stop()

        # Resultado
        st.markdown("## Resultado")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="metric-card"><div>Tipo de ecografía</div>'
                f'<h2 style="margin:6px 0">{organ_key.upper()}</h2>'
                f'<div>Confianza (tipo): {type_conf*100:.1f}%</div></div>', unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                '<div class="metric-card"><div>Categoría / diagnóstico</div>'
                f'<h2 style="margin:6px 0">{diag_name}</h2>'
                f'<div>Confianza (categoría): {diag_conf*100:.1f}%</div></div>', unsafe_allow_html=True
            )

        st.markdown("---")
        if organ_key == "higado":
            st.caption("Nota hígado: si tu modelo usa F0–F4, valores mayores indican fibrosis más avanzada.")
        elif organ_key == "rinon":
            st.caption("Nota riñón: categorías típicas entrenadas como *normal* / *cálculo (stone)*.")
        elif organ_key == "mamaria":
            st.caption("Nota mamaria: categorías según tu entrenamiento (p. ej., benigno/maligno).")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Apoyo con IA. No reemplaza el criterio médico profesional.")
