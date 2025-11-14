# app.py (Streamlit)
import io
import unicodedata
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch

# ------------------------------
# CONFIG GENERAL
# ------------------------------
st.set_page_config(page_title="MediScan AI", page_icon="MS", layout="centered")

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #f9fbff 0%, #f4f6fb 100%); color: #2f3640; }
        .block-container { max-width: 900px; padding-top: 1.2rem; }
        h1, h2, h3 { color: #5b8def; text-align: center; font-weight: 700; }
        [data-testid="stImage"] img { border-radius: 10px; border: 1px solid #e9ecef; box-shadow: 0 8px 24px rgba(0,0,0,.06); max-height: 420px; object-fit: contain; }
        .metric-card{ border:1px solid #d6e4ff; border-radius:14px; padding:18px 18px 14px; background:linear-gradient(145deg,#ffffff 0%,#f0f4ff 100%); box-shadow:0 4px 18px rgba(0,0,0,.07); text-align:center; }
        .metric-card h2{ color:#343a7a; font-weight:700; }
        .stButton>button { background: linear-gradient(90deg, #5b8def 0%, #8a5bff 100%); color:#fff; border:0; padding:.6rem 1rem; border-radius:10px; box-shadow:0 6px 14px rgba(91,141,239,.35);} .stButton>button:hover{filter:brightness(1.03); transform: translateY(-1px);} </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>MediScan AI</h1>", unsafe_allow_html=True)

# ==============================
# CLIP: verificador de ECOGRAFIA / NO-ECOGRAFIA + grupos
# ==============================
try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception as e:
    st.error(
        "No se pudo importar CLIP. Instala con: pip install git+https://github.com/openai/CLIP.git\n"
        f"Detalle: {e}"
    )
    st.stop()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess

clip_model, clip_preprocess = load_clip_model()

# Prompts por grupo (ASCII-only para evitar problemas de encoding)
ULTRASOUND_POS_PROMPTS: List[str] = [
    # ES
    "una ecografia",
    "una imagen de ecografia",
    "una ecografia medica",
    "ecografia doppler",
    "ecografia abdominal",
    "ecografia hepatica",
    "ecografia de higado",
    "ecografia renal",
    "ecografia de rinon",
    "ecografia mamaria",
    "ultrasonido medico",
    "ultrasonografia",
    # EN
    "an ultrasound image",
    "a medical ultrasound scan",
    "a doppler ultrasound",
    "an abdominal ultrasound",
    "a liver ultrasound",
    "a kidney ultrasound",
    "a breast ultrasound",
]

MEDICAL_OTHER_PROMPTS: List[str] = [
    # EN
    "a chest x-ray",
    "an x-ray image",
    "a radiography image",
    "a mammography image",
    "a CT scan",
    "a computed tomography scan",
    "an MRI scan",
    "a magnetic resonance image",
    "a PET scan",
    "an endoscopy image",
    "a colonoscopy image",
    "a histology microscopy slide",
    "a dermoscopy image",
    # ES
    "una radiografia",
    "una mamografia",
    "una tomografia computarizada",
    "una resonancia magnetica",
    "una endoscopia",
    "una colonoscopia",
    "una micrografia de histologia",
    "una dermatoscopia",
]

ANIMALS_PROMPTS: List[str] = [
    # EN animals
    "a cat",
    "a dog",
    "a bird",
    "a horse",
    "a cow",
    "a sheep",
    "a goat",
    "a pig",
    "a lion",
    "a tiger",
    "a bear",
    "a monkey",
    "an animal photo",
    "a wildlife photo",
    "a pet photo",
    # ES animals
    "un gato",
    "un perro",
    "un pajaro",
    "un caballo",
    "una vaca",
    "una oveja",
    "una cabra",
    "un cerdo",
    "un leon",
    "un tigre",
    "un oso",
    "un mono",
    "foto de un animal",
    "foto de una mascota",
]

PEOPLE_PROMPTS: List[str] = [
    # EN
    "a photograph of a person",
    "a portrait photo",
    "a selfie",
    "a face photo",
    # ES
    "una persona",
    "un retrato",
    "una selfie",
    "una foto de rostro",
]

GRAPHICS_PROMPTS: List[str] = [
    # EN
    "a diagram",
    "a chart",
    "a bar chart",
    "a line plot",
    "an infographic",
    "a screenshot",
    "a document scan",
    "an illustration",
    "a drawing",
    "a painting",
    # ES
    "un diagrama",
    "un grafico",
    "una grafica",
    "un plot",
    "una infografia",
    "una captura de pantalla",
    "un documento escaneado",
    "una ilustracion",
    "un dibujo",
    "una pintura",
]

SCENERY_PROMPTS: List[str] = [
    # EN
    "a landscape photo",
    "a cityscape",
    "a street photo",
    "a beach",
    "a mountain",
    "a nature scene",
    # ES
    "un paisaje",
    "una ciudad",
    "una calle",
    "una playa",
    "una montana",
    "una escena de naturaleza",
]

CLIP_GROUPS: Dict[str, List[str]] = {
    "ecografia": ULTRASOUND_POS_PROMPTS,
    "animales": ANIMALS_PROMPTS,
    "personas": PEOPLE_PROMPTS,
    "graficos": GRAPHICS_PROMPTS,
    "otras_medicas": MEDICAL_OTHER_PROMPTS,
    "escenas": SCENERY_PROMPTS,
}

@st.cache_resource
def _clip_text_features_groups() -> Tuple[torch.Tensor, List[str], List[Tuple[str, int, int]]]:
    """Flatten prompts and return (features, labels, group_spans)."""
    labels: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    start = 0
    for gname, prompts in CLIP_GROUPS.items():
        labels.extend(prompts)
        end = start + len(prompts)
        spans.append((gname, start, end))
        start = end
    tokens = clip.tokenize(labels).to(DEVICE)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt, labels, spans

@torch.no_grad()
def clip_groups(pil_img: Image.Image):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    text_features, labels, spans = _clip_text_features_groups()
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    img = clip_model.encode_image(image_input)
    img = img / img.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * img @ text_features.T
    probs = logits.softmax(dim=-1).squeeze(0)

    group_probs: Dict[str, float] = {}
    for gname, s, e in spans:
        group_probs[gname] = float(probs[s:e].sum().item())

    pos_prob = group_probs.get("ecografia", 0.0)
    neg_prob = 1.0 - pos_prob

    # Top-5 etiquetas crudas
    topk = torch.topk(probs, k=min(5, probs.shape[0]))
    top = [(labels[i], float(probs[i].item())) for i in topk.indices.tolist()]
    return {"pos_prob": pos_prob, "neg_prob": neg_prob, "groups": group_probs, "top": top}


def is_ultrasound_image(pil_img: Image.Image, pos_threshold: float = 0.58, margin: float = 0.08):
    """Devuelve (es_ecografia: bool, info: dict)."""
    info = clip_groups(pil_img)
    es_eco = (info["pos_prob"] >= pos_threshold) and (info["pos_prob"] - info["neg_prob"] >= margin)
    return es_eco, info

# ==============================
# YOLO: identificacion + modelos
# ==============================
TYPE_MODEL_PATH = "modelo_identificacion_eco_best.pt"
ORGAN_MODELS: Dict[str, str] = {
    "higado": "best_fibrosis_y11s.pt",
    "rinon": "kidney_normal_stone_best.pt",
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
        raise RuntimeError("El modelo no devolvio probabilidades de clasificacion (probs).")
    idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    name = model.names[idx] if hasattr(model, "names") else str(idx)
    return idx, name, conf


def map_type_name(raw_name: str) -> str:
    n = _strip_accents(raw_name)
    if any(k in n for k in ["higado", "liver"]):
        return "higado"
    if any(k in n for k in ["rinon", "rinon", "kidney"]):
        return "rinon"
    if any(k in n for k in ["mamaria", "mama", "breast"]):
        return "mamaria"
    return n

# Carga modelos cacheados
try:
    type_model = load_yolo(TYPE_MODEL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo de identificacion: {TYPE_MODEL_PATH}\nDetalle: {e}")
    st.stop()

organ_loaded: Dict[str, YOLO] = {}
for k, p in ORGAN_MODELS.items():
    try:
        organ_loaded[k] = load_yolo(p)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo '{k}': {p}\nDetalle: {e}")

# ==============================
# UI
# ==============================
st.markdown("### Cargar imagen")
uploaded = st.file_uploader("Selecciona una imagen", type=None)

if uploaded:
    try:
        raw = uploaded.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        st.error("El archivo seleccionado no parece ser una imagen compatible. Prueba con JPG/PNG/BMP/TIFF/WEBP.")
        st.stop()
    st.image(img, caption="Vista previa", use_container_width=False, width=420)

    with st.expander("Ajustes CLIP (avanzado)"):
        c1, c2 = st.columns(2)
        with c1:
            pos_thr = st.slider(
                "Umbral ecografia",
                0.3, 0.95, 0.58, 0.01,
                help="Valor minimo de probabilidad para aceptar que la imagen es una ecografia. Si rechaza ecografias validas, baja un poco este valor."
            )
        with c2:
            margin = st.slider(
                "Margen vs no-eco",
                0.0, 0.3, 0.08, 0.01,
                help="Diferencia minima entre 'Prob. ecografia' y 'Prob. no-ecografia' para evitar casos ambiguos. Si acepta imagenes que no son ecografias, sube este margen."
            )
        st.markdown(
            "- Si CLIP rechaza ecografias validas: baja el umbral (p.ej., 0.58 -> 0.54) o reduce el margen (0.08 -> 0.05).\n"
            "- Si CLIP acepta imagenes que no son ecografias: sube el umbral (0.62-0.68) y/o aumenta el margen (0.10-0.15).\n"
            "- Valores recomendados iniciales: umbral 0.58 y margen 0.08."
        )

    if st.button("Analizar imagen", type="primary"):
        with st.spinner("Verificando con CLIP si es ecografia..."):
            es_eco, info = is_ultrasound_image(img, pos_threshold=pos_thr, margin=margin)

        # Tarjetas principales
        st.markdown("### Verificacion CLIP")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="metric-card"><div>Prob. ecografia</div>'
                f'<h2 style="margin:6px 0">{info["pos_prob"]*100:.1f}%</h2>'
                f'<div>Umbral: {pos_thr*100:.0f}%</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="metric-card"><div>Prob. no-ecografia</div>'
                f'<h2 style="margin:6px 0">{info["neg_prob"]*100:.1f}%</h2>'
                f'<div>Margen requerido: {margin*100:.0f}%</div></div>',
                unsafe_allow_html=True,
            )


        # Mejores coincidencias CLIP ocultas por requerimiento

        if not es_eco:
            st.warning("La imagen NO parece ser una ecografia segun CLIP.")
            force = st.checkbox("Forzar analisis con YOLO igualmente (bajo su responsabilidad)")
            if not force:
                st.stop()

        with st.spinner("Identificando tipo de ecografia..."):
            try:
                _, raw_type, type_conf = predict_top1(type_model, img)
                organ_key = map_type_name(raw_type)
            except Exception as e:
                st.error(f"Error al identificar tipo de ecografia: {e}")
                st.stop()

        mdl = organ_loaded.get(organ_key)
        if mdl is None:
            st.error(
                f"No hay modelo especifico cargado para el tipo detectado: {organ_key} "
                f"(clase original: '{raw_type}'). Revisa nombres y rutas."
            )
            st.stop()

        with st.spinner("Clasificando categoria especifica..."):
            try:
                _, diag_name, diag_conf = predict_top1(mdl, img)
            except Exception as e:
                st.error(f"Error al clasificar con el modelo especifico ({organ_key}): {e}")
                st.stop()

        # Resultado
        st.markdown("## Resultado")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="metric-card"><div>Tipo de ecografia</div>'
                f'<h2 style="margin:6px 0">{organ_key.upper()}</h2>'
                f'<div>Confianza (tipo): {type_conf*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="metric-card"><div>Categoria / diagnostico</div>'
                f'<h2 style="margin:6px 0">{diag_name}</h2>'
                f'<div>Confianza (categoria): {diag_conf*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Apoyo con IA. No reemplaza el criterio medico profesional.")
