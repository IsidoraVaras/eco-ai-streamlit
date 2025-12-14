# MediScan AI - Guía rápida

## Requisitos
- Python 3.10+ (recomendado)
- Pip reciente
- Modelos `.pt` en la carpeta del proyecto:
  - `modelo_identificacion_eco_best.pt` (tipo de eco: mamaria/higado/rinon)
  - `best_fibrosis_y11s.pt` (higado)
  - `kidney_normal_stone_best.pt` (riñon)
  - `clasificacion_mama.pt` (mamaria, clasificacion)
  - `segmentacion_mama.pt` (mamaria, segmentacion)

## Instalacion (entorno virtual y dependencias)
```bash
# 1) Crear y activar entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3) Instalar CLIP (repositorio de OpenAI)
pip install git+https://github.com/openai/CLIP.git
```

## Ejecucion
```bash
streamlit run app.py
```

## Archivos y rol
- `app.py`: aplicación Streamlit. Flujo:
  1) Carga de imagen y vista previa.
  2) CLIP verifica si parece ecografáa 
  3) `modelo_identificacion_eco_best.pt` detecta órgano (mamaria/higado/rinon).
  4) Modelo específico clasifica:
     - Hígado: `best_fibrosis_y11s.pt`
     - Riñón: `kidney_normal_stone_best.pt`
     - Mama: `clasificacion_mama.pt`
  5) Mama: si `segmentacion_mama.pt` existe, genera máscara y la superpone; si no hay máscara, muestra la imagen y un aviso de “sin lesiones”.
  6) Muestra tarjetas con probabilidades y confianza.

## Notas de uso
- CLIP: si rechaza ecografáas validas, baja umbral/margen; si acepta no-ecografáas, subelos. Valores iniciales: umbral 0.58, margen 0.08.
- Segmentación mamaria: si el modelo no devuelve máscaras, se muestra “Sin lesiones detectadas” debajo de la imagen y se trata como normal.

## Estructura esperada 
```
.
├─ app.py
├─ modelo_identificacion_eco_best.pt
├─ best_fibrosis_y11s.pt
├─ kidney_normal_stone_best.pt
├─ clasificacion_mama.pt
└─ segmentacion_mama.pt
```


