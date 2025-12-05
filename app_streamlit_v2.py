"""
Netflix AI - Demo de Recomendación con NLP
==========================================
Versión simplificada y modular.

Ejecutar:
    streamlit run app_streamlit.py
"""
import streamlit as st
import os
import sys

# Configurar página PRIMERO
st.set_page_config(
    page_title="Netflix AI - Recomendador NLP",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Asegurar imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CSS_STYLES, CSV_PATH, RECOMMENDER_MODELS, CLASSIFIER_MODELS
from src.model_persistence import load_model, load_benchmark_results
from src.data_loader import DataLoader
from src.ui_components import inject_css, render_header, render_sidebar

# Importar tabs
from src.tabs import (
    render_tab_recommender,
    render_tab_search,
    render_tab_classifier,
    render_tab_evaluation,
    render_tab_info
)

# =============================================================================
# CARGA DE DATOS Y MODELOS (con cache)
# =============================================================================

@st.cache_resource
def load_all_models():
    """Carga todos los modelos disponibles."""
    models = {"recommenders": {}, "classifiers": {}}
    
    # Recomendadores
    for name, config in RECOMMENDER_MODELS.items():
        model = load_model(config["key"])
        if model:
            models["recommenders"][name] = model
    
    # Clasificadores
    for name, config in CLASSIFIER_MODELS.items():
        model = load_model(config["key"])
        if model:
            models["classifiers"][name] = model
    
    return models

@st.cache_resource
def load_data():
    """Carga los datos del dataset."""
    csv_path = os.path.join("data", "netflix_titles.csv")
    if os.path.exists(csv_path):
        loader = DataLoader(csv_path)
        return loader.get_processed_data()
    return None

@st.cache_data
def load_benchmark():
    """Carga datos del benchmark."""
    return load_benchmark_results() or {}

# Cargar todo
models = load_all_models()
data = load_data()
benchmark = load_benchmark()
titles = data['title'].tolist() if data is not None else []

# =============================================================================
# ESTILOS Y LAYOUT
# =============================================================================

inject_css()
render_header()
render_sidebar(models, len(titles))

# =============================================================================
# TABS PRINCIPALES (SIMPLIFICADOS)
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Recomendador",
    "🔎 Búsqueda Semántica",
    "🏷️ Clasificador",
    "📊 Evaluación",
    "📖 Info"
])

with tab1:
    render_tab_recommender(models, titles, data)

with tab2:
    render_tab_search(models, data)

with tab3:
    render_tab_classifier(models)

with tab4:
    render_tab_evaluation(models, data, benchmark)

with tab5:
    render_tab_info()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 0.5rem;">
    <p style="margin: 0;">Netflix AI - Sistema de Recomendación con NLP</p>
    <p style="font-size: 0.75rem; margin: 0.2rem 0 0 0;">TF-IDF • Doc2Vec • SBERT</p>
</div>
""", unsafe_allow_html=True)
