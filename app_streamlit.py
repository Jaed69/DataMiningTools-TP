"""
Netflix AI - Demo de Exposición (Streamlit)
=============================================
Versión estable para presentaciones.
Usa modelos PRE-ENTRENADOS y datos PRE-CALCULADOS.

Ejecutar:
    streamlit run app_streamlit.py
"""
import streamlit as st
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configurar página PRIMERO (debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="Netflix AI - Recomendador NLP",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Asegurar imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_persistence import (
    load_model, load_benchmark_results, model_exists, MODELS_DIR
)
from src.data_loader import DataLoader

# =============================================================================
# CONFIGURACIÓN Y CARGA (con cache de Streamlit)
# =============================================================================
CSV_PATH = os.path.join("data", "netflix_titles.csv")

@st.cache_resource
def load_all_models():
    """Carga modelos una sola vez y los mantiene en cache."""
    models = {"recommenders": {}, "classifiers": {}}
    
    # Recomendadores
    for name, key in [("TF-IDF", "tfidf_recommender"), 
                       ("Doc2Vec", "doc2vec_recommender"), 
                       ("SBERT", "sbert_recommender")]:
        model = load_model(key)
        if model:
            models["recommenders"][name] = model
    
    # Clasificadores
    for name, key in [("Logistic", "logistic_classifier"),
                       ("NaiveBayes", "naive_bayes_classifier"),
                       ("RandomForest", "random_forest_classifier")]:
        model = load_model(key)
        if model:
            models["classifiers"][name] = model
    
    return models

@st.cache_resource
def load_data():
    """Carga datos una sola vez."""
    if os.path.exists(CSV_PATH):
        loader = DataLoader(CSV_PATH)
        return loader.get_processed_data()
    return None

@st.cache_data
def load_benchmark():
    """Carga benchmark pre-calculado."""
    return load_benchmark_results() or {}

# Cargar todo
models = load_all_models()
data = load_data()
benchmark = load_benchmark()
titles = data['title'].tolist() if data is not None else []

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
<style>
    /* Fondo oscuro estilo Netflix */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%);
    }
    
    /* Header */
    .netflix-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #333;
        margin-bottom: 2rem;
    }
    
    .netflix-title {
        color: #e50914;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .netflix-subtitle {
        color: #b3b3b3;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Cards de recomendación */
    .rec-card {
        background: linear-gradient(145deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-left: 4px solid #e50914;
        transition: transform 0.2s;
    }
    
    .rec-card:hover {
        transform: translateX(5px);
    }
    
    .rec-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .rec-score {
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    
    .rec-genre {
        color: #8c8c8c;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* Metric cards */
    .metric-box {
        background: #252525;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e50914;
    }
    
    .metric-label {
        color: #8c8c8c;
        font-size: 0.95rem;
    }
    
    /* Algorithm cards */
    .algo-card {
        background: #1f1f1f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .algo-card h3 {
        margin-top: 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #141414;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a1a;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8c8c8c;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e50914 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="netflix-header">
    <h1 class="netflix-title">NETFLIX AI</h1>
    <p class="netflix-subtitle">Sistema Multi-Modelo de Recomendación con NLP</p>
    <p style="color: #666; font-size: 0.9rem;">🚀 Demo con modelos PRE-ENTRENADOS</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎛️ Panel de Control")
    st.markdown("---")
    
    # Estado de modelos
    st.markdown("### 📊 Modelos Disponibles")
    
    st.markdown("**Recomendadores:**")
    for name in ["TF-IDF", "Doc2Vec", "SBERT"]:
        status = "✅" if name in models["recommenders"] else "❌"
        st.markdown(f"{status} {name}")
    
    st.markdown("**Clasificadores:**")
    for name in ["Logistic", "NaiveBayes", "RandomForest"]:
        status = "✅" if name in models["classifiers"] else "❌"
        st.markdown(f"{status} {name}")
    
    st.markdown("---")
    st.markdown(f"📁 **{len(titles):,}** títulos en catálogo")
    
    if not models["recommenders"]:
        st.warning("⚠️ Ejecuta `python train_models.py` primero")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎬 Recomendador", 
    "🔍 Explicación",
    "🏷️ Clasificador",
    "📈 Evaluación",
    "⏱️ Benchmark", 
    "📖 ¿Cómo Funciona?",
    "📊 Métricas"
])

# -----------------------------------------------------------------------------
# TAB 1: RECOMENDADOR
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🎬 Sistema de Recomendación")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Selector de película
        selected_movie = st.selectbox(
            "Selecciona una película o serie:",
            options=[""] + titles,
            index=0,
            help="Escribe para buscar en el catálogo"
        )
        
        # Selector de modelo
        available_recs = list(models["recommenders"].keys())
        selected_model = st.selectbox(
            "Modelo de NLP:",
            options=available_recs if available_recs else ["No hay modelos"],
            help="Elige el algoritmo de recomendación"
        )
        
        # Botón
        get_recs = st.button("🔍 Obtener Recomendaciones", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Recomendaciones")
        
        if get_recs and selected_movie and selected_model in models["recommenders"]:
            with st.spinner("Buscando..."):
                try:
                    model = models["recommenders"][selected_model]
                    recs = model.recommend(selected_movie, top_k=5)
                    
                    st.markdown(f"**🎯 Usando modelo: {selected_model}**")
                    
                    for i, rec in enumerate(recs, 1):
                        title = rec[0]
                        score = rec[1] if len(rec) > 1 else 0
                        genre = rec[2] if len(rec) > 2 else "N/A"
                        
                        st.markdown(f"""
                        <div class="rec-card">
                            <p class="rec-title">{i}. {title} <span class="rec-score">{score*100:.1f}%</span></p>
                            <p class="rec-genre">🏷️ {genre}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif get_recs and not selected_movie:
            st.warning("⚠️ Selecciona una película primero")

# -----------------------------------------------------------------------------
# TAB 2: EXPLICACIÓN - ¿Por qué cada algoritmo recomienda diferente?
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔍 ¿Por Qué Cada Algoritmo Recomienda Cosas Diferentes?")
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: #e50914;">🎯 La Clave: Cada algoritmo "ve" el texto de forma diferente</h4>
        <table style="width: 100%; color: #b3b3b3; margin-top: 1rem;">
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;"><b style="color: #4CAF50;">TF-IDF</b></td>
                <td style="padding: 10px;">Busca <b>palabras exactas</b> en común. Si dos películas usan "zombie", "apocalypse", son similares.</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;"><b style="color: #2196F3;">Doc2Vec</b></td>
                <td style="padding: 10px;">Aprende <b>patrones del documento completo</b>. Detecta estilo de escritura y estructura.</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><b style="color: #e50914;">SBERT</b></td>
                <td style="padding: 10px;">Entiende el <b>significado semántico</b>. "Terror" y "miedo" son lo mismo para SBERT.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector de película
    exp_movie = st.selectbox(
        "🎬 Selecciona una película para analizar:",
        options=[""] + titles,
        key="exp_movie_main"
    )
    
    if exp_movie and len(models["recommenders"]) > 0 and data is not None:
        # =====================================================================
        # INFORMACIÓN DE LA PELÍCULA ORIGINAL
        # =====================================================================
        movie_data = data[data['title'] == exp_movie].iloc[0] if len(data[data['title'] == exp_movie]) > 0 else None
        
        if movie_data is not None:
            st.markdown("---")
            st.markdown("## 🎬 Película Seleccionada")
            
            col_info1, col_info2 = st.columns([1, 2])
            
            with col_info1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e50914 0%, #b20710 100%); padding: 1.5rem; border-radius: 10px;">
                    <h3 style="color: white; margin: 0;">{exp_movie}</h3>
                    <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
                        📅 {movie_data.get('release_year', 'N/A')} | 
                        🎭 {movie_data.get('type', 'N/A')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Géneros
                genres = movie_data.get('listed_in', '')
                if genres:
                    st.markdown("**🏷️ Géneros:**")
                    for g in str(genres).split(',')[:4]:
                        st.markdown(f"`{g.strip()}`")
            
            with col_info2:
                st.markdown("**📝 Descripción:**")
                description = movie_data.get('description', 'Sin descripción disponible')
                st.markdown(f"""
                <div style="background: #252525; padding: 1rem; border-radius: 10px; border-left: 4px solid #e50914;">
                    <p style="color: #b3b3b3; margin: 0; font-style: italic;">"{description}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Director/Cast si está disponible
                director = movie_data.get('director', '')
                cast = movie_data.get('cast', '')
                if director:
                    st.caption(f"🎬 Director: {director}")
                if cast:
                    st.caption(f"🌟 Cast: {str(cast)[:100]}...")
        
        # =====================================================================
        # OBTENER RECOMENDACIONES DE CADA MODELO
        # =====================================================================
        all_recommendations = {}
        for model_name, model in models["recommenders"].items():
            try:
                recs = model.recommend(exp_movie, top_k=3)
                all_recommendations[model_name] = recs
            except Exception as e:
                st.error(f"Error con {model_name}: {e}")
        
        if all_recommendations:
            # =================================================================
            # COMPARACIÓN LADO A LADO CON INFORMACIÓN DETALLADA
            # =================================================================
            st.markdown("---")
            st.markdown("## 📊 ¿Qué recomienda cada algoritmo?")
            
            for model_name, recs in all_recommendations.items():
                color = {"TF-IDF": "#4CAF50", "Doc2Vec": "#2196F3", "SBERT": "#e50914"}.get(model_name, "#666")
                icon = {"TF-IDF": "🟢", "Doc2Vec": "🔵", "SBERT": "🔴"}.get(model_name, "⚪")
                
                st.markdown(f"""
                <div style="background: #1a1a1a; padding: 1rem; border-radius: 10px; border-left: 5px solid {color}; margin: 1rem 0;">
                    <h3 style="color: {color}; margin: 0;">{icon} {model_name}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar cada recomendación con detalles
                for i, rec in enumerate(recs[:3], 1):
                    rec_title, score, rec_genre, rec_desc = rec
                    
                    # Obtener más información del dataframe
                    rec_data = data[data['title'] == rec_title].iloc[0] if len(data[data['title'] == rec_title]) > 0 else None
                    rec_year = rec_data.get('release_year', 'N/A') if rec_data is not None else 'N/A'
                    rec_type = rec_data.get('type', '') if rec_data is not None else ''
                    rec_full_desc = rec_data.get('description', rec_desc) if rec_data is not None else rec_desc
                    
                    col_score, col_content = st.columns([1, 4])
                    
                    with col_score:
                        st.markdown(f"""
                        <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center;">
                            <span style="font-size: 2rem; font-weight: bold; color: white;">{score*100:.0f}%</span>
                            <br><span style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">similitud</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_content:
                        st.markdown(f"""
                        <div style="background: #252525; padding: 1rem; border-radius: 10px;">
                            <h4 style="color: white; margin: 0 0 0.5rem 0;">{i}. {rec_title}</h4>
                            <p style="color: #888; margin: 0;">📅 {rec_year} | 🎭 {rec_type}</p>
                            <p style="color: #666; margin: 0.3rem 0;">🏷️ {rec_genre}</p>
                            <p style="color: #999; margin: 0.5rem 0; font-size: 0.9rem; font-style: italic;">
                                "{str(rec_full_desc)[:200]}..."
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("")  # Espaciado
                
                # Mostrar explicación específica del modelo
                if model_name == "TF-IDF" and hasattr(models["recommenders"]["TF-IDF"], 'explain_recommendation'):
                    with st.expander(f"🔍 Ver palabras clave compartidas ({model_name})"):
                        first_rec = recs[0][0]
                        try:
                            explanation = models["recommenders"]["TF-IDF"].explain_recommendation(exp_movie, first_rec)
                            if "error" not in explanation:
                                st.markdown(f"**Palabras en común entre '{exp_movie}' y '{first_rec}':**")
                                common_terms = explanation.get("common_terms", [])
                                if common_terms:
                                    cols_terms = st.columns(min(5, len(common_terms)))
                                    for j, (term, weight) in enumerate(common_terms[:5]):
                                        with cols_terms[j]:
                                            st.markdown(f"""
                                            <div style="background: #333; padding: 0.5rem; border-radius: 5px; text-align: center;">
                                                <span style="color: #4CAF50; font-weight: bold;">{term}</span>
                                                <br><span style="color: #666; font-size: 0.7rem;">peso: {weight:.2f}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                st.caption(f"Total de términos compartidos: {explanation.get('total_common_terms', 0)}")
                        except:
                            pass
            
            # =================================================================
            # ANÁLISIS COMPARATIVO
            # =================================================================
            st.markdown("---")
            st.markdown("## 🔬 Análisis: ¿Por Qué Son Diferentes?")
            
            # Encontrar películas únicas por modelo
            recs_by_model = {}
            all_rec_titles = set()
            for model_name, recs in all_recommendations.items():
                recs_by_model[model_name] = set([r[0] for r in recs])
                all_rec_titles.update([r[0] for r in recs])
            
            # Películas en común vs únicas
            common_recs = set.intersection(*recs_by_model.values()) if len(recs_by_model) > 1 else set()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤝 Coincidencias")
                if common_recs:
                    for title in common_recs:
                        # Obtener info de la película
                        rec_data = data[data['title'] == title].iloc[0] if len(data[data['title'] == title]) > 0 else None
                        genre = rec_data.get('listed_in', 'N/A')[:40] if rec_data is not None else 'N/A'
                        st.success(f"✅ **{title}**\n\n🏷️ {genre}")
                    st.markdown("""
                    <div style="background: #1a3d1a; padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
                        <p style="color: #90EE90; margin: 0; font-size: 0.9rem;">
                        💡 <b>Alta confianza:</b> Todos los algoritmos coinciden. 
                        Similar en palabras, patrones y significado.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Los modelos recomiendan películas completamente diferentes")
                    st.markdown("""
                    <div style="background: #333; padding: 0.8rem; border-radius: 8px;">
                        <p style="color: #b3b3b3; margin: 0; font-size: 0.9rem;">
                        🎯 Esto indica que la película tiene características 
                        que cada algoritmo interpreta de forma diferente.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Recomendaciones Únicas")
                for model_name, rec_titles in recs_by_model.items():
                    unique = rec_titles - set.union(*[v for k, v in recs_by_model.items() if k != model_name]) if len(recs_by_model) > 1 else rec_titles
                    color_emoji = {"TF-IDF": "🟢", "Doc2Vec": "🔵", "SBERT": "🔴"}.get(model_name, "⚪")
                    
                    for title in unique:
                        # Obtener info de la película
                        rec_data = data[data['title'] == title].iloc[0] if len(data[data['title'] == title]) > 0 else None
                        genre = rec_data.get('listed_in', 'N/A')[:30] if rec_data is not None else 'N/A'
                        
                        st.markdown(f"{color_emoji} **{title}** ← Solo {model_name}")
                        st.caption(f"  🏷️ {genre}")
            
            # =================================================================
            # EXPLICACIÓN DETALLADA POR ALGORITMO
            # =================================================================
            st.markdown("---")
            st.markdown("## 📖 ¿Por Qué Cada Algoritmo Eligió Estas Películas?")
            
            # TF-IDF
            if "TF-IDF" in all_recommendations:
                with st.expander("🟢 **TF-IDF: Busca palabras exactas**", expanded=True):
                    st.markdown(f"""
                    **¿Por qué TF-IDF recomienda estas películas para "{exp_movie}"?**
                    
                    TF-IDF encontró que estas películas **comparten las mismas palabras** en sus descripciones.
                    """)
                    
                    # Mostrar descripción original y las recomendadas
                    if movie_data is not None:
                        original_desc = str(movie_data.get('description', ''))
                        st.markdown("**Descripción original:**")
                        st.markdown(f"> _{original_desc[:300]}..._")
                        
                        st.markdown("**Palabras clave que TF-IDF detectó:**")
                        if hasattr(models["recommenders"]["TF-IDF"], 'explain_recommendation'):
                            first_rec = all_recommendations["TF-IDF"][0][0]
                            try:
                                exp = models["recommenders"]["TF-IDF"].explain_recommendation(exp_movie, first_rec)
                                if "source_keywords" in exp:
                                    keywords = [t[0] for t in exp["source_keywords"][:6]]
                                    st.markdown(" • ".join([f"`{k}`" for k in keywords]))
                            except:
                                pass
                    
                    st.warning("⚠️ **Limitación:** Si usan sinónimos (ej: 'terror' vs 'miedo'), TF-IDF NO las conectará.")
            
            # Doc2Vec
            if "Doc2Vec" in all_recommendations:
                with st.expander("🔵 **Doc2Vec: Detecta patrones de escritura**"):
                    st.markdown(f"""
                    **¿Por qué Doc2Vec recomienda estas películas para "{exp_movie}"?**
                    
                    Doc2Vec aprendió que estas descripciones tienen **estructura y estilo similar**.
                    
                    Busca patrones como:
                    - Longitud similar de oraciones
                    - Combinaciones de palabras frecuentes
                    - Estilo narrativo parecido
                    
                    Por eso puede recomendar películas diferentes a TF-IDF: 
                    detecta **cómo** está escrito, no solo **qué** palabras usa.
                    """)
            
            # SBERT
            if "SBERT" in all_recommendations:
                with st.expander("🔴 **SBERT: Entiende el significado**"):
                    st.markdown(f"""
                    **¿Por qué SBERT recomienda estas películas para "{exp_movie}"?**
                    
                    SBERT usa un modelo de **inteligencia artificial pre-entrenado** que ya conoce 
                    el significado de las palabras y sus relaciones.
                    
                    Puede entender que:
                    - "película de terror" ≈ "film de miedo"
                    - "viaje espacial" ≈ "aventura intergaláctica"
                    - "historia de amor" ≈ "romance"
                    
                    Por eso SBERT puede encontrar películas **temáticamente similares** 
                    aunque no compartan ninguna palabra exacta.
                    """)
                    
                    st.success("✅ SBERT es el más preciso para búsqueda semántica.")
            
            # =================================================================
            # RESUMEN FINAL
            # =================================================================
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); padding: 1.5rem; border-radius: 10px; border: 1px solid #333;">
                <h4 style="color: #e50914;">📋 Resumen: ¿Cuál Algoritmo Usar?</h4>
                <table style="width: 100%; color: #b3b3b3;">
                    <tr style="border-bottom: 1px solid #444;">
                        <td style="padding: 8px;"><b>Situación</b></td>
                        <td style="padding: 8px;"><b>Mejor Algoritmo</b></td>
                        <td style="padding: 8px;"><b>Por qué</b></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px;">Buscar secuelas o spin-offs</td>
                        <td style="padding: 8px; color: #4CAF50;">TF-IDF</td>
                        <td style="padding: 8px;">Comparten nombres y términos específicos</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px;">Encontrar películas con "estilo" similar</td>
                        <td style="padding: 8px; color: #2196F3;">Doc2Vec</td>
                        <td style="padding: 8px;">Captura patrones de escritura</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px;">Buscar por tema/concepto</td>
                        <td style="padding: 8px; color: #e50914;">SBERT</td>
                        <td style="padding: 8px;">Entiende significado semántico</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">Máxima confianza</td>
                        <td style="padding: 8px; color: #FFD700;">Los 3 coinciden</td>
                        <td style="padding: 8px;">Si todos recomiendan lo mismo, es muy probable que sea buena recomendación</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
    
    elif exp_movie:
        st.warning("No hay modelos cargados o no hay datos disponibles")
    else:
        st.info("👆 Selecciona una película arriba para ver el análisis comparativo completo")

# -----------------------------------------------------------------------------
# TAB 3: CLASIFICADOR
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🏷️ Clasificador de Géneros")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        synopsis = st.text_area(
            "Escribe una sinopsis:",
            height=150,
            placeholder="Describe la trama de una película o serie..."
        )
        
        available_clfs = list(models["classifiers"].keys())
        selected_clf = st.selectbox(
            "Clasificador:",
            options=available_clfs if available_clfs else ["No hay modelos"],
            key="clf_select"
        )
        
        predict_btn = st.button("🎯 Predecir Géneros", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Géneros Predichos")
        
        if predict_btn and synopsis.strip() and selected_clf in models["classifiers"]:
            with st.spinner("Clasificando..."):
                try:
                    model = models["classifiers"][selected_clf]
                    predictions = model.predict(synopsis.strip(), top_k=5)
                    
                    # Crear gráfico de barras
                    if predictions:
                        genres = list(predictions.keys())
                        probs = [v * 100 for v in predictions.values()]
                        
                        fig = go.Figure(go.Bar(
                            x=probs,
                            y=genres,
                            orientation='h',
                            marker=dict(
                                color=probs,
                                colorscale=[[0, '#ff6b6b'], [1, '#e50914']]
                            ),
                            text=[f"{p:.1f}%" for p in probs],
                            textposition='inside'
                        ))
                        
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            height=300,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif predict_btn and not synopsis.strip():
            st.warning("⚠️ Escribe una sinopsis primero")

# -----------------------------------------------------------------------------
# TAB 4: EVALUACIÓN COMPARATIVA DE ALGORITMOS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📈 Evaluación: Comparación Cuantitativa de Modelos")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #e50914;">
        <h4 style="color: #e50914; margin: 0;">🎯 ¿Cuál algoritmo es mejor?</h4>
        <p style="color: #b3b3b3; margin: 0.5rem 0 0 0;">Selecciona una película y compara cómo cada modelo genera sus recomendaciones usando métricas objetivas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector de película para evaluar
    col_eval1, col_eval2 = st.columns([2, 1])
    
    with col_eval1:
        eval_movie = st.selectbox(
            "🎬 Película para evaluar:",
            options=[""] + titles,
            key="eval_movie_select",
            help="Selecciona una película para ver cómo la evalúa cada algoritmo"
        )
    
    with col_eval2:
        n_recs_eval = st.slider("Top N:", min_value=5, max_value=20, value=10, key="eval_n_recs")
    
    if eval_movie and st.button("🔬 Analizar Algoritmos", key="run_evaluation", type="primary"):
        with st.spinner("Ejecutando evaluación comparativa..."):
            try:
                # Obtener recomendaciones de cada modelo
                all_results = {}
                
                for model_name, model in models["recommenders"].items():
                    try:
                        # recommend() retorna List[Tuple[str, float, str, str]]
                        # (título, score, género, descripción)
                        recs_raw = model.recommend(eval_movie, top_k=n_recs_eval)
                        # Convertir a lista de diccionarios para facilitar el acceso
                        recs = [
                            {"title": r[0], "score": r[1], "genre": r[2], "description": r[3]}
                            for r in recs_raw
                        ]
                        all_results[model_name] = recs
                    except Exception as e:
                        st.warning(f"⚠️ {model_name} no disponible: {e}")
                
                if len(all_results) >= 2:
                    # =====================================================
                    # MÉTRICAS DE EVALUACIÓN (basadas en tu PPT)
                    # =====================================================
                    
                    st.markdown("---")
                    st.markdown("## 📊 Resultados de Evaluación")
                    
                    # --- 1. Distribución de Scores de Similitud ---
                    st.markdown("### 1️⃣ Distribución de Scores de Similitud")
                    st.markdown("""
                    <p style="color: #b3b3b3;">El <b>score de similitud</b> indica qué tan "seguro" está el modelo de cada recomendación. 
                    Un modelo con scores más altos tiene mayor confianza en sus predicciones.</p>
                    """, unsafe_allow_html=True)
                    
                    score_data = []
                    for model_name, recs in all_results.items():
                        for rec in recs:
                            score_data.append({
                                "Modelo": model_name,
                                "Título": rec.get('title', 'N/A')[:30],
                                "Score": rec.get('score', 0)
                            })
                    
                    df_scores = pd.DataFrame(score_data)
                    
                    # Box plot de distribución de scores
                    fig_box = px.box(
                        df_scores, x="Modelo", y="Score", color="Modelo",
                        color_discrete_map={"TF-IDF": "#4CAF50", "Doc2Vec": "#2196F3", "SBERT": "#e50914"},
                        title="Distribución de Confianza por Modelo"
                    )
                    fig_box.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False,
                        height=350
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Estadísticas de scores
                    col_stats = st.columns(len(all_results))
                    for i, (model_name, recs) in enumerate(all_results.items()):
                        scores = [r.get('score', 0) for r in recs]
                        with col_stats[i]:
                            avg_score = sum(scores) / len(scores) if scores else 0
                            max_score = max(scores) if scores else 0
                            min_score = min(scores) if scores else 0
                            st.metric(
                                label=f"📊 {model_name}",
                                value=f"{avg_score:.2%}",
                                delta=f"Rango: {min_score:.2%} - {max_score:.2%}"
                            )
                    
                    # --- 2. Diversidad de Géneros (Cobertura) ---
                    st.markdown("---")
                    st.markdown("### 2️⃣ Diversidad de Géneros (Coverage)")
                    st.markdown("""
                    <p style="color: #b3b3b3;">La <b>diversidad de géneros</b> mide cuántos géneros diferentes cubre cada modelo. 
                    Mayor diversidad = recomendaciones más variadas. Menor = más enfocado en un tipo específico.</p>
                    """, unsafe_allow_html=True)
                    
                    genre_coverage = {}
                    for model_name, recs in all_results.items():
                        all_genres = set()
                        for rec in recs:
                            # El género viene directamente en el resultado
                            genres = rec.get('genre', '')
                            if genres:
                                for g in str(genres).split(','):
                                    all_genres.add(g.strip())
                        genre_coverage[model_name] = all_genres
                    
                    # Gráfico de barras de diversidad
                    diversity_data = [
                        {"Modelo": name, "Géneros Únicos": len(genres), "Géneros": ", ".join(list(genres)[:5]) + ("..." if len(genres) > 5 else "")}
                        for name, genres in genre_coverage.items()
                    ]
                    df_diversity = pd.DataFrame(diversity_data)
                    
                    fig_div = px.bar(
                        df_diversity, x="Modelo", y="Géneros Únicos", color="Modelo",
                        color_discrete_map={"TF-IDF": "#4CAF50", "Doc2Vec": "#2196F3", "SBERT": "#e50914"},
                        title="Cantidad de Géneros Diferentes en Recomendaciones",
                        text="Géneros Únicos"
                    )
                    fig_div.update_traces(textposition='outside')
                    fig_div.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # Detalle de géneros por modelo
                    with st.expander("📋 Ver géneros cubiertos por cada modelo"):
                        for model_name, genres in genre_coverage.items():
                            st.markdown(f"**{model_name}:** {', '.join(sorted(genres)) if genres else 'N/A'}")
                    
                    # --- 3. Concordancia entre Modelos (Agreement) ---
                    st.markdown("---")
                    st.markdown("### 3️⃣ Concordancia entre Modelos (Precision@k)")
                    st.markdown("""
                    <p style="color: #b3b3b3;">La <b>concordancia</b> mide cuántos títulos coinciden entre modelos. 
                    Si dos modelos diferentes recomiendan lo mismo, hay mayor "consenso" de que es una buena recomendación.</p>
                    """, unsafe_allow_html=True)
                    
                    # Obtener sets de títulos
                    title_sets = {
                        name: set(r.get('title') for r in recs)
                        for name, recs in all_results.items()
                    }
                    
                    # Calcular intersecciones
                    model_names = list(title_sets.keys())
                    overlap_data = []
                    
                    for i, m1 in enumerate(model_names):
                        for j, m2 in enumerate(model_names):
                            if i < j:
                                common = title_sets[m1] & title_sets[m2]
                                overlap_pct = len(common) / n_recs_eval * 100
                                overlap_data.append({
                                    "Par de Modelos": f"{m1} ∩ {m2}",
                                    "Títulos en Común": len(common),
                                    "% Overlap": overlap_pct,
                                    "Títulos": ", ".join(list(common)[:3]) + ("..." if len(common) > 3 else "")
                                })
                    
                    # Gráfico de overlap
                    if overlap_data:
                        df_overlap = pd.DataFrame(overlap_data)
                        
                        fig_overlap = px.bar(
                            df_overlap, x="Par de Modelos", y="% Overlap",
                            color="% Overlap",
                            color_continuous_scale=["#e50914", "#4CAF50"],
                            text="Títulos en Común",
                            title="Porcentaje de Coincidencia entre Modelos"
                        )
                        fig_overlap.update_traces(texttemplate='%{text} títulos', textposition='outside')
                        fig_overlap.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            height=300
                        )
                        st.plotly_chart(fig_overlap, use_container_width=True)
                        
                        # Títulos en común
                        all_common = set.intersection(*title_sets.values()) if title_sets else set()
                        if all_common:
                            st.success(f"🎯 **Consenso Total:** {len(all_common)} título(s) recomendados por TODOS los modelos: {', '.join(all_common)}")
                        else:
                            st.info("📌 No hay títulos recomendados por todos los modelos simultáneamente.")
                    
                    # --- 4. Ranking Comparativo Visual ---
                    st.markdown("---")
                    st.markdown("### 4️⃣ Ranking Comparativo (Top 5)")
                    
                    # Crear tabla comparativa side by side
                    cols_ranking = st.columns(len(all_results))
                    
                    for idx, (model_name, recs) in enumerate(all_results.items()):
                        with cols_ranking[idx]:
                            color = {"TF-IDF": "#4CAF50", "Doc2Vec": "#2196F3", "SBERT": "#e50914"}.get(model_name, "#666")
                            st.markdown(f"""
                            <div style="background: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem; text-align: center;">
                                <h4 style="color: {color}; margin: 0;">{model_name}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, rec in enumerate(recs[:5]):
                                st.markdown(f"""
                                <div style="background: #1f1f1f; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border-left: 3px solid {color};">
                                    <small style="color: #666;">#{i+1}</small><br>
                                    <span style="color: white;">{rec.get('title', 'N/A')[:25]}</span><br>
                                    <small style="color: {color};">{rec.get('score', 0):.1%}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # --- 5. Resumen y Recomendación ---
                    st.markdown("---")
                    st.markdown("### 📋 Resumen y Conclusión")
                    
                    # Calcular métricas resumen
                    summary_data = []
                    for model_name, recs in all_results.items():
                        scores = [r.get('score', 0) for r in recs]
                        summary_data.append({
                            "Modelo": model_name,
                            "Score Promedio": f"{sum(scores)/len(scores):.2%}" if scores else "N/A",
                            "Diversidad (géneros)": len(genre_coverage.get(model_name, set())),
                            "Tipo de Enfoque": {
                                "TF-IDF": "📝 Palabras exactas",
                                "Doc2Vec": "📄 Contexto del documento",
                                "SBERT": "🧠 Significado semántico"
                            }.get(model_name, "N/A")
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                    
                    # Recomendación basada en métricas
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
                        <h4 style="color: #e50914;">💡 ¿Cuál elegir?</h4>
                        <ul style="color: #b3b3b3;">
                            <li><b style="color: #4CAF50;">TF-IDF:</b> Ideal para encontrar películas con descripciones similares (palabras clave compartidas)</li>
                            <li><b style="color: #2196F3;">Doc2Vec:</b> Bueno para capturar el "tono" general y tema del contenido</li>
                            <li><b style="color: #e50914;">SBERT:</b> El más avanzado - entiende el significado incluso con palabras diferentes</li>
                        </ul>
                        <p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">📌 No hay un "mejor" universal - depende de lo que busques. Para precisión semántica, SBERT. Para rapidez y simplicidad, TF-IDF.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Se necesitan al menos 2 modelos para comparar.")
                    
            except Exception as e:
                st.error(f"Error en evaluación: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # =========================================================================
    # SECCIÓN 2: EVALUACIÓN DE CLASIFICADORES CON TRAIN/TEST SPLIT
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🏷️ Evaluación de Clasificadores de Géneros")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #9C27B0;">
        <h4 style="color: #9C27B0; margin: 0;">🎯 Evaluación Real con Train/Test Split</h4>
        <p style="color: #b3b3b3; margin: 0.5rem 0 0 0;">Entrenamos los clasificadores con datos de entrenamiento y evaluamos con datos de prueba para obtener métricas reales de rendimiento.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuración del experimento
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        test_size = st.slider("📊 Porcentaje de datos para prueba:", min_value=10, max_value=40, value=20, step=5, key="clf_test_size")
    
    with col_cfg2:
        sample_size = st.slider("📁 Tamaño de muestra (títulos):", min_value=500, max_value=8000, value=2000, step=500, key="clf_sample_size")
    
    if st.button("🔬 Ejecutar Evaluación Real", key="run_real_clf_evaluation", type="primary"):
        with st.spinner("Entrenando y evaluando clasificadores... (esto puede tomar unos segundos)"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import MultiLabelBinarizer
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, accuracy_score
                import time
                
                # Preparar datos
                if data is not None:
                    # Tomar muestra aleatoria
                    df_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
                    
                    # Preparar textos y etiquetas
                    texts = df_sample['processed_description'].tolist()
                    labels = df_sample['genres_list'].tolist()
                    
                    # Filtrar filas con listas vacías
                    valid_idx = [i for i, (t, l) in enumerate(zip(texts, labels)) if t and l]
                    texts = [texts[i] for i in valid_idx]
                    labels = [labels[i] for i in valid_idx]
                    
                    st.info(f"📊 Usando {len(texts)} títulos para el experimento ({100-test_size}% train / {test_size}% test)")
                    
                    # Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        texts, labels, test_size=test_size/100, random_state=42, shuffle=True
                    )
                    
                    st.markdown(f"""
                    <div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <p style="color: #b3b3b3; margin: 0;">
                            📚 <b>Train:</b> {len(X_train)} títulos | 
                            🧪 <b>Test:</b> {len(X_test)} títulos
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Vectorizar
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    
                    # Binarizar etiquetas
                    mlb = MultiLabelBinarizer()
                    y_train_bin = mlb.fit_transform(y_train)
                    y_test_bin = mlb.transform(y_test)
                    
                    st.markdown(f"🏷️ **{len(mlb.classes_)} géneros** detectados: {', '.join(mlb.classes_[:5])}...")
                    
                    # Definir clasificadores
                    classifiers = {
                        "Logistic Regression": OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000)),
                        "Naive Bayes": OneVsRestClassifier(MultinomialNB()),
                        "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
                    }
                    
                    results = {}
                    
                    # Entrenar y evaluar cada clasificador
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (clf_name, clf) in enumerate(classifiers.items()):
                        status_text.text(f"Entrenando {clf_name}...")
                        
                        # Entrenar
                        start_time = time.time()
                        clf.fit(X_train_vec, y_train_bin)
                        train_time = time.time() - start_time
                        
                        # Predecir
                        y_pred = clf.predict(X_test_vec)
                        
                        # Calcular métricas
                        results[clf_name] = {
                            "train_time": train_time,
                            "f1_micro": f1_score(y_test_bin, y_pred, average='micro'),
                            "f1_macro": f1_score(y_test_bin, y_pred, average='macro'),
                            "hamming_loss": hamming_loss(y_test_bin, y_pred),
                            "precision_micro": precision_score(y_test_bin, y_pred, average='micro', zero_division=0),
                            "recall_micro": recall_score(y_test_bin, y_pred, average='micro', zero_division=0),
                            "y_pred": y_pred,
                            "classifier": clf
                        }
                        
                        progress_bar.progress((idx + 1) / len(classifiers))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # =====================================================
                    # MOSTRAR RESULTADOS
                    # =====================================================
                    
                    st.markdown("---")
                    st.markdown("## 📊 Resultados de Evaluación")
                    
                    # --- 1. Métricas Principales ---
                    st.markdown("### 1️⃣ Métricas de Rendimiento en Datos de Prueba")
                    
                    # Cards de métricas
                    cols_metrics = st.columns(len(results))
                    colors = {"Logistic Regression": "#FF9800", "Naive Bayes": "#9C27B0", "Random Forest": "#4CAF50"}
                    
                    for idx, (clf_name, res) in enumerate(results.items()):
                        with cols_metrics[idx]:
                            color = colors.get(clf_name, "#666")
                            st.markdown(f"""
                            <div style="background: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem; text-align: center;">
                                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{clf_name.split()[0]}</h4>
                                <p style="color: white; font-size: 2rem; margin: 0; font-weight: bold;">{res['f1_micro']:.1%}</p>
                                <p style="color: #888; font-size: 0.8rem; margin: 0;">F1 Micro</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # --- 2. Gráfico de Barras Comparativo ---
                    st.markdown("### 2️⃣ Comparación Visual de Métricas")
                    
                    metrics_data = []
                    for clf_name, res in results.items():
                        metrics_data.append({"Clasificador": clf_name, "Métrica": "F1 Micro", "Valor": res["f1_micro"]})
                        metrics_data.append({"Clasificador": clf_name, "Métrica": "F1 Macro", "Valor": res["f1_macro"]})
                        metrics_data.append({"Clasificador": clf_name, "Métrica": "Precision", "Valor": res["precision_micro"]})
                        metrics_data.append({"Clasificador": clf_name, "Métrica": "Recall", "Valor": res["recall_micro"]})
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    
                    fig_metrics = px.bar(
                        df_metrics, x="Métrica", y="Valor", color="Clasificador",
                        barmode="group",
                        color_discrete_map=colors,
                        title="Comparación de Métricas por Clasificador"
                    )
                    fig_metrics.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400,
                        yaxis=dict(tickformat='.0%')
                    )
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # --- 3. Hamming Loss (menor es mejor) ---
                    st.markdown("### 3️⃣ Hamming Loss (menor es mejor)")
                    st.markdown("""
                    <p style="color: #b3b3b3;">El Hamming Loss mide la fracción de etiquetas incorrectas. Un valor de 0 significa predicción perfecta.</p>
                    """, unsafe_allow_html=True)
                    
                    hamming_data = [{"Clasificador": name, "Hamming Loss": res["hamming_loss"]} for name, res in results.items()]
                    df_hamming = pd.DataFrame(hamming_data)
                    
                    fig_hamming = px.bar(
                        df_hamming, x="Clasificador", y="Hamming Loss", color="Clasificador",
                        color_discrete_map=colors,
                        text="Hamming Loss"
                    )
                    fig_hamming.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_hamming.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_hamming, use_container_width=True)
                    
                    # --- 4. Tiempo de Entrenamiento ---
                    st.markdown("### 4️⃣ Tiempo de Entrenamiento")
                    
                    time_data = [{"Clasificador": name, "Tiempo (s)": res["train_time"]} for name, res in results.items()]
                    df_time = pd.DataFrame(time_data)
                    
                    fig_time = px.bar(
                        df_time, x="Clasificador", y="Tiempo (s)", color="Clasificador",
                        color_discrete_map=colors,
                        text="Tiempo (s)"
                    )
                    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
                    fig_time.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # --- 5. Tabla Resumen ---
                    st.markdown("### 📋 Tabla Resumen Completa")
                    
                    summary_data = []
                    for clf_name, res in results.items():
                        summary_data.append({
                            "Clasificador": clf_name,
                            "F1 Micro": f"{res['f1_micro']:.2%}",
                            "F1 Macro": f"{res['f1_macro']:.2%}",
                            "Precision": f"{res['precision_micro']:.2%}",
                            "Recall": f"{res['recall_micro']:.2%}",
                            "Hamming Loss": f"{res['hamming_loss']:.4f}",
                            "Tiempo (s)": f"{res['train_time']:.2f}"
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                    
                    # --- 6. Conclusión Automática ---
                    st.markdown("### 💡 Conclusión")
                    
                    # Encontrar el mejor clasificador
                    best_f1 = max(results.items(), key=lambda x: x[1]["f1_micro"])
                    best_hamming = min(results.items(), key=lambda x: x[1]["hamming_loss"])
                    fastest = min(results.items(), key=lambda x: x[1]["train_time"])
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px;">
                        <h4 style="color: #9C27B0;">🏆 Resultados del Experimento</h4>
                        <ul style="color: #b3b3b3;">
                            <li><b style="color: #4CAF50;">Mejor F1-Score:</b> {best_f1[0]} con {best_f1[1]['f1_micro']:.2%}</li>
                            <li><b style="color: #2196F3;">Menor Hamming Loss:</b> {best_hamming[0]} con {best_hamming[1]['hamming_loss']:.4f}</li>
                            <li><b style="color: #FF9800;">Más Rápido:</b> {fastest[0]} con {fastest[1]['train_time']:.2f}s</li>
                        </ul>
                        <p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">
                            📌 <b>Interpretación:</b> F1 Micro considera todas las predicciones por igual, 
                            mientras que F1 Macro da el mismo peso a cada género (mejor para clases desbalanceadas).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- 7. Probar con nueva sinopsis ---
                    st.markdown("---")
                    st.markdown("### 🧪 Probar con Nueva Sinopsis")
                    
                    test_synopsis = st.text_area(
                        "📝 Escribe una sinopsis para clasificar:",
                        value="A young wizard discovers he has magical powers and must battle dark forces to save the world from an ancient evil.",
                        height=80,
                        key="clf_new_synopsis"
                    )
                    
                    if st.button("🏷️ Clasificar Sinopsis", key="classify_new"):
                        # Vectorizar la nueva sinopsis
                        new_vec = vectorizer.transform([test_synopsis])
                        
                        st.markdown("#### Predicciones por Clasificador:")
                        
                        cols_pred = st.columns(len(results))
                        
                        for idx, (clf_name, res) in enumerate(results.items()):
                            with cols_pred[idx]:
                                clf = res["classifier"]
                                # Obtener probabilidades
                                if hasattr(clf, 'predict_proba'):
                                    probs = clf.predict_proba(new_vec)[0]
                                    top_idx = probs.argsort()[-3:][::-1]
                                    top_genres = [(mlb.classes_[i], probs[i]) for i in top_idx]
                                else:
                                    pred = clf.predict(new_vec)[0]
                                    top_genres = [(mlb.classes_[i], 1.0) for i, v in enumerate(pred) if v == 1][:3]
                                
                                color = colors.get(clf_name, "#666")
                                st.markdown(f"""
                                <div style="background: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem;">
                                    <h5 style="color: {color}; margin: 0;">{clf_name.split()[0]}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for genre, prob in top_genres:
                                    st.markdown(f"- **{genre}**: {prob:.1%}")
                    
                else:
                    st.error("❌ No hay datos disponibles para la evaluación.")
                    
            except Exception as e:
                st.error(f"Error en evaluación: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Mostrar explicación cuando no hay evaluación en progreso
    st.markdown("---")
    st.markdown("### 📚 Guía de Métricas de Evaluación")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #9C27B0;">
            <h4 style="color: #9C27B0;">📊 Métricas para Recomendación</h4>
            <ul style="color: #b3b3b3;">
                <li><b>Score de Similitud:</b> Confianza del modelo (0-100%)</li>
                <li><b>Precision@k:</b> ¿Cuántos de los k recomendados son relevantes?</li>
                <li><b>Diversidad:</b> ¿Qué tan variadas son las recomendaciones?</li>
                <li><b>Concordancia:</b> ¿Los modelos están de acuerdo?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("""
        <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #9C27B0;">
            <h4 style="color: #9C27B0;">🏷️ Métricas para Clasificación</h4>
            <ul style="color: #b3b3b3;">
                <li><b>F1-Score:</b> Balance entre precisión y recall</li>
                <li><b>Hamming Loss:</b> Fracción de etiquetas incorrectas</li>
                <li><b>Accuracy:</b> Porcentaje de predicciones correctas</li>
                <li><b>Macro/Micro F1:</b> Para clases desbalanceadas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👆 Usa las herramientas de arriba para comparar **Recomendadores** (selecciona película) y **Clasificadores** (escribe sinopsis).")

# -----------------------------------------------------------------------------
# TAB 5: BENCHMARK DE RENDIMIENTO
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### ⏱️ Benchmark: Rendimiento de los Algoritmos")
    
    st.info("💡 Este tab muestra los tiempos de entrenamiento y características técnicas. Para ver **por qué** cada algoritmo recomienda diferentes películas, ve al tab 🔍 **Explicación**.")
    
    recs_data = benchmark.get("recommenders", {})
    if recs_data:
        chart_data = []
        for name, data in recs_data.items():
            if data.get("available"):
                chart_data.append({
                    "Modelo": name,
                    "Entrenamiento (s)": data.get("training_time", 0),
                    "Tipo": "Tiempo"
                })
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            
            fig = px.bar(
                df, 
                x="Modelo", 
                y="Entrenamiento (s)",
                color="Modelo",
                color_discrete_map={
                    "TF-IDF": "#4CAF50",
                    "Doc2Vec": "#2196F3", 
                    "SBERT": "#e50914"
                },
                text="Entrenamiento (s)"
            )
            
            fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa
    st.markdown("### 📊 Comparativa Técnica")
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px;">
        <table style="width: 100%; color: #b3b3b3; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #e50914;">
                <th style="padding: 12px; text-align: left; color: white;">Característica</th>
                <th style="padding: 12px; text-align: center; color: #4CAF50;">TF-IDF</th>
                <th style="padding: 12px; text-align: center; color: #2196F3;">Doc2Vec</th>
                <th style="padding: 12px; text-align: center; color: #e50914;">SBERT</th>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;">⚡ Velocidad de entrenamiento</td>
                <td style="padding: 10px; text-align: center;">⭐⭐⭐⭐⭐</td>
                <td style="padding: 10px; text-align: center;">⭐⭐⭐</td>
                <td style="padding: 10px; text-align: center;">⭐⭐</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;">🎯 Precisión semántica</td>
                <td style="padding: 10px; text-align: center;">⭐⭐</td>
                <td style="padding: 10px; text-align: center;">⭐⭐⭐</td>
                <td style="padding: 10px; text-align: center;">⭐⭐⭐⭐⭐</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;">💾 Uso de memoria</td>
                <td style="padding: 10px; text-align: center;">Bajo</td>
                <td style="padding: 10px; text-align: center;">Medio</td>
                <td style="padding: 10px; text-align: center;">Alto</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;">📦 Dependencias</td>
                <td style="padding: 10px; text-align: center;">sklearn</td>
                <td style="padding: 10px; text-align: center;">gensim</td>
                <td style="padding: 10px; text-align: center;">transformers</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px;">🔤 Entiende sinónimos</td>
                <td style="padding: 10px; text-align: center;">❌ No</td>
                <td style="padding: 10px; text-align: center;">⚡ Parcial</td>
                <td style="padding: 10px; text-align: center;">✅ Sí</td>
            </tr>
            <tr>
                <td style="padding: 10px;">🌐 Pre-entrenado</td>
                <td style="padding: 10px; text-align: center;">❌ No</td>
                <td style="padding: 10px; text-align: center;">❌ No</td>
                <td style="padding: 10px; text-align: center;">✅ Sí</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 ¿Cuándo usar cada uno?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #252525; padding: 1rem; border-radius: 10px; border-top: 4px solid #4CAF50;">
            <h4 style="color: #4CAF50;">TF-IDF</h4>
            <p style="color: #b3b3b3;">Úsalo cuando:</p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Necesitas velocidad</li>
                <li>Recursos limitados</li>
                <li>Las películas comparten nombres específicos</li>
                <li>Quieres un baseline simple</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #252525; padding: 1rem; border-radius: 10px; border-top: 4px solid #2196F3;">
            <h4 style="color: #2196F3;">Doc2Vec</h4>
            <p style="color: #b3b3b3;">Úsalo cuando:</p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Tienes muchos documentos</li>
                <li>Quieres capturar el "estilo"</li>
                <li>Balance entre velocidad y precisión</li>
                <li>No necesitas dependencias pesadas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #252525; padding: 1rem; border-radius: 10px; border-top: 4px solid #e50914;">
            <h4 style="color: #e50914;">SBERT</h4>
            <p style="color: #b3b3b3;">Úsalo cuando:</p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Precisión es lo más importante</li>
                <li>Necesitas entender significado</li>
                <li>Tienes GPU disponible</li>
                <li>Búsqueda semántica real</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 6: ¿CÓMO FUNCIONA?
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("### 📖 ¿Cómo Funcionan los Algoritmos de Recomendación?")
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h4 style="color: #e50914;">🎯 El Concepto Base: Similitud del Coseno</h4>
        <p style="color: #b3b3b3;">
        Todos los modelos convierten el texto en <b>vectores numéricos</b> (embeddings).
        Luego calculan qué tan "cerca" están dos vectores usando la <b>similitud del coseno</b>.
        </p>
        <div style="background: #0d0d0d; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <code style="color: #4CAF50;">
            similitud = cos(θ) = (A · B) / (||A|| × ||B||)
            </code>
            <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
            Si θ = 0° → similitud = 1 (idénticos)<br>
            Si θ = 90° → similitud = 0 (nada en común)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #1a1a1a; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4CAF50; height: 400px;">
            <h3 style="color: #4CAF50;">📊 TF-IDF</h3>
            <p style="color: #e50914;"><b>Term Frequency × Inverse Document Frequency</b></p>
            
            <p style="color: #b3b3b3;"><b>¿Cómo funciona?</b></p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Cuenta cuántas veces aparece cada palabra</li>
                <li>Penaliza palabras muy comunes (the, a, is)</li>
                <li>Premia palabras raras pero relevantes</li>
            </ul>
            
            <p style="color: #b3b3b3;"><b>Ejemplo:</b></p>
            <p style="color: #666; font-size: 0.85rem;">
            "zombie apocalypse" tiene alto peso porque es específico.<br>
            "movie about" tiene bajo peso porque aparece en todo.
            </p>
            
            <div style="background: #0d0d0d; padding: 0.5rem; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #4CAF50;">✅ Rápido</span><br>
                <span style="color: #ff6b6b;">❌ No entiende sinónimos</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1a1a1a; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196F3; height: 400px;">
            <h3 style="color: #2196F3;">📚 Doc2Vec</h3>
            <p style="color: #e50914;"><b>Paragraph Vectors (Word2Vec para documentos)</b></p>
            
            <p style="color: #b3b3b3;"><b>¿Cómo funciona?</b></p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Red neuronal que aprende contexto</li>
                <li>Cada documento → vector de 100 dimensiones</li>
                <li>Palabras cercanas en contexto → vectores cercanos</li>
            </ul>
            
            <p style="color: #b3b3b3;"><b>Ejemplo:</b></p>
            <p style="color: #666; font-size: 0.85rem;">
            Aprende que "rey" - "hombre" + "mujer" ≈ "reina".<br>
            Captura relaciones semánticas.
            </p>
            
            <div style="background: #0d0d0d; padding: 0.5rem; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #4CAF50;">✅ Captura contexto</span><br>
                <span style="color: #ff6b6b;">❌ Necesita entrenamiento</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #1a1a1a; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #e50914; height: 400px;">
            <h3 style="color: #e50914;">🧠 SBERT</h3>
            <p style="color: #e50914;"><b>Sentence-BERT (Transformers)</b></p>
            
            <p style="color: #b3b3b3;"><b>¿Cómo funciona?</b></p>
            <ul style="color: #8c8c8c; font-size: 0.9rem;">
                <li>Modelo pre-entrenado en millones de textos</li>
                <li>Arquitectura Transformer (como GPT)</li>
                <li>Entiende significado, no solo palabras</li>
            </ul>
            
            <p style="color: #b3b3b3;"><b>Ejemplo:</b></p>
            <p style="color: #666; font-size: 0.85rem;">
            "película de terror" ≈ "film de miedo"<br>
            Aunque no comparten palabras, entiende que son similares.
            </p>
            
            <div style="background: #0d0d0d; padding: 0.5rem; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #4CAF50;">✅ Mejor precisión</span><br>
                <span style="color: #ff6b6b;">❌ Más lento y pesado</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Diagrama de flujo
    st.markdown("### 🔄 Proceso de Recomendación")
    
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        FLUJO DE RECOMENDACIÓN                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   1. ENTRADA          2. VECTORIZACIÓN         3. SIMILITUD             │
    │   ─────────           ────────────────         ──────────               │
    │                                                                         │
    │   "Stranger Things"   ┌─────────────┐         Calcular coseno           │
    │         │             │  [0.2, 0.8, │         con todos los             │
    │         ▼             │   0.1, 0.5, │         vectores del              │
    │   Descripción  ──────▶│   ..., 0.3] │ ──────▶ catálogo                  │
    │   del título          └─────────────┘                │                  │
    │                         Vector 384D                  ▼                  │
    │                         (SBERT)              ┌──────────────┐           │
    │                                              │ Top 5 más    │           │
    │   4. RESULTADO                               │ similares    │           │
    │   ────────────                               └──────────────┘           │
    │                                                      │                  │
    │   1. Dark (85%)  ◄───────────────────────────────────┘                  │
    │   2. The OA (78%)                                                       │
    │   3. Black Mirror (72%)                                                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    ```
    """)

# -----------------------------------------------------------------------------
# TAB 7: MÉTRICAS
# -----------------------------------------------------------------------------
with tab7:
    st.markdown("### 📊 Dashboard del Sistema")
    
    info = benchmark.get("dataset_info", {})
    recs = benchmark.get("recommenders", {})
    clfs = benchmark.get("classifiers", {})
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{info.get('total_titles', 0):,}</div>
            <div class="metric-label">Títulos</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{info.get('unique_genres', 0)}</div>
            <div class="metric-label">Géneros</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        n_recs = sum(1 for r in recs.values() if r.get("available"))
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{n_recs}</div>
            <div class="metric-label">Recomendadores</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(clfs)}</div>
            <div class="metric-label">Clasificadores</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabla de modelos
    st.markdown("#### 📋 Detalle de Modelos")
    
    table_data = []
    for name, data in recs.items():
        table_data.append({
            "Modelo": name,
            "Tipo": "Recomendador",
            "Tiempo (s)": f"{data.get('training_time', 0):.3f}" if data.get("available") else "-",
            "Estado": "✅ Listo" if data.get("available") else "❌ No disponible"
        })
    
    for name, data in clfs.items():
        table_data.append({
            "Modelo": name,
            "Tipo": "Clasificador",
            "Tiempo (s)": f"{data.get('training_time', 0):.3f}",
            "Estado": "✅ Listo"
        })
    
    if table_data:
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Info adicional
    st.markdown("---")
    st.markdown("#### ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Directorio de modelos:** `{MODELS_DIR}`")
        st.markdown(f"**Dataset:** `{CSV_PATH}`")
    
    with col2:
        if benchmark.get("generated_at"):
            st.markdown(f"**Benchmark generado:** {benchmark['generated_at'][:19]}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Netflix AI - Sistema de Recomendación con NLP | Demo Educativa</p>
    <p style="font-size: 0.8rem;">TF-IDF • Doc2Vec • SBERT • Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
