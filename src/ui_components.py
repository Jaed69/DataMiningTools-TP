"""
Componentes de UI reutilizables para Streamlit.
================================================
Funciones para crear elementos visuales consistentes.
"""
import streamlit as st
from typing import List, Tuple, Dict, Any, Optional
from src.config import get_model_color, get_model_icon, CSS_STYLES


def inject_css():
    """Inyecta los estilos CSS personalizados."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def render_header():
    """Renderiza el header principal de la aplicación."""
    st.markdown("""
    <div class="netflix-header">
        <h1 class="netflix-title">NETFLIX AI</h1>
        <p class="netflix-subtitle">Sistema de Recomendación con NLP</p>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(
    rank: int, 
    title: str, 
    score: float, 
    genre: str, 
    description: str = "",
    color: str = "#e50914"
):
    """Renderiza una tarjeta de recomendación."""
    st.markdown(f"""
    <div class="rec-card" style="border-left-color: {color};">
        <p class="rec-title">{rank}. {title} <span class="rec-score">{score*100:.1f}%</span></p>
        <p class="rec-genre">🏷️ {genre}</p>
        {f'<p style="color: #666; font-size: 0.85rem; margin-top: 5px;">"{description[:150]}..."</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def render_recommendations(
    recommendations: List[Tuple], 
    model_name: str = ""
):
    """Renderiza una lista completa de recomendaciones."""
    color = get_model_color(model_name) if model_name else "#e50914"
    
    if model_name:
        icon = get_model_icon(model_name)
        st.markdown(f"**{icon} Modelo: {model_name}**")
    
    for i, rec in enumerate(recommendations, 1):
        title = rec[0]
        score = rec[1] if len(rec) > 1 else 0.0
        genre = rec[2] if len(rec) > 2 else "N/A"
        desc = rec[3] if len(rec) > 3 else ""
        
        render_recommendation_card(i, title, score, genre, desc, color)


def render_metric_card(label: str, value: str, delta: str = None, color: str = "#e50914"):
    """Renderiza una tarjeta de métrica."""
    delta_html = f'<p style="color: #888; font-size: 0.8rem; margin: 0;">{delta}</p>' if delta else ''
    
    st.markdown(f"""
    <div class="metric-card" style="border-top: 3px solid {color};">
        <p style="color: {color}; font-size: 1.8rem; font-weight: bold; margin: 0;">{value}</p>
        <p style="color: #b3b3b3; font-size: 0.9rem; margin: 0.3rem 0 0 0;">{label}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(title: str, content: str, color: str = "#e50914"):
    """Renderiza una caja de información."""
    st.markdown(f"""
    <div class="info-box" style="border-left-color: {color};">
        <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{title}</h4>
        <p style="color: #b3b3b3; margin: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_model_comparison_table(results: Dict[str, List[Tuple]]):
    """Renderiza tabla comparativa de modelos lado a lado."""
    cols = st.columns(len(results))
    
    for idx, (model_name, recs) in enumerate(results.items()):
        color = get_model_color(model_name)
        icon = get_model_icon(model_name)
        
        with cols[idx]:
            st.markdown(f"""
            <div style="background: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 0.8rem; text-align: center; margin-bottom: 0.5rem;">
                <h4 style="color: {color}; margin: 0;">{icon} {model_name}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, rec in enumerate(recs[:5], 1):
                title = rec[0]
                score = rec[1] if len(rec) > 1 else 0.0
                st.markdown(f"""
                <div style="background: #1f1f1f; padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px; border-left: 3px solid {color};">
                    <small style="color: #666;">#{i}</small>
                    <span style="color: white; display: block;">{title[:30]}{'...' if len(title) > 30 else ''}</span>
                    <small style="color: {color};">{score:.1%}</small>
                </div>
                """, unsafe_allow_html=True)


def render_sidebar(models: Dict, titles_count: int):
    """Renderiza el sidebar con información del sistema."""
    with st.sidebar:
        st.markdown("## 🎛️ Panel de Control")
        st.markdown("---")
        
        # Estado de modelos
        st.markdown("### 📊 Modelos")
        
        # Recomendadores
        rec_count = len(models.get("recommenders", {}))
        st.markdown(f"**Recomendadores:** {rec_count}")
        for name in models.get("recommenders", {}).keys():
            color = get_model_color(name)
            st.markdown(f"<span style='color:{color}'>✅ {name}</span>", unsafe_allow_html=True)
        
        # Clasificadores  
        clf_count = len(models.get("classifiers", {}))
        st.markdown(f"**Clasificadores:** {clf_count}")
        for name in models.get("classifiers", {}).keys():
            st.markdown(f"✅ {name}")
        
        st.markdown("---")
        st.markdown(f"📁 **{titles_count:,}** títulos")
        
        if not models.get("recommenders"):
            st.warning("⚠️ Ejecuta `python train_models.py`")


def render_movie_info(movie_data, title: str):
    """Renderiza información detallada de una película."""
    if movie_data is None:
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        year = movie_data.get('release_year', 'N/A')
        movie_type = movie_data.get('type', 'N/A')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e50914 0%, #b20710 100%); padding: 1rem; border-radius: 10px;">
            <h4 style="color: white; margin: 0;">{title}</h4>
            <p style="color: rgba(255,255,255,0.8); margin-top: 0.3rem;">📅 {year} | 🎭 {movie_type}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Géneros
        genres = movie_data.get('listed_in', '')
        if genres:
            st.markdown("**🏷️ Géneros:**")
            for g in str(genres).split(',')[:4]:
                st.markdown(f"`{g.strip()}`")
    
    with col2:
        description = movie_data.get('description', 'Sin descripción')
        st.markdown(f"""
        <div style="background: #252525; padding: 1rem; border-radius: 10px; border-left: 4px solid #e50914;">
            <p style="color: #b3b3b3; margin: 0; font-style: italic;">"{description[:300]}..."</p>
        </div>
        """, unsafe_allow_html=True)
