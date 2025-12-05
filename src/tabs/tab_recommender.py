"""
Tab de Recomendador - Funcionalidad principal.
"""
import streamlit as st
from typing import Dict, Any, List
import pandas as pd

from src.ui_components import render_recommendations, render_model_comparison_table, render_movie_info
from src.config import get_model_color, get_model_icon


def render_tab_recommender(models: Dict, titles: List[str], data: pd.DataFrame):
    """
    Renderiza el tab principal de recomendaciones.
    
    Args:
        models: Diccionario de modelos cargados
        titles: Lista de títulos disponibles
        data: DataFrame con datos
    """
    st.markdown("### 🎬 Sistema de Recomendación")
    
    # Layout principal
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        # Selector de película
        selected_movie = st.selectbox(
            "Selecciona una película:",
            options=[""] + titles,
            key="rec_movie",
            help="Escribe para buscar"
        )
        
        # Selector de modelo
        available_models = list(models.get("recommenders", {}).keys())
        
        # Opción de comparar todos
        compare_mode = st.checkbox("🔄 Comparar todos los modelos", key="compare_mode")
        
        if not compare_mode:
            selected_model = st.selectbox(
                "Modelo de NLP:",
                options=available_models if available_models else ["No hay modelos"],
                key="rec_model"
            )
        
        # Número de recomendaciones
        n_recs = st.slider("Número de recomendaciones:", 3, 10, 5, key="n_recs")
        
        # Botón
        get_recs = st.button("🔍 Obtener Recomendaciones", type="primary", use_container_width=True)
    
    with col_results:
        st.markdown("#### Resultados")
        
        if get_recs and selected_movie:
            # Mostrar info de la película seleccionada
            movie_data = data[data['title'] == selected_movie].iloc[0] if len(data[data['title'] == selected_movie]) > 0 else None
            if movie_data is not None:
                with st.expander("📋 Película seleccionada", expanded=False):
                    render_movie_info(movie_data, selected_movie)
            
            if compare_mode:
                # Modo comparación: mostrar todos los modelos
                _render_comparison_view(models["recommenders"], selected_movie, n_recs, data)
            else:
                # Modo simple: un solo modelo
                if selected_model in models["recommenders"]:
                    _render_single_model(models["recommenders"][selected_model], selected_model, selected_movie, n_recs)
                else:
                    st.warning("⚠️ Modelo no disponible")
        
        elif get_recs:
            st.warning("⚠️ Selecciona una película primero")


def _render_single_model(model, model_name: str, movie: str, n_recs: int):
    """Renderiza recomendaciones de un solo modelo."""
    with st.spinner(f"Buscando con {model_name}..."):
        try:
            recs = model.recommend(movie, top_k=n_recs)
            render_recommendations(recs, model_name)
            
            # Explicación si está disponible
            if hasattr(model, 'explain_recommendation') and recs:
                with st.expander("🔍 Ver explicación"):
                    try:
                        explanation = model.explain_recommendation(movie, recs[0][0])
                        if "common_terms" in explanation:
                            st.markdown("**Palabras en común:**")
                            terms = [t[0] for t in explanation["common_terms"][:5]]
                            st.markdown(" | ".join([f"`{t}`" for t in terms]))
                    except:
                        st.info("Explicación no disponible para este par")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")


def _render_comparison_view(models: Dict, movie: str, n_recs: int, data: pd.DataFrame):
    """Renderiza comparación de todos los modelos."""
    with st.spinner("Obteniendo recomendaciones de todos los modelos..."):
        all_recs = {}
        
        for name, model in models.items():
            try:
                recs = model.recommend(movie, top_k=n_recs)
                all_recs[name] = recs
            except Exception as e:
                st.warning(f"⚠️ {name}: {str(e)}")
        
        if all_recs:
            # Mostrar comparación lado a lado
            render_model_comparison_table(all_recs)
            
            # Análisis de coincidencias
            st.markdown("---")
            st.markdown("#### 🔬 Análisis de Coincidencias")
            
            title_sets = {name: set(r[0] for r in recs) for name, recs in all_recs.items()}
            
            # Títulos en común de todos
            common_all = set.intersection(*title_sets.values()) if len(title_sets) > 1 else set()
            
            if common_all:
                st.success(f"🎯 **Consenso total:** {', '.join(common_all)}")
            else:
                st.info("Los modelos recomiendan películas diferentes")
            
            # Overlap por pares
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤝 Coincidencias por par:**")
                model_names = list(title_sets.keys())
                for i, m1 in enumerate(model_names):
                    for j, m2 in enumerate(model_names):
                        if i < j:
                            common = title_sets[m1] & title_sets[m2]
                            if common:
                                st.markdown(f"- {m1} ∩ {m2}: {len(common)} títulos")
            
            with col2:
                st.markdown("**🎯 Recomendaciones únicas:**")
                for name, rec_set in title_sets.items():
                    others = set.union(*[s for n, s in title_sets.items() if n != name]) if len(title_sets) > 1 else set()
                    unique = rec_set - others
                    if unique:
                        icon = get_model_icon(name)
                        st.markdown(f"{icon} **{name}:** {', '.join(list(unique)[:2])}...")
