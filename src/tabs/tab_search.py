"""
Tab de Búsqueda Semántica.
"""
import streamlit as st
from typing import Dict, Any, List
import pandas as pd

from src.ui_components import render_info_box


def render_tab_search(models: Dict, data: pd.DataFrame):
    """
    Renderiza el tab de búsqueda semántica.
    
    Args:
        models: Diccionario de modelos
        data: DataFrame con datos
    """
    st.markdown("### 🔎 Búsqueda Semántica")
    
    render_info_box(
        "🧠 Búsqueda por Significado",
        "Describe lo que quieres ver con tus propias palabras. SBERT entiende el significado semántico de tu consulta.",
        "#e50914"
    )
    
    # Input de búsqueda
    col_search, col_n = st.columns([3, 1])
    
    with col_search:
        query = st.text_input(
            "🔍 ¿Qué tipo de película buscas?",
            placeholder="Ej: película sobre supervivencia en una isla desierta",
            key="semantic_query"
        )
    
    with col_n:
        n_results = st.slider("Resultados:", 5, 20, 10, key="search_n")
    
    # Ejemplos rápidos
    st.markdown("**💡 Ejemplos:**")
    examples = [
        "historia de amor prohibido",
        "thriller psicológico con giros",
        "comedia familiar para niños",
        "documental sobre naturaleza"
    ]
    
    example_cols = st.columns(4)
    selected_example = None
    
    for col, example in zip(example_cols, examples):
        with col:
            if st.button(example[:18] + "...", key=f"ex_{example[:10]}", use_container_width=True):
                selected_example = example
    
    # Usar ejemplo si se seleccionó
    search_query = selected_example if selected_example else query
    
    # Buscar
    if st.button("🔎 Buscar", type="primary", use_container_width=True) or selected_example:
        if search_query and search_query.strip():
            _perform_search(models, data, search_query.strip(), n_results)
        else:
            st.warning("⚠️ Escribe una descripción")
    
    # Explicación
    with st.expander("ℹ️ ¿Cómo funciona?"):
        st.markdown("""
        **Búsqueda Semántica vs Tradicional:**
        
        | Tradicional | Semántica |
        |------------|-----------|
        | Busca palabras exactas | Entiende el significado |
        | "zombie" solo encuentra "zombie" | "muertos vivientes" = "zombie" |
        
        **SBERT** convierte tu consulta en un vector de 384 dimensiones que representa 
        su significado, luego encuentra las películas más similares.
        """)


def _perform_search(models: Dict, data: pd.DataFrame, query: str, n_results: int):
    """Ejecuta la búsqueda semántica."""
    
    if "SBERT" not in models.get("recommenders", {}):
        st.warning("⚠️ Modelo SBERT no disponible. Ejecuta `python train_models.py`")
        return
    
    sbert = models["recommenders"]["SBERT"]
    
    with st.spinner("Buscando..."):
        try:
            # Intentar búsqueda semántica
            if hasattr(sbert, 'semantic_search'):
                results = sbert.semantic_search(query, top_k=n_results)
            else:
                st.warning("⚠️ Método semantic_search no disponible")
                return
            
            if not results:
                st.info("No se encontraron resultados. Intenta con otra descripción.")
                return
            
            st.markdown(f"### 🎯 {len(results)} resultados para: *\"{query}\"*")
            
            for i, (title, score, genre, desc) in enumerate(results, 1):
                # Info adicional
                movie_info = data[data['title'] == title].iloc[0] if len(data[data['title'] == title]) > 0 else None
                year = movie_info['release_year'] if movie_info is not None else 'N/A'
                movie_type = movie_info['type'] if movie_info is not None else 'N/A'
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #2a2a2a 0%, #1a1a1a 100%); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #e50914;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="color: white; margin: 0;">{i}. {title}</h4>
                        <span style="background: #e50914; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{score*100:.1f}%</span>
                    </div>
                    <p style="color: #888; margin: 0.2rem 0;">📅 {year} | 🎭 {movie_type} | 🏷️ {genre}</p>
                    <p style="color: #999; font-size: 0.9rem; font-style: italic;">"{desc[:180]}..."</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
