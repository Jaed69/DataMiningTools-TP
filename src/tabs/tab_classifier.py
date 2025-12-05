"""
Tab de Clasificador de Géneros.
"""
import streamlit as st
from typing import Dict, Any, List
import pandas as pd

from src.visualization import plot_genre_predictions


def render_tab_classifier(models: Dict):
    """
    Renderiza el tab de clasificación de géneros.
    
    Args:
        models: Diccionario de modelos
    """
    st.markdown("### 🏷️ Clasificador de Géneros")
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1rem; border-radius: 10px; border-left: 4px solid #9C27B0; margin-bottom: 1rem;">
        <p style="color: #b3b3b3; margin: 0;">Escribe una sinopsis y el modelo predecirá los géneros más probables.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        synopsis = st.text_area(
            "📝 Escribe una sinopsis:",
            height=150,
            placeholder="Describe la trama de una película o serie...",
            key="clf_synopsis"
        )
        
        # Ejemplos
        st.markdown("**Ejemplos rápidos:**")
        examples = {
            "🧙 Fantasía": "A young wizard discovers magical powers and must battle dark forces to save the world.",
            "😂 Comedia": "A group of friends get into hilarious misadventures during a road trip.",
            "🔪 Terror": "A family moves into a haunted house where supernatural events begin to occur."
        }
        
        for label, text in examples.items():
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                synopsis = text
        
        # Selector de clasificador
        available_clfs = list(models.get("classifiers", {}).keys())
        selected_clf = st.selectbox(
            "Clasificador:",
            options=available_clfs if available_clfs else ["No hay modelos"],
            key="clf_model"
        )
        
        predict_btn = st.button("🎯 Predecir Géneros", type="primary", use_container_width=True)
    
    with col_output:
        st.markdown("#### Géneros Predichos")
        
        if predict_btn and synopsis.strip():
            if selected_clf in models.get("classifiers", {}):
                with st.spinner("Clasificando..."):
                    try:
                        model = models["classifiers"][selected_clf]
                        predictions = model.predict(synopsis.strip(), top_k=5)
                        
                        if predictions:
                            # Gráfico
                            fig = plot_genre_predictions(predictions)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Top género
                            top_genre = max(predictions.items(), key=lambda x: x[1])
                            st.success(f"🎯 Género principal: **{top_genre[0]}** ({top_genre[1]*100:.1f}%)")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("⚠️ Clasificador no disponible")
        
        elif predict_btn:
            st.warning("⚠️ Escribe una sinopsis primero")
        
        # Información sobre clasificadores
        with st.expander("ℹ️ Sobre los clasificadores"):
            st.markdown("""
            | Clasificador | Descripción |
            |-------------|-------------|
            | **Logistic** | Regresión logística. Rápido y simple. |
            | **NaiveBayes** | Basado en probabilidades. Bueno para texto. |
            | **RandomForest** | Ensemble de árboles. Más robusto. |
            
            Todos usan **One-vs-Rest** para clasificación multi-etiqueta.
            """)
