"""
Tab de Evaluación de Modelos.
"""
import streamlit as st
from typing import Dict, Any, List
import pandas as pd

from src.evaluation import ModelEvaluator, evaluate_classifiers_with_split
from src.visualization import (
    plot_metrics_comparison, 
    plot_scores_distribution,
    plot_genre_diversity,
    plot_overlap_comparison,
    plot_classifier_comparison,
    plot_training_times
)
from src.config import get_model_color


def _render_metrics_help():
    """Renderiza el panel de ayuda de métricas."""
    with st.expander("📚 **Guía de Métricas** - ¿Qué significa cada una?", expanded=False):
        st.markdown("""
        ### 🎯 Métricas de Recomendación
        
        | Métrica | ¿Qué mide? | Fórmula | Ejemplo |
        |---------|-----------|---------|---------|
        | **Precision@K** | De las K recomendaciones, ¿cuántas son buenas? | `Relevantes en K / K` | K=5, 3 buenas → **60%** |
        | **Recall@K** | De TODAS las buenas, ¿cuántas capturamos en K? | `Relevantes en K / Total Relevantes` | 20 relevantes, 4 en K=5 → **20%** |
        | **nDCG@K** | ¿Los buenos están arriba en el ranking? | Penaliza posiciones bajas | Relevante en #1 > en #5 |
        | **MAP** | Precisión promedio en cada "acierto" | Combina orden y cantidad | Calidad general del ranking |
        | **Genre Diversity** | ¿Qué tan variadas son las recomendaciones? | `Géneros únicos / Total` | Alta = más variedad |
        
        ---
        
        ### ⚖️ Trade-offs Importantes
        
        ```
        K pequeño (3-5)          vs          K grande (15-20)
        ─────────────────────────────────────────────────────
        ✅ Alta Precision                 ✅ Alto Recall
        ❌ Bajo Recall                    ❌ Baja Precision
        🎯 Más exigente                   🎯 Más permisivo
        ```
        
        ---
        
        ### 🎲 ¿Qué hace cada parámetro?
        
        | Parámetro | Efecto al AUMENTAR | Efecto al DISMINUIR |
        |-----------|-------------------|---------------------|
        | **Top K** | ↑ Recall, ↓ Precision | ↓ Recall, ↑ Precision |
        | **Películas de prueba** | Más lento pero más confiable | Más rápido pero ruidoso |
        | **Semilla** | Controla qué películas se prueban (reproducibilidad) | Diferente seed = diferentes películas |
        
        ---
        
        ### 💡 Configuración Recomendada
        
        | Escenario | K | Películas | Por qué |
        |-----------|---|-----------|---------|
        | Demo rápida | 5 | 20 | Resultados instantáneos |
        | Evaluación seria | 10 | 50+ | Balance velocidad/precisión |
        | Comparación final | 10 | 100 | Resultados estables |
        """)


def _render_parameters_help():
    """Renderiza ayuda sobre los parámetros."""
    with st.expander("⚙️ **¿Cómo afectan los parámetros?**", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🔢 Top K
            
            **K = 5 (pocos resultados)**
            - Precision alta (fácil acertar con pocos)
            - Recall bajo (menos oportunidades)
            - Más exigente con el modelo
            
            **K = 20 (muchos resultados)**
            - Precision baja (difícil que todos sean buenos)
            - Recall alto (más oportunidades de encontrar)
            - Más permisivo
            """)
        
        with col2:
            st.markdown("""
            #### 🎲 Semilla Aleatoria
            
            **¿Por qué importa?**
            - Controla QUÉ películas se usan para evaluar
            - Mismo valor = resultados reproducibles
            - Diferente valor = diferentes películas de prueba
            
            **Tip:** Prueba varias semillas (42, 123, 456) 
            y promedia para resultados más robustos.
            """)


def render_tab_evaluation(models: Dict, data: pd.DataFrame, benchmark: Dict):
    """
    Renderiza el tab de evaluación de modelos.
    
    Args:
        models: Diccionario de modelos
        data: DataFrame con datos
        benchmark: Datos del benchmark
    """
    st.markdown("### 📊 Evaluación de Modelos")
    
    # Panel de ayuda principal
    _render_metrics_help()
    
    # Sub-tabs para diferentes tipos de evaluación
    eval_tab1, eval_tab2, eval_tab3 = st.tabs([
        "🎯 Recomendadores", 
        "🏷️ Clasificadores",
        "⏱️ Benchmark"
    ])
    
    with eval_tab1:
        _render_recommender_evaluation(models, data)
    
    with eval_tab2:
        _render_classifier_evaluation(data)
    
    with eval_tab3:
        _render_benchmark(benchmark)


def _render_recommender_evaluation(models: Dict, data: pd.DataFrame):
    """Evaluación de recomendadores."""
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1rem; border-radius: 10px; border-left: 4px solid #e50914; margin-bottom: 1rem;">
        <p style="color: #b3b3b3; margin: 0;">
        <b>Metodología:</b> Usamos <b>géneros como ground truth</b>. Si una recomendación comparte 
        al menos un género con la película original, la consideramos "relevante".
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Panel de ayuda de parámetros
    _render_parameters_help()
    
    # Configuración
    st.markdown("#### ⚙️ Configuración del Experimento")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_test = st.slider(
            "🎬 Películas de prueba:", 
            10, 100, 30, 
            key="eval_n_test",
            help="Más películas = resultados más estables pero más lento"
        )
    with col2:
        k_value = st.slider(
            "🔢 Top K:", 
            3, 20, 5, 
            key="eval_k",
            help="K bajo = más exigente (alta precision). K alto = más permisivo (alto recall)"
        )
    with col3:
        seed = st.number_input(
            "🎲 Semilla:", 
            0, 1000, 42, 
            key="eval_seed",
            help="Mismo valor = mismas películas de prueba (reproducible)"
        )
    
    if st.button("🔬 Calcular Métricas", type="primary", key="run_rec_eval"):
        if models.get("recommenders") and data is not None:
            evaluator = ModelEvaluator(data)
            
            progress = st.progress(0)
            status = st.empty()
            
            def update_progress(p, text):
                progress.progress(p)
                status.text(text)
            
            results = evaluator.evaluate_all_recommenders(
                models["recommenders"],
                n_test=n_test,
                k=k_value,
                seed=seed,
                progress_callback=update_progress
            )
            
            progress.empty()
            status.empty()
            
            if results:
                st.session_state['rec_eval_results'] = results
                st.success(f"✅ Evaluados {len(results)} modelos")
    
    # Mostrar resultados
    if 'rec_eval_results' in st.session_state:
        results = st.session_state['rec_eval_results']
        
        # Tabla
        df = pd.DataFrame(results).T.round(4)
        st.dataframe(df, use_container_width=True)
        
        # Gráfico
        fig = plot_metrics_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mejor por métrica
        st.markdown("#### 🏆 Mejor Modelo por Métrica")
        metrics = ["Precision@K", "Recall@K", "nDCG@K", "MAP"]
        cols = st.columns(len(metrics))
        
        for col, metric in zip(cols, metrics):
            with col:
                best = max(results.items(), key=lambda x: x[1].get(metric, 0))
                color = get_model_color(best[0])
                st.markdown(f"""
                <div style="background: {color}22; border: 2px solid {color}; border-radius: 8px; padding: 0.8rem; text-align: center;">
                    <small style="color: #888;">{metric}</small><br>
                    <b style="color: white;">{best[0]}</b><br>
                    <span style="color: {color};">{best[1].get(metric, 0):.4f}</span>
                </div>
                """, unsafe_allow_html=True)


def _render_classifier_evaluation(data: pd.DataFrame):
    """Evaluación de clasificadores."""
    
    st.markdown("""
    Entrenamos clasificadores con train/test split para obtener métricas reales.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("% Test:", 10, 40, 20, key="clf_test_size")
    with col2:
        sample_size = st.slider("Muestra:", 500, 5000, 2000, key="clf_sample")
    
    if st.button("🔬 Evaluar Clasificadores", type="primary", key="run_clf_eval"):
        if data is not None:
            progress = st.progress(0)
            status = st.empty()
            
            def update_progress(p, text):
                progress.progress(p)
                status.text(text)
            
            results = evaluate_classifiers_with_split(
                data,
                test_size=test_size/100,
                sample_size=sample_size,
                progress_callback=update_progress
            )
            
            progress.empty()
            status.empty()
            
            if results:
                st.session_state['clf_eval_results'] = results
                st.success(f"✅ Evaluados {len(results)} clasificadores")
    
    # Mostrar resultados
    if 'clf_eval_results' in st.session_state:
        results = st.session_state['clf_eval_results']
        
        # Cards de resumen
        cols = st.columns(len(results))
        colors = {"Logistic Regression": "#FF9800", "Naive Bayes": "#9C27B0", "Random Forest": "#4CAF50"}
        
        for col, (name, metrics) in zip(cols, results.items()):
            with col:
                color = colors.get(name, "#666")
                st.markdown(f"""
                <div style="background: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem; text-align: center;">
                    <b style="color: {color};">{name.split()[0]}</b><br>
                    <span style="color: white; font-size: 1.5rem;">{metrics['f1_micro']:.1%}</span><br>
                    <small style="color: #888;">F1 Micro</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Gráfico
        fig = plot_classifier_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla completa
        with st.expander("📋 Tabla completa"):
            df_data = []
            for name, m in results.items():
                df_data.append({
                    "Clasificador": name,
                    "F1 Micro": f"{m['f1_micro']:.2%}",
                    "F1 Macro": f"{m['f1_macro']:.2%}",
                    "Precision": f"{m['precision']:.2%}",
                    "Recall": f"{m['recall']:.2%}",
                    "Hamming Loss": f"{m['hamming_loss']:.4f}",
                    "Tiempo (s)": f"{m['train_time']:.2f}"
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)


def _render_benchmark(benchmark: Dict):
    """Muestra datos del benchmark."""
    
    info = benchmark.get("dataset_info", {})
    
    # Métricas del dataset
    st.markdown("#### 📊 Dataset")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Títulos", f"{info.get('total_titles', 0):,}")
    with cols[1]:
        st.metric("Géneros", info.get('unique_genres', 0))
    with cols[2]:
        st.metric("Películas", info.get('movies', 0))
    with cols[3]:
        st.metric("Series", info.get('tv_shows', 0))
    
    # Tiempos de entrenamiento
    st.markdown("#### ⏱️ Tiempos de Entrenamiento")
    
    fig = plot_training_times(benchmark)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa
    st.markdown("#### 📋 Comparativa Técnica")
    
    st.markdown("""
    | Característica | TF-IDF | Doc2Vec | SBERT |
    |---------------|--------|---------|-------|
    | ⚡ Velocidad | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
    | 🎯 Precisión | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    | 💾 Memoria | Bajo | Medio | Alto |
    | 🔤 Sinónimos | ❌ | ⚡ Parcial | ✅ |
    | 🌐 Pre-entrenado | ❌ | ❌ | ✅ |
    """)
    
    if benchmark.get("generated_at"):
        st.caption(f"📅 Generado: {benchmark['generated_at'][:19]}")
