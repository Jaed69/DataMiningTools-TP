"""
Módulo de Visualización Centralizado.
=====================================
Funciones para crear gráficos y visualizaciones con Plotly.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any, Optional
from src.config import RECOMMENDER_MODELS


def get_dark_layout():
    """Retorna configuración de layout oscuro para Plotly."""
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )


def create_model_colors_map() -> Dict[str, str]:
    """Crea mapeo de colores para modelos."""
    colors = {}
    for name, config in RECOMMENDER_MODELS.items():
        colors[name] = config["color"]
    return colors


def plot_scores_distribution(results: Dict[str, List], title: str = "Distribución de Scores"):
    """
    Crea box plot de distribución de scores por modelo.
    
    Args:
        results: {modelo: lista de recomendaciones}
        title: Título del gráfico
    """
    score_data = []
    for model_name, recs in results.items():
        for rec in recs:
            score = rec[1] if len(rec) > 1 else 0
            score_data.append({
                "Modelo": model_name,
                "Score": score
            })
    
    df = pd.DataFrame(score_data)
    
    fig = px.box(
        df, x="Modelo", y="Score", color="Modelo",
        color_discrete_map=create_model_colors_map(),
        title=title
    )
    
    fig.update_layout(**get_dark_layout(), height=350, showlegend=False)
    return fig


def plot_genre_diversity(genre_coverage: Dict[str, set], title: str = "Diversidad de Géneros"):
    """
    Crea gráfico de barras de diversidad de géneros.
    
    Args:
        genre_coverage: {modelo: set de géneros}
    """
    data = [
        {"Modelo": name, "Géneros Únicos": len(genres)}
        for name, genres in genre_coverage.items()
    ]
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x="Modelo", y="Géneros Únicos", color="Modelo",
        color_discrete_map=create_model_colors_map(),
        title=title,
        text="Géneros Únicos"
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(**get_dark_layout(), height=300, showlegend=False)
    return fig


def plot_overlap_comparison(overlaps: Dict[str, Dict], k: int, title: str = "Coincidencia entre Modelos"):
    """
    Crea gráfico de overlap entre modelos.
    
    Args:
        overlaps: {par: {count, percentage, titles}}
        k: Número total de recomendaciones
    """
    data = [
        {
            "Par de Modelos": pair,
            "% Overlap": info["percentage"],
            "Títulos": info["count"]
        }
        for pair, info in overlaps.items()
    ]
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x="Par de Modelos", y="% Overlap",
        color="% Overlap",
        color_continuous_scale=["#e50914", "#4CAF50"],
        text="Títulos",
        title=title
    )
    
    fig.update_traces(texttemplate='%{text} títulos', textposition='outside')
    fig.update_layout(**get_dark_layout(), height=300)
    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]], 
    metrics_to_plot: List[str] = None,
    title: str = "Comparación de Métricas"
):
    """
    Crea gráfico de barras comparativo de métricas.
    
    Args:
        metrics: {modelo: {métrica: valor}}
        metrics_to_plot: Lista de métricas a incluir
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["Precision@K", "Recall@K", "nDCG@K", "MAP"]
    
    data = []
    for model_name, model_metrics in metrics.items():
        for metric_name in metrics_to_plot:
            if metric_name in model_metrics:
                data.append({
                    "Modelo": model_name,
                    "Métrica": metric_name,
                    "Valor": model_metrics[metric_name]
                })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x="Métrica", y="Valor", color="Modelo",
        barmode="group",
        color_discrete_map=create_model_colors_map(),
        title=title
    )
    
    fig.update_layout(**get_dark_layout(), height=400)
    return fig


def plot_training_times(benchmark: Dict, title: str = "Tiempos de Entrenamiento"):
    """
    Crea gráfico de tiempos de entrenamiento.
    
    Args:
        benchmark: Datos del benchmark con tiempos
    """
    data = []
    for name, info in benchmark.get("recommenders", {}).items():
        if info.get("available"):
            data.append({
                "Modelo": name,
                "Tiempo (s)": info.get("training_time", 0)
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x="Modelo", y="Tiempo (s)", color="Modelo",
        color_discrete_map=create_model_colors_map(),
        text="Tiempo (s)",
        title=title
    )
    
    fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig.update_layout(**get_dark_layout(), height=300, showlegend=False)
    return fig


def plot_genre_predictions(predictions: Dict[str, float], title: str = "Géneros Predichos"):
    """
    Crea gráfico de barras horizontal para predicciones de género.
    
    Args:
        predictions: {género: probabilidad}
    """
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
        **get_dark_layout(),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        title=title
    )
    
    return fig


def plot_classifier_comparison(
    results: Dict[str, Dict], 
    title: str = "Comparación de Clasificadores"
):
    """
    Crea gráfico comparativo de clasificadores.
    
    Args:
        results: {clasificador: {métrica: valor}}
    """
    colors = {
        "Logistic Regression": "#FF9800",
        "Naive Bayes": "#9C27B0",
        "Random Forest": "#4CAF50"
    }
    
    data = []
    for clf_name, metrics in results.items():
        data.append({"Clasificador": clf_name, "Métrica": "F1 Micro", "Valor": metrics.get("f1_micro", 0)})
        data.append({"Clasificador": clf_name, "Métrica": "F1 Macro", "Valor": metrics.get("f1_macro", 0)})
        data.append({"Clasificador": clf_name, "Métrica": "Precision", "Valor": metrics.get("precision", 0)})
        data.append({"Clasificador": clf_name, "Métrica": "Recall", "Valor": metrics.get("recall", 0)})
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x="Métrica", y="Valor", color="Clasificador",
        barmode="group",
        color_discrete_map=colors,
        title=title
    )
    
    fig.update_layout(**get_dark_layout(), height=400, yaxis=dict(tickformat='.0%'))
    return fig


def plot_clustering_scatter(
    x: List[float], 
    y: List[float], 
    labels: List[int], 
    titles: List[str],
    genres: List[str] = None,
    method: str = "UMAP",
    title: str = None
):
    """
    Crea scatter plot de clustering.
    
    Args:
        x, y: Coordenadas 2D
        labels: Etiquetas de cluster
        titles: Títulos de películas
        genres: Géneros opcionales
        method: Método de reducción usado
    """
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'cluster': labels,
        'title': titles,
        'genre': genres if genres else ['N/A'] * len(titles)
    })
    
    fig = px.scatter(
        df,
        x='x', y='y',
        color='cluster',
        hover_name='title',
        hover_data={'genre': True, 'cluster': True, 'x': False, 'y': False},
        title=title or f"Películas agrupadas ({method.upper()})",
        color_continuous_scale='rainbow'
    )
    
    fig.update_layout(**get_dark_layout(), height=600, showlegend=True)
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    return fig
