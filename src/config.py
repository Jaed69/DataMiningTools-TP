"""
Configuración centralizada del proyecto Netflix AI.
===================================================
Todas las constantes, rutas y configuraciones en un solo lugar.
"""
import os

# =============================================================================
# RUTAS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_cache")

# Archivos de datos
CSV_PATH = os.path.join(DATA_DIR, "netflix_titles.csv")
CSV_CLEAN_PATH = os.path.join(DATA_DIR, "netflix_titles_clean.csv")

# =============================================================================
# MODELOS DISPONIBLES
# =============================================================================
RECOMMENDER_MODELS = {
    "TF-IDF": {
        "key": "tfidf_recommender",
        "color": "#4CAF50",
        "icon": "🟢",
        "description": "Búsqueda por palabras exactas (rápido)",
        "pros": ["Muy rápido", "Bajo uso de memoria", "Fácil de interpretar"],
        "cons": ["No entiende sinónimos", "Solo coincidencias exactas"]
    },
    "Doc2Vec": {
        "key": "doc2vec_recommender", 
        "color": "#2196F3",
        "icon": "🔵",
        "description": "Detecta patrones de escritura",
        "pros": ["Captura contexto", "Balance velocidad/precisión"],
        "cons": ["Requiere entrenamiento", "Menos preciso que SBERT"]
    },
    "SBERT": {
        "key": "sbert_recommender",
        "color": "#e50914",
        "icon": "🔴", 
        "description": "Comprensión semántica profunda",
        "pros": ["Mejor precisión", "Entiende sinónimos", "Pre-entrenado"],
        "cons": ["Más lento", "Requiere más memoria", "Necesita GPU para óptimo"]
    },
    "BM25": {
        "key": "bm25_recommender",
        "color": "#8BC34A",
        "icon": "🟡",
        "description": "Alternativa mejorada a TF-IDF",
        "pros": ["Mejor que TF-IDF", "Rápido", "No requiere entrenamiento pesado"],
        "cons": ["No entiende semántica"]
    }
}

CLASSIFIER_MODELS = {
    "Logistic": {
        "key": "logistic_classifier",
        "color": "#FF9800",
        "description": "Regresión Logística One-vs-Rest"
    },
    "NaiveBayes": {
        "key": "naive_bayes_classifier",
        "color": "#9C27B0",
        "description": "Naive Bayes Multinomial"
    },
    "RandomForest": {
        "key": "random_forest_classifier",
        "color": "#4CAF50",
        "description": "Random Forest Ensemble"
    }
}

# =============================================================================
# CONFIGURACIÓN DE UI
# =============================================================================
UI_CONFIG = {
    "page_title": "Netflix AI - Recomendador NLP",
    "page_icon": "🎬",
    "layout": "wide",
    "primary_color": "#e50914",
    "background_color": "#0d0d0d",
    "secondary_bg": "#1a1a2e"
}

# =============================================================================
# TABS DE LA APLICACIÓN (SIMPLIFICADOS)
# =============================================================================
TABS = [
    {"name": "🎬 Recomendador", "key": "recommender"},
    {"name": "🔎 Búsqueda", "key": "search"},
    {"name": "🏷️ Clasificador", "key": "classifier"},
    {"name": "📊 Evaluación", "key": "evaluation"},
    {"name": "📖 Info", "key": "info"}
]

# =============================================================================
# MÉTRICAS DE EVALUACIÓN
# =============================================================================
EVALUATION_CONFIG = {
    "default_k": 5,
    "default_test_size": 30,
    "default_seed": 42,
    "metrics": ["Precision@K", "Recall@K", "nDCG@K", "MAP", "Genre Diversity"]
}

# =============================================================================
# ESTILOS CSS
# =============================================================================
CSS_STYLES = """
<style>
    /* Fondo oscuro estilo Netflix */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%);
    }
    
    /* Header */
    .netflix-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid #333;
        margin-bottom: 1.5rem;
    }
    
    .netflix-title {
        color: #e50914;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 3px;
        margin: 0;
    }
    
    .netflix-subtitle {
        color: #b3b3b3;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    /* Cards de recomendación */
    .rec-card {
        background: linear-gradient(145deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e50914;
    }
    
    .rec-title {
        color: white;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .rec-score {
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .rec-genre {
        color: #8c8c8c;
        font-size: 0.85rem;
        margin-top: 3px;
    }
    
    /* Metric cards */
    .metric-card {
        background: #252525;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
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
    }
    
    .stTabs [aria-selected="true"] {
        background: #e50914 !important;
        color: white !important;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e50914;
        margin: 0.5rem 0;
    }
</style>
"""

def get_model_color(model_name: str) -> str:
    """Obtiene el color asociado a un modelo."""
    if model_name in RECOMMENDER_MODELS:
        return RECOMMENDER_MODELS[model_name]["color"]
    if model_name in CLASSIFIER_MODELS:
        return CLASSIFIER_MODELS[model_name]["color"]
    return "#666666"

def get_model_icon(model_name: str) -> str:
    """Obtiene el icono asociado a un modelo."""
    return RECOMMENDER_MODELS.get(model_name, {}).get("icon", "⚪")
