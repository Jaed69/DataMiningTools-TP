"""
Módulo src del sistema de recomendación Netflix AI.
===================================================
Estructura modular y centralizada.
"""
from .data_loader import DataLoader
from .engine import MultiModelEngine
from .metrics import RecommenderMetrics, ClassifierMetrics, BenchmarkRunner
from .config import RECOMMENDER_MODELS, CLASSIFIER_MODELS, UI_CONFIG
from .evaluation import ModelEvaluator

__all__ = [
    # Core
    'DataLoader',
    'MultiModelEngine', 
    # Métricas
    'RecommenderMetrics',
    'ClassifierMetrics',
    'BenchmarkRunner',
    # Evaluación
    'ModelEvaluator',
    # Configuración
    'RECOMMENDER_MODELS',
    'CLASSIFIER_MODELS',
    'UI_CONFIG',
]
