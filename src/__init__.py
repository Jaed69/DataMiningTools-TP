"""
Módulo src del sistema de recomendación Netflix AI.
"""
from .data_loader import DataLoader
from .engine import MultiModelEngine
from .metrics import RecommenderMetrics, ClassifierMetrics, BenchmarkRunner

__all__ = [
    'DataLoader',
    'MultiModelEngine', 
    'RecommenderMetrics',
    'ClassifierMetrics',
    'BenchmarkRunner'
]
