"""
Módulo de modelos NLP para el sistema de recomendación.
"""
from .base_model import BaseRecommenderModel, BaseClassifierModel, ModelBenchmark
from .tfidf_model import TFIDFRecommender
from .doc2vec_model import Doc2VecRecommender, GENSIM_AVAILABLE
from .sbert_model import SBERTRecommender, SBERT_AVAILABLE
from .classifier_models import TFIDFClassifier, EnsembleClassifier

# Nuevos modelos
try:
    from .bm25_model import BM25Recommender, BM25_AVAILABLE
except ImportError:
    BM25_AVAILABLE = False
    BM25Recommender = None

try:
    from .cross_encoder import CrossEncoderReranker, CROSSENCODER_AVAILABLE
except ImportError:
    CROSSENCODER_AVAILABLE = False
    CrossEncoderReranker = None

__all__ = [
    'BaseRecommenderModel',
    'BaseClassifierModel', 
    'ModelBenchmark',
    'TFIDFRecommender',
    'Doc2VecRecommender',
    'SBERTRecommender',
    'BM25Recommender',
    'CrossEncoderReranker',
    'TFIDFClassifier',
    'EnsembleClassifier',
    'GENSIM_AVAILABLE',
    'SBERT_AVAILABLE',
    'BM25_AVAILABLE',
    'CROSSENCODER_AVAILABLE'
]
