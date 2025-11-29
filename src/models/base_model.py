"""
Clase base abstracta para todos los modelos de NLP.
Provee una interfaz común para recomendación y clasificación.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np
import time


class BaseRecommenderModel(ABC):
    """Clase base para modelos de recomendación."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_trained = False
        self.training_time = 0.0
        self.embeddings = None
        self.indices = None
        
    @abstractmethod
    def fit(self, texts: List[str], titles: List[str]) -> None:
        """Entrena el modelo con los textos dados."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para nuevos textos."""
        pass
    
    @abstractmethod
    def recommend(self, title: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recomienda títulos similares.
        Returns: Lista de tuplas (título, score de similitud)
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        return {
            "name": self.name,
            "description": self.description,
            "is_trained": self.is_trained,
            "training_time": self.training_time
        }


class BaseClassifierModel(ABC):
    """Clase base para modelos de clasificación."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_trained = False
        self.training_time = 0.0
        self.classes = []
        
    @abstractmethod
    def fit(self, texts: List[str], labels: List[List[str]]) -> None:
        """Entrena el clasificador."""
        pass
    
    @abstractmethod
    def predict(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """
        Predice géneros para un texto.
        Returns: Diccionario {género: probabilidad}
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict[str, float]]:
        """Predice géneros para múltiples textos."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        return {
            "name": self.name,
            "description": self.description,
            "is_trained": self.is_trained,
            "training_time": self.training_time,
            "num_classes": len(self.classes)
        }


class ModelBenchmark:
    """Clase para realizar benchmarks entre modelos."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_recommender(
        self, 
        model: BaseRecommenderModel, 
        test_titles: List[str],
        ground_truth: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Realiza benchmark de un modelo de recomendación.
        """
        results = {
            "model_name": model.name,
            "training_time": model.training_time,
            "inference_times": [],
            "avg_inference_time": 0.0
        }
        
        for title in test_titles:
            start = time.time()
            try:
                _ = model.recommend(title, top_k=5)
                elapsed = time.time() - start
                results["inference_times"].append(elapsed)
            except Exception:
                continue
                
        if results["inference_times"]:
            results["avg_inference_time"] = np.mean(results["inference_times"])
            
        self.results[model.name] = results
        return results
    
    def compare_models(self, models: List[BaseRecommenderModel]) -> Dict[str, Any]:
        """Compara múltiples modelos."""
        comparison = {}
        for model in models:
            comparison[model.name] = {
                "training_time": model.training_time,
                "is_trained": model.is_trained,
                "description": model.description
            }
        return comparison
