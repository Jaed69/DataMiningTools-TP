"""
Motor principal del sistema de recomendación.
Gestiona múltiples modelos y proporciona una interfaz unificada.
"""
import os
import pandas as pd
import time
from typing import Dict, List, Tuple, Any, Optional
from src.data_loader import DataLoader
from src.models import (
    TFIDFRecommender, 
    Doc2VecRecommender, 
    SBERTRecommender,
    TFIDFClassifier,
    EnsembleClassifier,
    GENSIM_AVAILABLE,
    SBERT_AVAILABLE
)


class MultiModelEngine:
    """
    Motor multi-modelo para recomendación y clasificación.
    Permite comparar diferentes algoritmos de NLP.
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa el motor con los datos.
        
        Args:
            data_path: Ruta al archivo CSV de Netflix
        """
        print("🚀 Iniciando MultiModelEngine...")
        
        # Cargar datos
        self.loader = DataLoader(data_path)
        self.data = self.loader.get_processed_data()
        self.titles = self.data['title'].tolist()
        self.texts = self.data['processed_description'].tolist()
        self.genres = self.data['genres_list'].tolist()
        
        # Diccionarios de modelos
        self.recommender_models: Dict[str, Any] = {}
        self.classifier_models: Dict[str, Any] = {}
        
        # Estado
        self.active_recommender: Optional[str] = None
        self.active_classifier: Optional[str] = None
        
        # Inicializar modelos disponibles
        self._init_models()
        
    def _init_models(self):
        """Inicializa los modelos disponibles."""
        # Siempre disponible: TF-IDF
        self.recommender_models["TF-IDF"] = {
            "model": TFIDFRecommender(),
            "trained": False,
            "available": True,
            "description": "Modelo clásico basado en frecuencia de términos. Rápido y eficiente."
        }
        
        # Doc2Vec (requiere gensim)
        self.recommender_models["Doc2Vec"] = {
            "model": Doc2VecRecommender() if GENSIM_AVAILABLE else None,
            "trained": False,
            "available": GENSIM_AVAILABLE,
            "description": "Embeddings de documentos. Captura semántica global."
        }
        
        # SBERT (requiere sentence-transformers)
        self.recommender_models["SBERT"] = {
            "model": SBERTRecommender() if SBERT_AVAILABLE else None,
            "trained": False,
            "available": SBERT_AVAILABLE,
            "description": "Estado del arte en comprensión semántica con Transformers."
        }
        
        # Clasificadores
        self.classifier_models["Logistic"] = {
            "model": TFIDFClassifier(classifier_type="logistic"),
            "trained": False,
            "available": True,
            "description": "Regresión Logística multietiqueta. Rápido y preciso."
        }
        
        self.classifier_models["NaiveBayes"] = {
            "model": TFIDFClassifier(classifier_type="naive_bayes"),
            "trained": False,
            "available": True,
            "description": "Naive Bayes multinomial. Excelente para texto."
        }
        
        self.classifier_models["RandomForest"] = {
            "model": TFIDFClassifier(classifier_type="random_forest"),
            "trained": False,
            "available": True,
            "description": "Random Forest. Robusto ante ruido."
        }
        
    def get_available_recommenders(self) -> Dict[str, Dict]:
        """Retorna información de los recomendadores disponibles."""
        return {
            name: {
                "available": info["available"],
                "trained": info["trained"],
                "description": info["description"]
            }
            for name, info in self.recommender_models.items()
        }
    
    def get_available_classifiers(self) -> Dict[str, Dict]:
        """Retorna información de los clasificadores disponibles."""
        return {
            name: {
                "available": info["available"],
                "trained": info["trained"],
                "description": info["description"]
            }
            for name, info in self.classifier_models.items()
        }
    
    def train_recommender(self, model_name: str) -> Dict[str, Any]:
        """
        Entrena un modelo de recomendación específico.
        
        Returns:
            Información del entrenamiento
        """
        if model_name not in self.recommender_models:
            raise ValueError(f"Modelo '{model_name}' no encontrado.")
            
        model_info = self.recommender_models[model_name]
        
        if not model_info["available"]:
            raise ImportError(f"Modelo '{model_name}' no está disponible. Instala las dependencias requeridas.")
            
        if model_info["trained"]:
            return {"status": "already_trained", "model": model_name}
            
        print(f"🔄 Entrenando modelo {model_name}...")
        start = time.time()
        
        model = model_info["model"]
        model.fit(self.texts, self.titles, self.data)
        
        model_info["trained"] = True
        self.active_recommender = model_name
        
        elapsed = time.time() - start
        print(f"✅ {model_name} entrenado en {elapsed:.2f}s")
        
        return {
            "status": "trained",
            "model": model_name,
            "training_time": elapsed,
            "info": model.get_info()
        }
    
    def train_classifier(self, model_name: str) -> Dict[str, Any]:
        """Entrena un modelo de clasificación específico."""
        if model_name not in self.classifier_models:
            raise ValueError(f"Clasificador '{model_name}' no encontrado.")
            
        model_info = self.classifier_models[model_name]
        
        if model_info["trained"]:
            return {"status": "already_trained", "model": model_name}
            
        print(f"🔄 Entrenando clasificador {model_name}...")
        start = time.time()
        
        model = model_info["model"]
        model.fit(self.texts, self.genres)
        
        model_info["trained"] = True
        self.active_classifier = model_name
        
        elapsed = time.time() - start
        print(f"✅ {model_name} entrenado en {elapsed:.2f}s")
        
        return {
            "status": "trained",
            "model": model_name,
            "training_time": elapsed,
            "info": model.get_info()
        }
    
    def train_all_recommenders(self) -> Dict[str, Any]:
        """Entrena todos los modelos de recomendación disponibles."""
        results = {}
        for name, info in self.recommender_models.items():
            if info["available"]:
                try:
                    results[name] = self.train_recommender(name)
                except Exception as e:
                    results[name] = {"status": "error", "error": str(e)}
        return results
    
    def train_all_classifiers(self) -> Dict[str, Any]:
        """Entrena todos los clasificadores disponibles."""
        results = {}
        for name in self.classifier_models:
            try:
                results[name] = self.train_classifier(name)
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results
    
    def recommend(
        self, 
        title: str, 
        model_name: Optional[str] = None, 
        top_k: int = 5
    ) -> List[Tuple[str, float, str, str]]:
        """
        Genera recomendaciones usando un modelo específico.
        
        Args:
            title: Título de referencia
            model_name: Nombre del modelo (usa el activo si no se especifica)
            top_k: Número de recomendaciones
            
        Returns:
            Lista de (título, score, género, descripción)
        """
        model_name = model_name or self.active_recommender
        
        if not model_name:
            raise ValueError("No hay modelo de recomendación activo.")
            
        model_info = self.recommender_models.get(model_name)
        
        if not model_info or not model_info["trained"]:
            raise ValueError(f"Modelo '{model_name}' no está entrenado.")
            
        return model_info["model"].recommend(title, top_k)
    
    def classify(
        self, 
        text: str, 
        model_name: Optional[str] = None, 
        top_k: int = 3
    ) -> Dict[str, float]:
        """Clasifica géneros de un texto."""
        model_name = model_name or self.active_classifier
        
        if not model_name:
            raise ValueError("No hay clasificador activo.")
            
        model_info = self.classifier_models.get(model_name)
        
        if not model_info or not model_info["trained"]:
            raise ValueError(f"Clasificador '{model_name}' no está entrenado.")
            
        # Limpiar texto antes de clasificar
        cleaned_text = self.loader.clean_text(text)
        return model_info["model"].predict(cleaned_text, top_k)
    
    def compare_recommenders(
        self, 
        title: str, 
        top_k: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compara recomendaciones de todos los modelos entrenados.
        
        Returns:
            Diccionario con resultados por modelo
        """
        results = {}
        
        for name, info in self.recommender_models.items():
            if info["trained"]:
                start = time.time()
                try:
                    recommendations = info["model"].recommend(title, top_k)
                    inference_time = time.time() - start
                    
                    results[name] = {
                        "recommendations": recommendations,
                        "inference_time": inference_time,
                        "training_time": info["model"].training_time,
                        "description": info["description"]
                    }
                except Exception as e:
                    results[name] = {"error": str(e)}
                    
        return results
    
    def compare_classifiers(
        self, 
        text: str, 
        top_k: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Compara clasificaciones de todos los modelos entrenados."""
        results = {}
        cleaned_text = self.loader.clean_text(text)
        
        for name, info in self.classifier_models.items():
            if info["trained"]:
                start = time.time()
                try:
                    predictions = info["model"].predict(cleaned_text, top_k)
                    inference_time = time.time() - start
                    
                    results[name] = {
                        "predictions": predictions,
                        "inference_time": inference_time,
                        "training_time": info["model"].training_time,
                        "description": info["description"]
                    }
                except Exception as e:
                    results[name] = {"error": str(e)}
                    
        return results
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str, str]]:
        """
        Búsqueda semántica (solo disponible con SBERT).
        """
        if not SBERT_AVAILABLE:
            raise ImportError("SBERT no está disponible para búsqueda semántica.")
            
        sbert_info = self.recommender_models.get("SBERT")
        if not sbert_info or not sbert_info["trained"]:
            raise ValueError("SBERT no está entrenado. Entrena primero con train_recommender('SBERT').")
            
        return sbert_info["model"].semantic_search(query, top_k)
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """
        Genera un reporte de benchmark de todos los modelos.
        """
        report = {
            "recommenders": {},
            "classifiers": {},
            "system_info": {
                "total_titles": len(self.titles),
                "gensim_available": GENSIM_AVAILABLE,
                "sbert_available": SBERT_AVAILABLE
            }
        }
        
        for name, info in self.recommender_models.items():
            if info["trained"]:
                model = info["model"]
                report["recommenders"][name] = {
                    "training_time": model.training_time,
                    "info": model.get_info()
                }
                
        for name, info in self.classifier_models.items():
            if info["trained"]:
                model = info["model"]
                report["classifiers"][name] = {
                    "training_time": model.training_time,
                    "info": model.get_info()
                }
                
        return report
