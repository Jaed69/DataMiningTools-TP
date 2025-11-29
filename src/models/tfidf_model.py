"""
Modelo de recomendación basado en TF-IDF.
Es el modelo baseline, rápido y efectivo para comparaciones.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommenderModel


class TFIDFRecommender(BaseRecommenderModel):
    """
    Modelo de recomendación usando TF-IDF + Similitud del Coseno.
    
    Ventajas:
    - Rápido de entrenar
    - No requiere librerías externas pesadas
    - Buen baseline
    
    Limitaciones:
    - No captura semántica profunda
    - Basado solo en frecuencia de palabras
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        super().__init__(
            name="TF-IDF",
            description="Modelo clásico basado en frecuencia de términos. Rápido y eficiente como baseline."
        )
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.similarity_matrix = None
        self.titles = []
        self.data = None
        
    def fit(self, texts: List[str], titles: List[str], data: pd.DataFrame = None) -> None:
        """Entrena el modelo TF-IDF."""
        start_time = time.time()
        
        self.titles = titles
        self.data = data
        self.indices = pd.Series(range(len(titles)), index=titles).drop_duplicates()
        
        # Vectorización TF-IDF
        self.embeddings = self.vectorizer.fit_transform(texts)
        
        # Matriz de similitud
        self.similarity_matrix = cosine_similarity(self.embeddings, self.embeddings)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings TF-IDF para nuevos textos."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        return self.vectorizer.transform(texts).toarray()
    
    def recommend(self, title: str, top_k: int = 5) -> List[Tuple[str, float, str, str]]:
        """
        Recomienda títulos similares.
        Returns: Lista de tuplas (título, score, género, descripción)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
            
        if title not in self.indices:
            return []
            
        idx = self.indices[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k + 1]  # Excluir el propio título
        
        results = []
        for i, score in sim_scores:
            rec_title = self.titles[i]
            genre = ""
            desc = ""
            if self.data is not None:
                row = self.data.iloc[i]
                genre = row.get('listed_in', '')
                desc = row.get('description', '')
            results.append((rec_title, float(score), genre, desc))
            
        return results
    
    def get_similarity_score(self, title1: str, title2: str) -> float:
        """Obtiene el score de similitud entre dos títulos."""
        if title1 not in self.indices or title2 not in self.indices:
            return 0.0
        idx1 = self.indices[title1]
        idx2 = self.indices[title2]
        return float(self.similarity_matrix[idx1][idx2])
    
    def explain_recommendation(self, source_title: str, recommended_title: str, top_terms: int = 5) -> Dict[str, Any]:
        """
        Explica por qué se recomienda un título.
        Muestra las palabras clave en común y su importancia.
        """
        if source_title not in self.indices or recommended_title not in self.indices:
            return {"error": "Título no encontrado"}
        
        idx1 = self.indices[source_title]
        idx2 = self.indices[recommended_title]
        
        # Obtener vectores TF-IDF
        vec1 = self.embeddings[idx1].toarray().flatten()
        vec2 = self.embeddings[idx2].toarray().flatten()
        
        # Palabras del vocabulario
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Encontrar términos en común (ambos tienen peso > 0)
        common_mask = (vec1 > 0) & (vec2 > 0)
        common_indices = np.where(common_mask)[0]
        
        # Calcular importancia combinada
        common_terms = []
        for idx in common_indices:
            term = feature_names[idx]
            importance = (vec1[idx] + vec2[idx]) / 2
            common_terms.append((term, float(importance)))
        
        # Ordenar por importancia
        common_terms.sort(key=lambda x: x[1], reverse=True)
        
        # Top términos del source
        source_top = [(feature_names[i], float(vec1[i])) for i in np.argsort(vec1)[-top_terms:][::-1] if vec1[i] > 0]
        
        # Top términos del recommended
        rec_top = [(feature_names[i], float(vec2[i])) for i in np.argsort(vec2)[-top_terms:][::-1] if vec2[i] > 0]
        
        return {
            "similarity_score": float(self.similarity_matrix[idx1][idx2]),
            "common_terms": common_terms[:top_terms],
            "source_keywords": source_top,
            "recommended_keywords": rec_top,
            "total_common_terms": len(common_terms),
            "explanation": f"Comparten {len(common_terms)} términos. Los más importantes: {', '.join([t[0] for t in common_terms[:3]])}"
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información detallada del modelo."""
        info = super().get_info()
        info.update({
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.is_trained else 0,
            "algorithm": "TF-IDF + Cosine Similarity",
            "complexity": "O(n²) para matriz de similitud"
        })
        return info
