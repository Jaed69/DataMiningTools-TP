"""
Cross-Encoder para reranking de recomendaciones.
Mejora la precisión de los top-k resultados.
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional

# Intentar importar sentence-transformers
try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """
    Reranker basado en Cross-Encoder para mejorar la precisión de recomendaciones.
    
    A diferencia de Bi-Encoders (como SBERT) que codifican textos de forma independiente,
    los Cross-Encoders procesan pares de textos juntos, capturando interacciones más finas.
    
    Flujo típico:
    1. Usar un modelo rápido (TF-IDF, BM25, SBERT) para obtener top-100 candidatos
    2. Usar Cross-Encoder para reranquear y obtener top-10 finales
    
    Ventajas:
    - Mayor precisión que Bi-Encoders
    - Captura relaciones sutiles entre pares de textos
    
    Limitaciones:
    - Mucho más lento (no escalable a todo el corpus)
    - Solo para reranking de candidatos pre-filtrados
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Args:
            model_name: Modelo de HuggingFace para Cross-Encoder
            batch_size: Tamaño de batch para inferencia
            max_length: Longitud máxima de tokens
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.is_loaded = False
        
        if not CROSSENCODER_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        
    def load_model(self) -> None:
        """Carga el modelo Cross-Encoder."""
        if not self.available:
            raise ImportError("sentence-transformers no está instalado.")
            
        if self.is_loaded:
            return
            
        print(f"Cargando Cross-Encoder: {self.model_name}...")
        self.model = STCrossEncoder(self.model_name, max_length=self.max_length)
        self.is_loaded = True
        print("✓ Cross-Encoder cargado")
        
    def rerank(
        self,
        query_text: str,
        candidates: List[Tuple[str, float, str, str]],
        candidate_texts: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float, str, str]]:
        """
        Reranquea una lista de candidatos usando Cross-Encoder.
        
        Args:
            query_text: Texto de la consulta (descripción de la película original)
            candidates: Lista de tuplas (título, score_original, género, descripción)
            candidate_texts: Textos de los candidatos para comparar
            top_k: Número de resultados finales
            
        Returns:
            Lista reranqueada de tuplas (título, score_nuevo, género, descripción)
        """
        if not self.available:
            raise ImportError("sentence-transformers no está instalado.")
            
        if not self.is_loaded:
            self.load_model()
            
        if not candidates:
            return []
            
        # Crear pares para el Cross-Encoder
        pairs = [(query_text, cand_text) for cand_text in candidate_texts]
        
        # Obtener scores del Cross-Encoder
        start_time = time.time()
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        inference_time = time.time() - start_time
        
        # Normalizar scores a [0, 1]
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Combinar con candidatos originales
        reranked = []
        for i, (candidate, new_score) in enumerate(zip(candidates, scores_normalized)):
            title, old_score, genre, desc = candidate
            reranked.append((title, float(new_score), genre, desc))
        
        # Ordenar por nuevo score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def rerank_with_details(
        self,
        query_text: str,
        candidates: List[Tuple[str, float, str, str]],
        candidate_texts: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Reranquea y retorna información detallada sobre los cambios.
        
        Returns:
            Diccionario con resultados y métricas de reranking
        """
        if not self.available:
            raise ImportError("sentence-transformers no está instalado.")
            
        if not self.is_loaded:
            self.load_model()
            
        if not candidates:
            return {"reranked": [], "changes": []}
            
        # Guardar ranking original
        original_ranking = {c[0]: i for i, c in enumerate(candidates)}
        
        # Crear pares
        pairs = [(query_text, cand_text) for cand_text in candidate_texts]
        
        # Obtener scores
        start_time = time.time()
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        inference_time = time.time() - start_time
        
        # Normalizar
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Crear lista con scores
        reranked = []
        for i, (candidate, new_score) in enumerate(zip(candidates, scores_normalized)):
            title, old_score, genre, desc = candidate
            reranked.append({
                "title": title,
                "new_score": float(new_score),
                "old_score": float(old_score),
                "genre": genre,
                "description": desc,
                "old_rank": original_ranking[title]
            })
        
        # Ordenar
        reranked.sort(key=lambda x: x["new_score"], reverse=True)
        
        # Calcular cambios de ranking
        changes = []
        for new_rank, item in enumerate(reranked[:top_k]):
            old_rank = item["old_rank"]
            change = old_rank - new_rank  # Positivo = subió, negativo = bajó
            changes.append({
                "title": item["title"],
                "old_rank": old_rank + 1,
                "new_rank": new_rank + 1,
                "rank_change": change,
                "score_improvement": item["new_score"] - item["old_score"]
            })
        
        # Convertir reranked a formato de tuplas
        final_results = [
            (r["title"], r["new_score"], r["genre"], r["description"])
            for r in reranked[:top_k]
        ]
        
        return {
            "reranked": final_results,
            "changes": changes,
            "inference_time_ms": inference_time * 1000,
            "candidates_evaluated": len(candidates),
            "model": self.model_name
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del reranker."""
        return {
            "name": "Cross-Encoder Reranker",
            "model": self.model_name,
            "available": self.available,
            "is_loaded": self.is_loaded,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "description": "Reranquea candidatos usando Cross-Encoder para mayor precisión"
        }
