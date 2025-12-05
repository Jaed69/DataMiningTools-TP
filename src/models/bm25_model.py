"""
Modelo de recomendación basado en BM25 (Okapi BM25).
Alternativa mejorada a TF-IDF para búsqueda de información.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Any, Optional
from .base_model import BaseRecommenderModel

# Intentar importar rank_bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25Recommender(BaseRecommenderModel):
    """
    Modelo de recomendación usando BM25 (Okapi BM25).
    
    BM25 es una mejora sobre TF-IDF que:
    - Normaliza por longitud del documento
    - Tiene saturación de términos (evita sobreponderar términos muy frecuentes)
    - Parámetros k1 y b para ajustar comportamiento
    
    Ventajas:
    - Mejor que TF-IDF para búsqueda de información
    - Considera longitud del documento
    - Parámetros ajustables
    
    Limitaciones:
    - Requiere librería rank_bm25
    - No captura semántica profunda
    - Basado en coincidencia de términos
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Parámetro de saturación de término (1.2-2.0 típico)
            b: Parámetro de normalización por longitud (0.75 típico)
        """
        super().__init__(
            name="BM25",
            description="Okapi BM25 - Mejora de TF-IDF con normalización por longitud de documento."
        )
        
        if not BM25_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = None
        self.titles = []
        self.data = None
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza un texto en palabras."""
        if not isinstance(text, str):
            return []
        return text.lower().split()
    
    def fit(self, texts: List[str], titles: List[str], data: pd.DataFrame = None) -> None:
        """Entrena el modelo BM25."""
        if not self.available:
            raise ImportError("rank_bm25 no está instalado. Ejecuta: pip install rank-bm25")
            
        start_time = time.time()
        
        self.titles = titles
        self.data = data
        self.indices = pd.Series(range(len(titles)), index=titles).drop_duplicates()
        
        # Tokenizar corpus
        print("Tokenizando corpus para BM25...")
        self.tokenized_corpus = [self._tokenize(text) for text in texts]
        
        # Crear modelo BM25
        print("Construyendo índice BM25...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"✓ BM25 listo en {self.training_time:.2f}s")
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        BM25 no genera embeddings densos.
        Retorna None para mantener compatibilidad.
        """
        return None
    
    def recommend(self, title: str, top_k: int = 5) -> List[Tuple[str, float, str, str]]:
        """
        Recomienda títulos similares usando BM25.
        
        Args:
            title: Título de referencia
            top_k: Número de recomendaciones
            
        Returns:
            Lista de tuplas (título, score, género, descripción)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
            
        if title not in self.indices:
            return []
            
        idx = self.indices[title]
        
        # Obtener tokens del documento de consulta
        query_tokens = self.tokenized_corpus[idx]
        
        # Calcular scores BM25 contra todo el corpus
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalizar scores a [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1
        scores_normalized = scores / max_score
        
        # Obtener top-k (excluyendo el propio documento)
        scores_normalized[idx] = -1  # Excluir self
        top_indices = np.argsort(scores_normalized)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            rec_title = self.titles[i]
            score = float(scores_normalized[i])
            genre = ""
            desc = ""
            if self.data is not None:
                row = self.data.iloc[i]
                genre = row.get('listed_in', '')
                desc = row.get('description', '')
            results.append((rec_title, score, genre, desc))
            
        return results
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str, str]]:
        """
        Búsqueda por texto libre usando BM25.
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            
        Returns:
            Lista de tuplas (título, score, género, descripción)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
            
        # Tokenizar query
        query_tokens = self._tokenize(query)
        
        # Calcular scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalizar
        max_score = max(scores) if max(scores) > 0 else 1
        scores_normalized = scores / max_score
        
        # Top-k
        top_indices = np.argsort(scores_normalized)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            if scores_normalized[i] > 0:
                rec_title = self.titles[i]
                score = float(scores_normalized[i])
                genre = ""
                desc = ""
                if self.data is not None:
                    row = self.data.iloc[i]
                    genre = row.get('listed_in', '')
                    desc = row.get('description', '')
                results.append((rec_title, score, genre, desc))
            
        return results
    
    def explain_recommendation(
        self, 
        source_title: str, 
        recommended_title: str, 
        top_terms: int = 5
    ) -> Dict[str, Any]:
        """
        Explica por qué se recomienda un título basándose en términos compartidos.
        """
        if source_title not in self.indices or recommended_title not in self.indices:
            return {"error": "Título no encontrado"}
        
        idx1 = self.indices[source_title]
        idx2 = self.indices[recommended_title]
        
        tokens1 = set(self.tokenized_corpus[idx1])
        tokens2 = set(self.tokenized_corpus[idx2])
        
        # Términos en común
        common_terms = tokens1.intersection(tokens2)
        
        # Calcular importancia de cada término común usando IDF aproximado
        doc_freq = {}
        for tokens in self.tokenized_corpus:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        
        n_docs = len(self.tokenized_corpus)
        term_importance = []
        for term in common_terms:
            idf = np.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
            term_importance.append((term, float(idf)))
        
        term_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Calcular score
        query_tokens = self.tokenized_corpus[idx1]
        scores = self.bm25.get_scores(query_tokens)
        similarity_score = scores[idx2] / max(scores) if max(scores) > 0 else 0
        
        return {
            "similarity_score": float(similarity_score),
            "common_terms": term_importance[:top_terms],
            "total_common_terms": len(common_terms),
            "source_unique_terms": len(tokens1 - tokens2),
            "recommended_unique_terms": len(tokens2 - tokens1),
            "algorithm": "BM25 (Okapi)",
            "parameters": {"k1": self.k1, "b": self.b},
            "explanation": f"Comparten {len(common_terms)} términos. Los más importantes: {', '.join([t[0] for t in term_importance[:3]])}"
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información detallada del modelo."""
        info = super().get_info()
        info.update({
            "available": self.available,
            "k1": self.k1,
            "b": self.b,
            "corpus_size": len(self.tokenized_corpus) if self.tokenized_corpus else 0,
            "avg_doc_length": np.mean([len(t) for t in self.tokenized_corpus]) if self.tokenized_corpus else 0,
            "algorithm": "Okapi BM25",
            "complexity": "O(n × avg_doc_length)"
        })
        return info
