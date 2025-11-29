"""
Modelo de recomendación basado en Sentence-Transformers (SBERT).
Estado del arte en similitud semántica.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommenderModel

# Intentar importar sentence-transformers (opcional)
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


class SBERTRecommender(BaseRecommenderModel):
    """
    Modelo de recomendación usando Sentence-Transformers (SBERT).
    
    Ventajas:
    - Estado del arte en similitud semántica
    - Pre-entrenado en grandes corpus
    - Captura contexto profundo
    
    Limitaciones:
    - Requiere más memoria y GPU (opcional)
    - Primera carga puede ser lenta
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        use_gpu: bool = False
    ):
        super().__init__(
            name="SBERT (Sentence-Transformers)",
            description="Modelo Transformer pre-entrenado. Estado del arte en comprensión semántica."
        )
        
        if not SBERT_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if use_gpu else "cpu"
        self.model = None
        self.titles = []
        self.data = None
        self.doc_vectors = None
        self.similarity_matrix = None
        
    def fit(self, texts: List[str], titles: List[str], data: pd.DataFrame = None) -> None:
        """Entrena (genera embeddings) con SBERT."""
        if not self.available:
            raise ImportError("sentence-transformers no está instalado. Ejecuta: pip install sentence-transformers")
            
        start_time = time.time()
        
        self.titles = titles
        self.data = data
        self.indices = pd.Series(range(len(titles)), index=titles).drop_duplicates()
        
        # Cargar modelo pre-entrenado
        print(f"Cargando modelo SBERT: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Generar embeddings
        print("Generando embeddings semánticos...")
        self.doc_vectors = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.embeddings = self.doc_vectors
        
        # Calcular matriz de similitud
        print("Calculando matriz de similitud...")
        self.similarity_matrix = cosine_similarity(self.doc_vectors, self.doc_vectors)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"✓ SBERT listo en {self.training_time:.2f}s")
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings SBERT para nuevos textos."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
    
    def recommend(self, title: str, top_k: int = 5) -> List[Tuple[str, float, str, str]]:
        """Recomienda títulos similares usando SBERT."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
            
        if title not in self.indices:
            return []
            
        idx = self.indices[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k + 1]
        
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
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str, str]]:
        """
        Búsqueda semántica: encuentra títulos similares a una consulta libre.
        Ejemplo: "película sobre un hombre varado en una isla"
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
            
        # Generar embedding de la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calcular similitud con todos los documentos
        similarities = cosine_similarity(query_embedding, self.doc_vectors)[0]
        
        # Ordenar por similitud
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            rec_title = self.titles[idx]
            score = float(similarities[idx])
            genre = ""
            desc = ""
            if self.data is not None:
                row = self.data.iloc[idx]
                genre = row.get('listed_in', '')
                desc = row.get('description', '')
            results.append((rec_title, score, genre, desc))
            
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información detallada del modelo."""
        info = super().get_info()
        info.update({
            "available": self.available,
            "model_name": self.model_name,
            "embedding_dimension": self.doc_vectors.shape[1] if self.doc_vectors is not None else 0,
            "device": self.device,
            "algorithm": "Sentence-BERT (Transformer)",
            "complexity": "O(n × sequence_length)"
        })
        return info
