"""
Modelo de recomendación basado en Sentence-Transformers (SBERT).
Estado del arte en similitud semántica.
"""
import numpy as np
import pandas as pd
import time
import os
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommenderModel

# Intentar importar sentence-transformers (opcional)
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Directorio para cache de embeddings
EMBEDDINGS_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_cache")


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
        use_gpu: bool = False,
        cache_embeddings: bool = True
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
        self.cache_embeddings = cache_embeddings
        self.model = None
        self.titles = []
        self.data = None
        self.doc_vectors = None
        self.similarity_matrix = None
        
    def _get_cache_path(self) -> str:
        """Retorna la ruta del archivo de cache de embeddings."""
        model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
        return os.path.join(EMBEDDINGS_CACHE_DIR, f"sbert_embeddings_{model_safe_name}.npy")
    
    def _get_metadata_path(self) -> str:
        """Retorna la ruta del archivo de metadatos de embeddings."""
        model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
        return os.path.join(EMBEDDINGS_CACHE_DIR, f"sbert_embeddings_{model_safe_name}_meta.json")
    
    def _load_cached_embeddings(self, expected_size: int) -> Optional[np.ndarray]:
        """Intenta cargar embeddings cacheados."""
        cache_path = self._get_cache_path()
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            embeddings = np.load(cache_path)
            if embeddings.shape[0] == expected_size:
                print(f"✓ Embeddings cargados desde cache: {cache_path}")
                return embeddings
            else:
                print(f"⚠️ Cache tiene tamaño diferente ({embeddings.shape[0]} vs {expected_size}), recalculando...")
                return None
        except Exception as e:
            print(f"⚠️ Error cargando cache: {e}")
            return None
    
    def _save_embeddings_to_cache(self, embeddings: np.ndarray) -> None:
        """Guarda embeddings en cache."""
        if not self.cache_embeddings:
            return
            
        cache_path = self._get_cache_path()
        
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"✓ Embeddings guardados en cache: {cache_path}")
        except Exception as e:
            print(f"⚠️ Error guardando cache: {e}")
        
    def fit(self, texts: List[str], titles: List[str], data: pd.DataFrame = None) -> None:
        """Entrena (genera embeddings) con SBERT."""
        if not self.available:
            raise ImportError("sentence-transformers no está instalado. Ejecuta: pip install sentence-transformers")
            
        start_time = time.time()
        
        self.titles = titles
        self.data = data
        self.indices = pd.Series(range(len(titles)), index=titles).drop_duplicates()
        
        # Intentar cargar embeddings cacheados
        cached_embeddings = self._load_cached_embeddings(len(texts))
        
        if cached_embeddings is not None:
            self.doc_vectors = cached_embeddings
            self.embeddings = self.doc_vectors
            # Solo necesitamos cargar el modelo si se van a generar nuevos embeddings
            print(f"Cargando modelo SBERT: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
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
            
            # Guardar en cache
            self._save_embeddings_to_cache(self.doc_vectors)
        
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
