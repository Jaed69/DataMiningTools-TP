"""
Modelo de recomendación basado en Doc2Vec.
Captura la semántica de documentos completos.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommenderModel

# Intentar importar gensim (opcional)
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


class Doc2VecRecommender(BaseRecommenderModel):
    """
    Modelo de recomendación usando Doc2Vec.
    
    Ventajas:
    - Captura semántica de documentos completos
    - Genera embeddings densos
    - Puede inferir vectores para documentos nuevos
    
    Limitaciones:
    - Requiere más tiempo de entrenamiento
    - Sensible a hiperparámetros
    """
    
    def __init__(
        self, 
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40,
        dm: int = 1  # 1 = PV-DM, 0 = PV-DBOW
    ):
        super().__init__(
            name="Doc2Vec",
            description="Modelo de embeddings de documentos. Captura la esencia semántica de cada sinopsis."
        )
        
        if not GENSIM_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.model = None
        self.titles = []
        self.data = None
        self.doc_vectors = None
        
    def fit(self, texts: List[str], titles: List[str], data: pd.DataFrame = None) -> None:
        """Entrena el modelo Doc2Vec."""
        if not self.available:
            raise ImportError("Gensim no está instalado. Ejecuta: pip install gensim")
            
        start_time = time.time()
        
        self.titles = titles
        self.data = data
        self.indices = pd.Series(range(len(titles)), index=titles).drop_duplicates()
        
        # Preparar documentos etiquetados
        tagged_docs = [
            TaggedDocument(words=text.split(), tags=[str(i)])
            for i, text in enumerate(texts)
        ]
        
        # Crear y entrenar modelo
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            workers=4
        )
        
        self.model.build_vocab(tagged_docs)
        self.model.train(tagged_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        # Almacenar vectores de documentos
        self.doc_vectors = np.array([self.model.dv[str(i)] for i in range(len(texts))])
        self.embeddings = self.doc_vectors
        
        # Calcular matriz de similitud
        self.similarity_matrix = cosine_similarity(self.doc_vectors, self.doc_vectors)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings Doc2Vec para nuevos textos."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        embeddings = []
        for text in texts:
            vector = self.model.infer_vector(text.split())
            embeddings.append(vector)
        return np.array(embeddings)
    
    def recommend(self, title: str, top_k: int = 5) -> List[Tuple[str, float, str, str]]:
        """Recomienda títulos similares usando Doc2Vec."""
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
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información detallada del modelo."""
        info = super().get_info()
        info.update({
            "available": self.available,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "dm_mode": "PV-DM" if self.dm == 1 else "PV-DBOW",
            "algorithm": "Doc2Vec (Paragraph Vectors)",
            "complexity": "O(n × epochs × vocab_size)"
        })
        return info
