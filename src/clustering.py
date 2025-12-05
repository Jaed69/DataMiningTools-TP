"""
Módulo de clustering para análisis exploratorio de películas.
Incluye K-Means, HDBSCAN y visualización con UMAP/t-SNE.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import time

# Intentar importar dependencias opcionales
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    SKLEARN_CLUSTERING_AVAILABLE = True
except ImportError:
    SKLEARN_CLUSTERING_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


class MovieClusterer:
    """
    Clustering de películas basado en embeddings.
    Permite descubrir grupos temáticos automáticamente.
    """
    
    def __init__(self):
        self.embeddings = None
        self.titles = None
        self.data = None
        self.labels = None
        self.cluster_info = {}
        self.reduced_embeddings = None
        self.reduction_method = None
        
    def set_embeddings(
        self, 
        embeddings: np.ndarray, 
        titles: List[str],
        data: pd.DataFrame = None
    ) -> None:
        """
        Configura los embeddings para clustering.
        
        Args:
            embeddings: Matriz de embeddings (n_samples, n_features)
            titles: Lista de títulos correspondientes
            data: DataFrame opcional con metadatos
        """
        self.embeddings = embeddings
        self.titles = titles
        self.data = data
        
    def fit_kmeans(
        self, 
        n_clusters: int = 15, 
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Clustering con K-Means.
        
        Args:
            n_clusters: Número de clusters
            random_state: Semilla para reproducibilidad
            
        Returns:
            Información del clustering
        """
        if not SKLEARN_CLUSTERING_AVAILABLE:
            raise ImportError("scikit-learn no disponible")
            
        if self.embeddings is None:
            raise ValueError("Primero configura los embeddings con set_embeddings()")
            
        print(f"Ejecutando K-Means con {n_clusters} clusters...")
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = kmeans.fit_predict(self.embeddings)
        
        # Calcular silhouette score
        silhouette = silhouette_score(self.embeddings, self.labels)
        
        training_time = time.time() - start_time
        
        self.cluster_info = {
            "method": "K-Means",
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette),
            "training_time": training_time,
            "cluster_sizes": self._get_cluster_sizes(),
            "cluster_centers": kmeans.cluster_centers_
        }
        
        print(f"✓ K-Means completado en {training_time:.2f}s (silhouette: {silhouette:.3f})")
        
        return self.cluster_info
    
    def fit_hdbscan(
        self, 
        min_cluster_size: int = 50,
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Clustering con HDBSCAN (detección automática de clusters).
        
        Args:
            min_cluster_size: Tamaño mínimo de cluster
            min_samples: Mínimo de muestras para core points
            
        Returns:
            Información del clustering
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan no instalado. Ejecuta: pip install hdbscan")
            
        if self.embeddings is None:
            raise ValueError("Primero configura los embeddings")
            
        print(f"Ejecutando HDBSCAN (min_cluster_size={min_cluster_size})...")
        start_time = time.time()
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        self.labels = clusterer.fit_predict(self.embeddings)
        
        # Número de clusters (excluyendo ruido)
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        # Silhouette (solo para muestras con cluster)
        mask = self.labels != -1
        if np.sum(mask) > 1 and n_clusters > 1:
            silhouette = silhouette_score(self.embeddings[mask], self.labels[mask])
        else:
            silhouette = 0.0
        
        training_time = time.time() - start_time
        
        self.cluster_info = {
            "method": "HDBSCAN",
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "silhouette_score": float(silhouette),
            "training_time": training_time,
            "cluster_sizes": self._get_cluster_sizes(),
            "probabilities": clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
        }
        
        print(f"✓ HDBSCAN completado: {n_clusters} clusters, {n_noise} ruido (silhouette: {silhouette:.3f})")
        
        return self.cluster_info
    
    def reduce_dimensions(
        self, 
        method: str = "umap", 
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionalidad para visualización.
        
        Args:
            method: "umap" o "tsne"
            n_components: Número de dimensiones (2 o 3)
            **kwargs: Parámetros adicionales para el método
            
        Returns:
            Embeddings reducidos
        """
        if self.embeddings is None:
            raise ValueError("Primero configura los embeddings")
            
        print(f"Reduciendo dimensionalidad con {method.upper()}...")
        start_time = time.time()
        
        if method.lower() == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("umap-learn no instalado. Ejecuta: pip install umap-learn")
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get("n_neighbors", 15),
                min_dist=kwargs.get("min_dist", 0.1),
                metric=kwargs.get("metric", "cosine"),
                random_state=kwargs.get("random_state", 42)
            )
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
            
        elif method.lower() == "tsne":
            if not TSNE_AVAILABLE:
                raise ImportError("scikit-learn no disponible para t-SNE")
            
            reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get("perplexity", 30),
                learning_rate=kwargs.get("learning_rate", 200),
                random_state=kwargs.get("random_state", 42)
            )
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
            
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        self.reduction_method = method
        print(f"✓ Reducción completada en {time.time() - start_time:.2f}s")
        
        return self.reduced_embeddings
    
    def get_cluster_titles(self, cluster_id: int, max_titles: int = 20) -> List[str]:
        """Obtiene títulos de un cluster específico."""
        if self.labels is None:
            raise ValueError("Primero ejecuta un método de clustering")
            
        mask = self.labels == cluster_id
        indices = np.where(mask)[0][:max_titles]
        
        return [self.titles[i] for i in indices]
    
    def get_cluster_genres(self, cluster_id: int) -> Dict[str, int]:
        """Obtiene distribución de géneros en un cluster."""
        if self.labels is None or self.data is None:
            return {}
            
        mask = self.labels == cluster_id
        cluster_data = self.data.iloc[mask]
        
        genre_counts = {}
        for genres in cluster_data['listed_in'].dropna():
            for genre in str(genres).split(','):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Ordenar por frecuencia
        return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """
        Genera un resumen de todos los clusters.
        
        Returns:
            Lista de diccionarios con info de cada cluster
        """
        if self.labels is None:
            raise ValueError("Primero ejecuta un método de clustering")
            
        summaries = []
        unique_labels = sorted(set(self.labels))
        
        for label in unique_labels:
            if label == -1:
                name = "Ruido/Outliers"
            else:
                # Intentar determinar nombre basado en géneros dominantes
                top_genres = self.get_cluster_genres(label)
                top_genre = list(top_genres.keys())[0] if top_genres else "Mixto"
                name = f"Cluster {label}: {top_genre}"
            
            size = int(np.sum(self.labels == label))
            sample_titles = self.get_cluster_titles(label, max_titles=5)
            
            summaries.append({
                "cluster_id": int(label),
                "name": name,
                "size": size,
                "percentage": size / len(self.labels) * 100,
                "sample_titles": sample_titles,
                "top_genres": dict(list(self.get_cluster_genres(label).items())[:5])
            })
        
        return summaries
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Prepara datos para visualización interactiva.
        
        Returns:
            Diccionario con coordenadas, labels, títulos y metadatos
        """
        if self.reduced_embeddings is None:
            raise ValueError("Primero ejecuta reduce_dimensions()")
            
        data = {
            "x": self.reduced_embeddings[:, 0].tolist(),
            "y": self.reduced_embeddings[:, 1].tolist(),
            "titles": self.titles,
            "labels": self.labels.tolist() if self.labels is not None else [0] * len(self.titles),
            "reduction_method": self.reduction_method
        }
        
        # Añadir z si es 3D
        if self.reduced_embeddings.shape[1] == 3:
            data["z"] = self.reduced_embeddings[:, 2].tolist()
        
        # Añadir metadatos si están disponibles
        if self.data is not None:
            data["genres"] = self.data['listed_in'].tolist()
            data["types"] = self.data['type'].tolist() if 'type' in self.data.columns else None
            data["years"] = self.data['release_year'].tolist() if 'release_year' in self.data.columns else None
        
        return data
    
    def _get_cluster_sizes(self) -> Dict[int, int]:
        """Obtiene tamaños de cada cluster."""
        if self.labels is None:
            return {}
        
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}
    
    def find_optimal_k(
        self, 
        k_range: range = range(5, 30, 5),
        method: str = "silhouette"
    ) -> Dict[str, Any]:
        """
        Encuentra el número óptimo de clusters.
        
        Args:
            k_range: Rango de valores de k a probar
            method: "silhouette" o "elbow"
            
        Returns:
            Resultados del análisis
        """
        if not SKLEARN_CLUSTERING_AVAILABLE:
            raise ImportError("scikit-learn no disponible")
            
        results = {"k_values": [], "scores": [], "inertias": []}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            results["k_values"].append(k)
            results["inertias"].append(float(kmeans.inertia_))
            
            if method == "silhouette":
                score = silhouette_score(self.embeddings, labels)
                results["scores"].append(float(score))
        
        # Encontrar mejor k
        if method == "silhouette":
            best_idx = np.argmax(results["scores"])
            results["optimal_k"] = results["k_values"][best_idx]
            results["best_score"] = results["scores"][best_idx]
        
        return results


def get_clustering_availability() -> Dict[str, bool]:
    """Retorna disponibilidad de métodos de clustering."""
    return {
        "kmeans": SKLEARN_CLUSTERING_AVAILABLE,
        "hdbscan": HDBSCAN_AVAILABLE,
        "umap": UMAP_AVAILABLE,
        "tsne": TSNE_AVAILABLE
    }
