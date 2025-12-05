"""
Módulo de Evaluación Centralizado.
==================================
Contiene todas las funciones de evaluación de modelos.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import streamlit as st


class ModelEvaluator:
    """Evaluador centralizado de modelos de recomendación y clasificación."""
    
    def __init__(self, data: pd.DataFrame, title_to_genres: Dict[str, List[str]] = None):
        """
        Args:
            data: DataFrame con los datos de películas
            title_to_genres: Mapeo de título a lista de géneros
        """
        self.data = data
        self.titles = data['title'].tolist() if data is not None else []
        
        # Crear mapeo de géneros si no se proporciona
        if title_to_genres is None and data is not None:
            self.title_to_genres = self._build_genre_mapping()
        else:
            self.title_to_genres = title_to_genres or {}
    
    def _build_genre_mapping(self) -> Dict[str, List[str]]:
        """Construye el mapeo título -> géneros."""
        mapping = {}
        for _, row in self.data.iterrows():
            title = row['title']
            genres = row.get('genres_list', [])
            if not genres and 'listed_in' in row:
                genres = [g.strip() for g in str(row['listed_in']).split(',') if g.strip()]
            mapping[title] = genres
        return mapping
    
    def get_relevant_titles(self, source_title: str) -> Set[str]:
        """Obtiene títulos relevantes (comparten al menos un género)."""
        source_genres = set(self.title_to_genres.get(source_title, []))
        if not source_genres:
            return set()
        
        relevant = set()
        for title, genres in self.title_to_genres.items():
            if title != source_title and source_genres.intersection(set(genres)):
                relevant.add(title)
        return relevant
    
    def evaluate_recommender(
        self, 
        model, 
        test_titles: List[str], 
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evalúa un modelo de recomendación.
        
        Args:
            model: Modelo con método recommend()
            test_titles: Lista de títulos de prueba
            k: Número de recomendaciones
            
        Returns:
            Diccionario con métricas promedio
        """
        from src.metrics import RecommenderMetrics
        
        precisions, recalls, ndcgs, aps, diversities = [], [], [], [], []
        successful_evals = 0
        
        for title in test_titles:
            try:
                # Obtener recomendaciones
                recs = model.recommend(title, top_k=k)
                rec_titles = [r[0] for r in recs]
                
                # Obtener ground truth
                relevant = self.get_relevant_titles(title)
                if not relevant:
                    continue
                
                # Calcular métricas
                precisions.append(RecommenderMetrics.precision_at_k(rec_titles, relevant, k))
                recalls.append(RecommenderMetrics.recall_at_k(rec_titles, relevant, k))
                ndcgs.append(RecommenderMetrics.ndcg_at_k(rec_titles, relevant, k))
                aps.append(RecommenderMetrics.average_precision(rec_titles, relevant))
                diversities.append(RecommenderMetrics.genre_diversity(rec_titles, self.title_to_genres))
                successful_evals += 1
                
            except Exception:
                continue
        
        if successful_evals == 0:
            return {}
        
        return {
            "Precision@K": np.mean(precisions),
            "Recall@K": np.mean(recalls),
            "nDCG@K": np.mean(ndcgs),
            "MAP": np.mean(aps),
            "Genre Diversity": np.mean(diversities),
            "Evaluaciones": successful_evals
        }
    
    def evaluate_all_recommenders(
        self, 
        models: Dict[str, Any], 
        n_test: int = 30, 
        k: int = 5, 
        seed: int = 42,
        progress_callback=None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evalúa todos los modelos de recomendación.
        
        Args:
            models: Diccionario {nombre: modelo}
            n_test: Número de títulos de prueba
            k: Top-K para métricas
            seed: Semilla aleatoria
            progress_callback: Función callback(progress, status_text)
            
        Returns:
            Diccionario con métricas por modelo
        """
        np.random.seed(seed)
        test_indices = np.random.choice(len(self.titles), min(n_test, len(self.titles)), replace=False)
        test_titles = [self.titles[i] for i in test_indices]
        
        results = {}
        model_list = list(models.items())
        
        for idx, (name, model) in enumerate(model_list):
            if progress_callback:
                progress_callback((idx + 1) / len(model_list), f"Evaluando {name}...")
            
            metrics = self.evaluate_recommender(model, test_titles, k)
            if metrics:
                results[name] = metrics
        
        return results
    
    def compare_recommendations(
        self, 
        models: Dict[str, Any], 
        source_title: str, 
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Compara recomendaciones de múltiples modelos para un título.
        
        Returns:
            Diccionario con recomendaciones, overlap, y estadísticas
        """
        recommendations = {}
        title_sets = {}
        genre_coverage = {}
        score_stats = {}
        
        for name, model in models.items():
            try:
                recs = model.recommend(source_title, top_k=k)
                recommendations[name] = recs
                
                # Títulos únicos
                rec_titles = set(r[0] for r in recs)
                title_sets[name] = rec_titles
                
                # Cobertura de géneros
                all_genres = set()
                for rec in recs:
                    genre = rec[2] if len(rec) > 2 else ""
                    for g in str(genre).split(','):
                        all_genres.add(g.strip())
                genre_coverage[name] = all_genres
                
                # Estadísticas de scores
                scores = [r[1] for r in recs if len(r) > 1]
                score_stats[name] = {
                    "mean": np.mean(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0
                }
                
            except Exception as e:
                continue
        
        # Calcular overlaps
        overlaps = {}
        model_names = list(title_sets.keys())
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    common = title_sets[m1] & title_sets[m2]
                    overlaps[f"{m1} ∩ {m2}"] = {
                        "count": len(common),
                        "titles": list(common),
                        "percentage": len(common) / k * 100
                    }
        
        # Títulos en común de todos
        all_common = set.intersection(*title_sets.values()) if title_sets else set()
        
        return {
            "recommendations": recommendations,
            "title_sets": title_sets,
            "genre_coverage": genre_coverage,
            "score_stats": score_stats,
            "overlaps": overlaps,
            "all_common": all_common
        }


def evaluate_classifiers_with_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    sample_size: int = 2000,
    seed: int = 42,
    progress_callback=None
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa clasificadores con train/test split.
    
    Args:
        data: DataFrame con datos
        test_size: Proporción de datos de test
        sample_size: Tamaño de muestra
        seed: Semilla aleatoria
        progress_callback: Función callback(progress, status)
        
    Returns:
        Diccionario con métricas por clasificador
    """
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
    import time
    
    # Preparar datos
    df_sample = data.sample(n=min(sample_size, len(data)), random_state=seed)
    texts = df_sample['processed_description'].tolist()
    labels = df_sample['genres_list'].tolist()
    
    # Filtrar válidos
    valid = [(t, l) for t, l in zip(texts, labels) if t and l]
    texts, labels = zip(*valid) if valid else ([], [])
    
    if len(texts) < 10:
        return {}
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed
    )
    
    # Vectorizar
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Binarizar
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)
    
    # Clasificadores
    classifiers = {
        "Logistic Regression": OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000)),
        "Naive Bayes": OneVsRestClassifier(MultinomialNB()),
        "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
    }
    
    results = {}
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        if progress_callback:
            progress_callback((idx + 1) / len(classifiers), f"Entrenando {name}...")
        
        start = time.time()
        clf.fit(X_train_vec, y_train_bin)
        train_time = time.time() - start
        
        y_pred = clf.predict(X_test_vec)
        
        results[name] = {
            "train_time": train_time,
            "f1_micro": f1_score(y_test_bin, y_pred, average='micro'),
            "f1_macro": f1_score(y_test_bin, y_pred, average='macro'),
            "hamming_loss": hamming_loss(y_test_bin, y_pred),
            "precision": precision_score(y_test_bin, y_pred, average='micro', zero_division=0),
            "recall": recall_score(y_test_bin, y_pred, average='micro', zero_division=0)
        }
    
    return results
