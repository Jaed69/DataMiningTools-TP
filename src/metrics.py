"""
Módulo de evaluación y métricas para los modelos.
Incluye métricas para recomendación y clasificación.
"""
import numpy as np
from typing import List, Dict, Set, Any, Tuple
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
    hamming_loss,
    accuracy_score
)
import time


class RecommenderMetrics:
    """
    Métricas de evaluación para sistemas de recomendación.
    """
    
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision@K: Proporción de items relevantes en las top-k recomendaciones.
        
        Args:
            recommended: Lista ordenada de items recomendados
            relevant: Conjunto de items relevantes (ground truth)
            k: Número de recomendaciones a considerar
            
        Returns:
            Precision@K (0-1)
        """
        if k <= 0:
            return 0.0
        
        recommended_at_k = set(recommended[:k])
        relevant_and_recommended = recommended_at_k.intersection(relevant)
        
        return len(relevant_and_recommended) / k
    
    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall@K: Proporción de items relevantes encontrados en top-k.
        
        Args:
            recommended: Lista ordenada de items recomendados
            relevant: Conjunto de items relevantes
            k: Número de recomendaciones
            
        Returns:
            Recall@K (0-1)
        """
        if not relevant:
            return 0.0
        
        recommended_at_k = set(recommended[:k])
        relevant_and_recommended = recommended_at_k.intersection(relevant)
        
        return len(relevant_and_recommended) / len(relevant)
    
    @staticmethod
    def average_precision(recommended: List[str], relevant: Set[str]) -> float:
        """
        Average Precision: Promedio de Precision@k para cada k donde hay un item relevante.
        
        Returns:
            AP (0-1)
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        hits = 0
        
        for i, item in enumerate(recommended, 1):
            if item in relevant:
                hits += 1
                score += hits / i
        
        return score / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_average_precision(
        recommendations_list: List[List[str]], 
        relevants_list: List[Set[str]]
    ) -> float:
        """
        Mean Average Precision: Promedio de AP sobre múltiples consultas.
        
        Args:
            recommendations_list: Lista de listas de recomendaciones
            relevants_list: Lista de conjuntos de items relevantes
            
        Returns:
            MAP (0-1)
        """
        if len(recommendations_list) != len(relevants_list):
            raise ValueError("Las listas deben tener el mismo tamaño")
        
        ap_scores = [
            RecommenderMetrics.average_precision(recs, rels)
            for recs, rels in zip(recommendations_list, relevants_list)
        ]
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain @K.
        Considera la posición de los items relevantes (mejores posiciones = más valor).
        
        Returns:
            NDCG@K (0-1)
        """
        def dcg(scores: List[float]) -> float:
            return sum(
                rel / np.log2(i + 2) 
                for i, rel in enumerate(scores)
            )
        
        # Relevancia binaria
        relevances = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
        
        actual_dcg = dcg(relevances)
        
        # DCG ideal (todos los relevantes primero)
        ideal_relevances = sorted(relevances, reverse=True)
        ideal_dcg = dcg(ideal_relevances)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def diversity(
        recommendations: List[List[str]], 
        similarity_matrix: np.ndarray,
        title_to_idx: Dict[str, int]
    ) -> float:
        """
        Diversidad: Mide qué tan diferentes son las recomendaciones entre sí.
        (1 - promedio de similitud entre pares)
        
        Returns:
            Diversity score (0-1, mayor = más diverso)
        """
        all_diversities = []
        
        for recs in recommendations:
            indices = [title_to_idx.get(r) for r in recs if r in title_to_idx]
            if len(indices) < 2:
                continue
            
            similarities = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    similarities.append(similarity_matrix[indices[i], indices[j]])
            
            if similarities:
                all_diversities.append(1 - np.mean(similarities))
        
        return np.mean(all_diversities) if all_diversities else 0.0
    
    @staticmethod
    def coverage(
        all_recommendations: List[List[str]], 
        catalog_size: int
    ) -> float:
        """
        Cobertura: Proporción del catálogo que aparece en las recomendaciones.
        
        Returns:
            Coverage (0-1)
        """
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        
        return len(unique_items) / catalog_size if catalog_size > 0 else 0.0


class ClassifierMetrics:
    """
    Métricas de evaluación para clasificación multietiqueta.
    """
    
    @staticmethod
    def evaluate_multilabel(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evalúa clasificación multietiqueta con múltiples métricas.
        
        Args:
            y_true: Etiquetas verdaderas (binarizadas)
            y_pred: Etiquetas predichas (binarizadas)
            
        Returns:
            Diccionario con métricas
        """
        return {
            "f1_micro": f1_score(y_true, y_pred, average='micro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            "precision_micro": precision_score(y_true, y_pred, average='micro'),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_micro": recall_score(y_true, y_pred, average='micro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "subset_accuracy": accuracy_score(y_true, y_pred)  # Exact match
        }
    
    @staticmethod
    def per_class_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Métricas por clase individual.
        
        Returns:
            Diccionario {clase: {precision, recall, f1}}
        """
        results = {}
        
        for i, class_name in enumerate(class_names):
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
            
            # Evitar división por cero
            tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
            fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
            fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(y_true_class))
            }
        
        return results


class BenchmarkRunner:
    """
    Ejecuta benchmarks completos de modelos.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_recommender(
        self,
        model: Any,
        test_titles: List[str],
        num_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark de un modelo de recomendación.
        
        Args:
            model: Modelo a evaluar
            test_titles: Títulos de prueba
            num_recommendations: Número de recomendaciones por título
            
        Returns:
            Resultados del benchmark
        """
        inference_times = []
        all_recommendations = []
        
        for title in test_titles:
            start = time.time()
            try:
                recs = model.recommend(title, top_k=num_recommendations)
                rec_titles = [r[0] for r in recs] if recs else []
                all_recommendations.append(rec_titles)
                inference_times.append(time.time() - start)
            except:
                all_recommendations.append([])
                inference_times.append(0)
        
        results = {
            "model_name": model.name,
            "training_time": model.training_time,
            "avg_inference_time_ms": np.mean(inference_times) * 1000,
            "std_inference_time_ms": np.std(inference_times) * 1000,
            "min_inference_time_ms": np.min(inference_times) * 1000 if inference_times else 0,
            "max_inference_time_ms": np.max(inference_times) * 1000 if inference_times else 0,
            "total_recommendations": sum(len(r) for r in all_recommendations),
            "coverage": RecommenderMetrics.coverage(
                all_recommendations, 
                len(model.titles) if hasattr(model, 'titles') else 1000
            )
        }
        
        self.results[model.name] = results
        return results
    
    def benchmark_classifier(
        self,
        model: Any,
        test_texts: List[str],
        test_labels: List[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark de un modelo de clasificación.
        """
        inference_times = []
        predictions = []
        
        for text in test_texts:
            start = time.time()
            try:
                pred = model.predict(text, top_k=3)
                predictions.append(pred)
                inference_times.append(time.time() - start)
            except:
                predictions.append({})
                inference_times.append(0)
        
        results = {
            "model_name": model.name,
            "training_time": model.training_time,
            "avg_inference_time_ms": np.mean(inference_times) * 1000,
            "std_inference_time_ms": np.std(inference_times) * 1000,
            "num_classes": len(model.classes) if hasattr(model, 'classes') else 0,
            "total_predictions": len(predictions)
        }
        
        self.results[model.name] = results
        return results
    
    def compare_models(
        self,
        models: List[Any],
        test_titles: List[str]
    ) -> Dict[str, Any]:
        """
        Compara múltiples modelos.
        """
        comparison = {}
        
        for model in models:
            if hasattr(model, 'recommend'):
                comparison[model.name] = self.benchmark_recommender(model, test_titles)
            elif hasattr(model, 'predict'):
                comparison[model.name] = self.benchmark_classifier(model, test_titles)
        
        return comparison
    
    def generate_report(self) -> str:
        """
        Genera un reporte en texto de todos los benchmarks.
        """
        report = "=" * 60 + "\n"
        report += "BENCHMARK REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for model_name, metrics in self.results.items():
            report += f"Model: {model_name}\n"
            report += "-" * 40 + "\n"
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"  {metric}: {value:.4f}\n"
                else:
                    report += f"  {metric}: {value}\n"
            
            report += "\n"
        
        return report
