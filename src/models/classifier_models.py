"""
Modelos de clasificación multietiqueta para géneros.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss
from .base_model import BaseClassifierModel


class TFIDFClassifier(BaseClassifierModel):
    """
    Clasificador de géneros usando TF-IDF + Regresión Logística.
    Baseline rápido y efectivo.
    """
    
    def __init__(
        self, 
        max_features: int = 5000,
        classifier_type: str = "logistic"  # "logistic", "naive_bayes", "random_forest"
    ):
        super().__init__(
            name=f"TF-IDF + {classifier_type.title()}",
            description=f"Clasificador multietiqueta basado en TF-IDF con {classifier_type}."
        )
        self.max_features = max_features
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2)
        )
        self.mlb = MultiLabelBinarizer()
        self.classifier = None
        
    def _get_base_classifier(self):
        """Retorna el clasificador base según el tipo."""
        if self.classifier_type == "logistic":
            return LogisticRegression(solver='liblinear', max_iter=1000)
        elif self.classifier_type == "naive_bayes":
            return MultinomialNB()
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, n_jobs=-1)
        else:
            return LogisticRegression(solver='liblinear')
        
    def fit(self, texts: List[str], labels: List[List[str]]) -> None:
        """Entrena el clasificador."""
        start_time = time.time()
        
        # Vectorizar textos
        X = self.vectorizer.fit_transform(texts)
        
        # Binarizar etiquetas
        y = self.mlb.fit_transform(labels)
        self.classes = list(self.mlb.classes_)
        
        # Entrenar clasificador
        base_clf = self._get_base_classifier()
        self.classifier = OneVsRestClassifier(base_clf)
        self.classifier.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
    def predict(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """Predice géneros para un texto."""
        if not self.is_trained:
            raise ValueError("El clasificador no ha sido entrenado.")
            
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        
        # Obtener top k
        top_indices = probs.argsort()[-top_k:][::-1]
        results = {self.classes[i]: float(probs[i]) for i in top_indices}
        
        return results
    
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict[str, float]]:
        """Predice géneros para múltiples textos."""
        return [self.predict(text, top_k) for text in texts]
    
    def evaluate(self, texts: List[str], true_labels: List[List[str]]) -> Dict[str, float]:
        """Evalúa el modelo con métricas estándar."""
        if not self.is_trained:
            raise ValueError("El clasificador no ha sido entrenado.")
            
        X = self.vectorizer.transform(texts)
        y_true = self.mlb.transform(true_labels)
        y_pred = self.classifier.predict(X)
        
        return {
            "f1_micro": f1_score(y_true, y_pred, average='micro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "hamming_loss": hamming_loss(y_true, y_pred)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        info = super().get_info()
        info.update({
            "classifier_type": self.classifier_type,
            "max_features": self.max_features,
            "algorithm": f"TF-IDF + OneVsRest({self.classifier_type})"
        })
        return info


class EnsembleClassifier(BaseClassifierModel):
    """
    Clasificador ensemble que combina múltiples modelos.
    """
    
    def __init__(self, classifiers: List[BaseClassifierModel] = None):
        super().__init__(
            name="Ensemble Classifier",
            description="Combina predicciones de múltiples clasificadores para mayor robustez."
        )
        self.classifiers = classifiers or []
        
    def add_classifier(self, classifier: BaseClassifierModel):
        """Agrega un clasificador al ensemble."""
        self.classifiers.append(classifier)
        
    def fit(self, texts: List[str], labels: List[List[str]]) -> None:
        """Entrena todos los clasificadores del ensemble."""
        start_time = time.time()
        
        for clf in self.classifiers:
            clf.fit(texts, labels)
            
        self.classes = self.classifiers[0].classes if self.classifiers else []
        self.training_time = time.time() - start_time
        self.is_trained = True
        
    def predict(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """Predice combinando resultados de todos los clasificadores."""
        if not self.is_trained or not self.classifiers:
            raise ValueError("El ensemble no ha sido entrenado.")
            
        # Recopilar predicciones
        all_predictions = {}
        for clf in self.classifiers:
            preds = clf.predict(text, top_k=len(self.classes))  # Obtener todas las clases
            for genre, prob in preds.items():
                if genre not in all_predictions:
                    all_predictions[genre] = []
                all_predictions[genre].append(prob)
        
        # Promediar probabilidades
        averaged = {
            genre: np.mean(probs) 
            for genre, probs in all_predictions.items()
        }
        
        # Ordenar y tomar top k
        sorted_preds = sorted(averaged.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])
    
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict[str, float]]:
        """Predice géneros para múltiples textos."""
        return [self.predict(text, top_k) for text in texts]
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del ensemble."""
        info = super().get_info()
        info.update({
            "num_classifiers": len(self.classifiers),
            "classifiers": [clf.name for clf in self.classifiers],
            "algorithm": "Ensemble (Average Voting)"
        })
        return info
