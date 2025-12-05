"""
Script para pre-entrenar todos los modelos y guardarlos en cache.
Ejecutar ANTES de la demo para tener todo listo.

Uso:
    python train_models.py

Esto entrenará y guardará:
    - Modelos de recomendación (TF-IDF, Doc2Vec, SBERT si disponibles)
    - Clasificadores de géneros (Logistic, NaiveBayes, RandomForest)
    - Resultados de benchmark pre-calculados
    - Ejemplos de comparación
"""
import os
import sys
import time
import json
from datetime import datetime

# Asegurar que src está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.model_persistence import (
    save_model, save_benchmark_results, ensure_models_dir, MODELS_DIR
)
from src.models import (
    TFIDFRecommender,
    TFIDFClassifier,
    GENSIM_AVAILABLE,
    SBERT_AVAILABLE
)

# Importar modelos opcionales
if GENSIM_AVAILABLE:
    from src.models import Doc2VecRecommender
if SBERT_AVAILABLE:
    from src.models import SBERTRecommender

# Intentar importar BM25
try:
    from src.models import BM25Recommender, BM25_AVAILABLE
except ImportError:
    BM25_AVAILABLE = False


CSV_PATH = os.path.join("data", "netflix_titles.csv")


def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def train_and_save_all():
    """Entrena y guarda todos los modelos disponibles."""
    
    print_header("🎬 NETFLIX AI - Entrenamiento de Modelos")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Directorio de cache: {MODELS_DIR}")
    
    # Verificar datos
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: No se encontró {CSV_PATH}")
        return
    
    # Cargar datos
    print_header("📊 Cargando datos...")
    loader = DataLoader(CSV_PATH)
    data = loader.get_processed_data()
    
    texts = data['processed_description'].tolist()
    titles = data['title'].tolist()
    genres = data['genres_list'].tolist()
    
    print(f"✅ Cargados {len(titles)} títulos")
    
    # Resultados de benchmark
    benchmark = {
        "dataset_info": {
            "total_titles": len(titles),
            "unique_genres": len(set(g for gs in genres for g in gs)),
            "avg_description_length": sum(len(t) for t in texts) / len(texts)
        },
        "recommenders": {},
        "classifiers": {},
        "examples": {},
        "generated_at": datetime.now().isoformat()
    }
    
    # =========================================================================
    # ENTRENAR RECOMENDADORES
    # =========================================================================
    print_header("🎯 Entrenando Modelos de Recomendación")
    
    # 1. TF-IDF (siempre disponible)
    print("\n[1/4] TF-IDF Recommender...")
    tfidf_rec = TFIDFRecommender()
    start = time.time()
    tfidf_rec.fit(texts, titles, data)
    tfidf_time = time.time() - start
    
    save_model(tfidf_rec, "tfidf_recommender", {
        "training_time": tfidf_time,
        "type": "recommender",
        "algorithm": "TF-IDF",
        "n_documents": len(titles)
    })
    benchmark["recommenders"]["TF-IDF"] = {
        "training_time": tfidf_time,
        "available": True,
        "description": "Modelo basado en frecuencia de términos"
    }
    print(f"   ✅ Entrenado en {tfidf_time:.2f}s")
    
    # 2. BM25 (opcional)
    if BM25_AVAILABLE:
        print("\n[2/4] BM25 Recommender...")
        bm25_rec = BM25Recommender()
        start = time.time()
        bm25_rec.fit(texts, titles, data)
        bm25_time = time.time() - start
        
        save_model(bm25_rec, "bm25_recommender", {
            "training_time": bm25_time,
            "type": "recommender",
            "algorithm": "BM25",
            "n_documents": len(titles)
        })
        benchmark["recommenders"]["BM25"] = {
            "training_time": bm25_time,
            "available": True,
            "description": "Okapi BM25 - Mejora de TF-IDF con normalización"
        }
        print(f"   ✅ Entrenado en {bm25_time:.2f}s")
    else:
        print("\n[2/4] BM25 - ⚠️ No disponible (instalar rank-bm25)")
        benchmark["recommenders"]["BM25"] = {
            "available": False,
            "description": "Requiere: pip install rank-bm25"
        }
    
    # 3. Doc2Vec (opcional)
    if GENSIM_AVAILABLE:
        print("\n[3/4] Doc2Vec Recommender...")
        doc2vec_rec = Doc2VecRecommender()
        start = time.time()
        doc2vec_rec.fit(texts, titles, data)
        doc2vec_time = time.time() - start
        
        save_model(doc2vec_rec, "doc2vec_recommender", {
            "training_time": doc2vec_time,
            "type": "recommender",
            "algorithm": "Doc2Vec",
            "n_documents": len(titles)
        })
        benchmark["recommenders"]["Doc2Vec"] = {
            "training_time": doc2vec_time,
            "available": True,
            "description": "Embeddings de documentos con redes neuronales"
        }
        print(f"   ✅ Entrenado en {doc2vec_time:.2f}s")
    else:
        print("\n[3/4] Doc2Vec - ⚠️ No disponible (instalar gensim)")
        benchmark["recommenders"]["Doc2Vec"] = {
            "available": False,
            "description": "Requiere: pip install gensim"
        }
    
    # 4. SBERT (opcional)
    if SBERT_AVAILABLE:
        print("\n[4/4] SBERT Recommender...")
        sbert_rec = SBERTRecommender()
        start = time.time()
        sbert_rec.fit(texts, titles, data)
        sbert_time = time.time() - start
        
        save_model(sbert_rec, "sbert_recommender", {
            "training_time": sbert_time,
            "type": "recommender",
            "algorithm": "SBERT",
            "model_name": "all-MiniLM-L6-v2",
            "n_documents": len(titles)
        })
        benchmark["recommenders"]["SBERT"] = {
            "training_time": sbert_time,
            "available": True,
            "description": "Transformers pre-entrenados (estado del arte)"
        }
        print(f"   ✅ Entrenado en {sbert_time:.2f}s")
    else:
        print("\n[4/4] SBERT - ⚠️ No disponible (instalar sentence-transformers)")
        benchmark["recommenders"]["SBERT"] = {
            "available": False,
            "description": "Requiere: pip install sentence-transformers"
        }
    
    # =========================================================================
    # ENTRENAR CLASIFICADORES
    # =========================================================================
    print_header("🏷️ Entrenando Clasificadores de Géneros")
    
    classifiers = [
        ("Logistic", "logistic", "Regresión Logística multietiqueta"),
        ("NaiveBayes", "naive_bayes", "Naive Bayes Multinomial"),
        ("RandomForest", "random_forest", "Ensemble de árboles de decisión")
    ]
    
    for i, (name, clf_type, desc) in enumerate(classifiers, 1):
        print(f"\n[{i}/3] {name} Classifier...")
        clf = TFIDFClassifier(classifier_type=clf_type)
        start = time.time()
        clf.fit(texts, genres)
        clf_time = time.time() - start
        
        save_model(clf, f"{clf_type}_classifier", {
            "training_time": clf_time,
            "type": "classifier",
            "algorithm": name,
            "n_samples": len(texts),
            "n_classes": len(clf.classes)
        })
        benchmark["classifiers"][name] = {
            "training_time": clf_time,
            "available": True,
            "n_classes": len(clf.classes),
            "description": desc
        }
        print(f"   ✅ Entrenado en {clf_time:.2f}s ({len(clf.classes)} géneros)")
    
    # =========================================================================
    # GENERAR EJEMPLOS DE COMPARACIÓN
    # =========================================================================
    print_header("📋 Generando Ejemplos de Comparación")
    
    # Seleccionar títulos de ejemplo
    example_titles = [
        "Stranger Things",
        "Breaking Bad",
        "The Crown",
        "Money Heist",
        "Black Mirror"
    ]
    
    # Filtrar solo títulos que existen
    example_titles = [t for t in example_titles if t in titles]
    if len(example_titles) < 3:
        example_titles = titles[:5]  # Fallback a los primeros 5
    
    print(f"Títulos de ejemplo: {example_titles}")
    
    comparison_examples = {}
    
    for title in example_titles:
        print(f"\n  Procesando: {title}")
        comparison_examples[title] = {}
        
        # TF-IDF
        start = time.time()
        recs = tfidf_rec.recommend(title, top_k=5)
        inference_time = time.time() - start
        comparison_examples[title]["TF-IDF"] = {
            "recommendations": [(r[0], float(r[1]), r[2] if len(r) > 2 else "") for r in recs],
            "inference_time_ms": inference_time * 1000
        }
        
        # Doc2Vec
        if GENSIM_AVAILABLE:
            start = time.time()
            recs = doc2vec_rec.recommend(title, top_k=5)
            inference_time = time.time() - start
            comparison_examples[title]["Doc2Vec"] = {
                "recommendations": [(r[0], float(r[1]), r[2] if len(r) > 2 else "") for r in recs],
                "inference_time_ms": inference_time * 1000
            }
        
        # SBERT
        if SBERT_AVAILABLE:
            start = time.time()
            recs = sbert_rec.recommend(title, top_k=5)
            inference_time = time.time() - start
            comparison_examples[title]["SBERT"] = {
                "recommendations": [(r[0], float(r[1]), r[2] if len(r) > 2 else "") for r in recs],
                "inference_time_ms": inference_time * 1000
            }
    
    benchmark["examples"]["recommendations"] = comparison_examples
    
    # Ejemplos de clasificación
    example_synopses = [
        {
            "text": "A group of friends discover a dark secret in their small town that leads them on a supernatural adventure.",
            "expected": "Thriller, Drama"
        },
        {
            "text": "A romantic comedy about two strangers who meet on a plane and fall in love during a layover in Paris.",
            "expected": "Comedy, Romance"
        },
        {
            "text": "Documentary exploring the effects of climate change on polar bear populations in the Arctic.",
            "expected": "Documentary"
        }
    ]
    
    clf_examples = {}
    for i, example in enumerate(example_synopses):
        text = example["text"]
        clf_examples[f"example_{i+1}"] = {
            "synopsis": text,
            "expected": example["expected"],
            "predictions": {}
        }
        
        # Logistic
        clf = TFIDFClassifier(classifier_type="logistic")
        clf.fit(texts, genres)
        preds = clf.predict(text, top_k=3)
        clf_examples[f"example_{i+1}"]["predictions"]["Logistic"] = {
            k: float(v) for k, v in preds.items()
        }
    
    benchmark["examples"]["classifications"] = clf_examples
    
    # =========================================================================
    # GUARDAR BENCHMARK
    # =========================================================================
    print_header("💾 Guardando Resultados de Benchmark")
    save_benchmark_results(benchmark)
    
    # Resumen final
    print_header("✅ ENTRENAMIENTO COMPLETADO")
    print(f"""
Modelos guardados en: {MODELS_DIR}

📊 Resumen:
   - Recomendadores: {sum(1 for r in benchmark['recommenders'].values() if r.get('available', False))} entrenados
   - Clasificadores: {len(benchmark['classifiers'])} entrenados
   - Ejemplos generados: {len(example_titles)} títulos

🚀 Ahora puedes ejecutar la demo:
   python app_demo.py
    """)


if __name__ == "__main__":
    train_and_save_all()
