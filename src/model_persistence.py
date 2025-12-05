"""
Sistema de persistencia de modelos pre-entrenados.
Permite guardar y cargar modelos para demos sin necesidad de re-entrenar.
"""
import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Directorio para modelos guardados
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_cache")


def ensure_models_dir():
    """Asegura que existe el directorio de modelos."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def save_model(model: Any, name: str, metadata: Dict = None) -> str:
    """
    Guarda un modelo entrenado en disco.
    
    Args:
        model: Modelo a guardar
        name: Nombre del modelo (ej: "tfidf_recommender")
        metadata: Información adicional (tiempo entrenamiento, etc.)
    
    Returns:
        Ruta del archivo guardado
    """
    ensure_models_dir()
    
    filepath = os.path.join(MODELS_DIR, f"{name}.pkl")
    meta_filepath = os.path.join(MODELS_DIR, f"{name}_meta.json")
    
    # Guardar modelo
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar metadata
    meta = metadata or {}
    meta["saved_at"] = datetime.now().isoformat()
    meta["model_name"] = name
    
    with open(meta_filepath, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Modelo guardado: {filepath}")
    return filepath


def load_model(name: str) -> Optional[Any]:
    """
    Carga un modelo pre-entrenado desde disco.
    
    Args:
        name: Nombre del modelo
    
    Returns:
        Modelo cargado o None si no existe o hay error
    """
    filepath = os.path.join(MODELS_DIR, f"{name}.pkl")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"📦 Modelo cargado: {name}")
        return model
    except ModuleNotFoundError as e:
        print(f"⚠️ No se puede cargar {name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error cargando {name}: {e}")
        return None


def load_model_metadata(name: str) -> Optional[Dict]:
    """Carga la metadata de un modelo."""
    meta_filepath = os.path.join(MODELS_DIR, f"{name}_meta.json")
    
    if not os.path.exists(meta_filepath):
        return None
    
    with open(meta_filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def model_exists(name: str) -> bool:
    """Verifica si un modelo existe en cache."""
    filepath = os.path.join(MODELS_DIR, f"{name}.pkl")
    return os.path.exists(filepath)


def list_saved_models() -> Dict[str, Dict]:
    """Lista todos los modelos guardados con su metadata."""
    ensure_models_dir()
    
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl'):
            name = filename.replace('.pkl', '')
            meta = load_model_metadata(name) or {}
            models[name] = {
                "path": os.path.join(MODELS_DIR, filename),
                "metadata": meta
            }
    
    return models


def save_benchmark_results(results: Dict, name: str = "benchmark_results") -> str:
    """Guarda resultados de benchmark pre-calculados."""
    ensure_models_dir()
    
    filepath = os.path.join(MODELS_DIR, f"{name}.json")
    
    # Agregar timestamp
    results["generated_at"] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Benchmark guardado: {filepath}")
    return filepath


def load_benchmark_results(name: str = "benchmark_results") -> Optional[Dict]:
    """Carga resultados de benchmark pre-calculados."""
    filepath = os.path.join(MODELS_DIR, f"{name}.json")
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_cache():
    """Limpia todos los modelos guardados."""
    ensure_models_dir()
    
    for filename in os.listdir(MODELS_DIR):
        filepath = os.path.join(MODELS_DIR, filename)
        os.remove(filepath)
        print(f"🗑️ Eliminado: {filename}")
