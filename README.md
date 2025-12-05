# 🎬 Netflix AI - Sistema Multi-Modelo de Recomendación

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20|%20Doc2Vec%20|%20SBERT-red?style=for-the-badge)

**Sistema avanzado de recomendación con múltiples modelos de NLP y comparador de algoritmos**

[Demo](#-inicio-rápido) | [Documentación](#-arquitectura) | [Modelos](#-modelos-disponibles)

</div>

---

## 📋 Descripción

Sistema de recomendación basado en contenido para el catálogo de Netflix que implementa **múltiples algoritmos de NLP** con una interfaz interactiva en Streamlit. Permite comparar el rendimiento de diferentes modelos (TF-IDF, Doc2Vec, SBERT) y clasificadores de géneros con métricas cuantitativas y explicaciones detalladas.

## 🚀 Características Principales

### 🎯 Recomendación Multi-Modelo
- **TF-IDF**: Modelo clásico basado en frecuencia de términos
- **BM25**: Mejora de TF-IDF con normalización por longitud de documento (Okapi BM25)
- **Doc2Vec**: Embeddings de documentos para captura semántica
- **SBERT**: Sentence Transformers - Estado del arte en similitud semántica
- **Cross-Encoder**: Reranking de alta precisión para mejorar top-K

### 🔎 Búsqueda Semántica
- Búsqueda por lenguaje natural ("películas sobre supervivencia")
- Entiende sinónimos y conceptos relacionados
- Powered by Sentence-BERT

### 🎨 Clustering y Visualización
- Agrupación automática de películas con K-Means y HDBSCAN
- Visualización 2D/3D con UMAP y t-SNE
- Descubrimiento de grupos temáticos

### 🏷️ Clasificación de Géneros
- **Logistic Regression**: Clasificador rápido y preciso
- **Naive Bayes**: Excelente para texto
- **Random Forest**: Robusto ante ruido

### 📈 Evaluación y Métricas
- Comparación cuantitativa entre algoritmos
- Métricas: Precision@K, Recall@K, F1-Score, Hamming Loss
- Visualización interactiva con gráficos

### 💡 Explicaciones Detalladas
- Entiende **por qué** cada algoritmo recomienda diferente
- Análisis de similitud semántica vs léxica
- Comparación lado a lado de resultados

## 📊 Dataset

El proyecto utiliza el dataset `netflix_titles.csv` que contiene información sobre películas y series disponibles en Netflix. El dataset incluye las siguientes columnas principales:

| Variable | Descripción |
|----------|-------------|
| `show_id` | Identificador único asignado a cada título |
| `type` | Clasifica el contenido como Movie (película) o TV Show (serie) |
| `title` | Nombre oficial del título |
| `director` | Nombre del director de la producción |
| `cast` | Lista de actores que participan en la obra |
| `country` | País o países de origen de la producción |
| `date_added` | Fecha en la que el título fue incorporado a Netflix |
| `release_year` | Año de estreno o lanzamiento |
| `rating` | Clasificación por edad o tipo de audiencia |
| `duration` | Duración de la película o cantidad de temporadas |
| `listed_in` | Géneros o categorías temáticas |
| `description` | Breve sinopsis del título |

## 🛠️ Tecnologías y Librerías

- **Python 3.x**
- **pandas** - Manipulación y análisis de datos
- **numpy** - Operaciones numéricas
- **matplotlib** y **seaborn** - Visualización de datos
- **scipy** - Análisis estadístico
- **nltk** - Procesamiento de lenguaje natural
- **sweetviz** - Análisis exploratorio automatizado
- **scikit-learn** - Modelos de machine learning (próxima implementación)

## 📁 Estructura del Proyecto

```
DataMiningTools-TP/
│
├── app_streamlit.py           # 🚀 Aplicación principal (Streamlit)
├── train_models.py            # Script para entrenar y cachear modelos
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Documentación
├── PPT.html                   # Presentación del proyecto
│
├── data/
│   ├── netflix_titles.csv         # Dataset original
│   └── netflix_titles_clean.csv   # Dataset limpio con texto enriquecido
│
├── models_cache/              # Modelos pre-entrenados (generados)
│   ├── tfidf_recommender.pkl
│   ├── doc2vec_recommender.pkl
│   ├── sbert_recommender.pkl
│   └── *_classifier.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Carga y preprocesamiento de datos
│   ├── engine.py              # Motor multi-modelo
│   ├── metrics.py             # Métricas de evaluación
│   ├── model_persistence.py   # Persistencia de modelos
│   │
│   └── models/                # Modelos de NLP
│       ├── __init__.py
│       ├── base_model.py      # Clases base abstractas
│       ├── tfidf_model.py     # Modelo TF-IDF
│       ├── doc2vec_model.py   # Modelo Doc2Vec
│       ├── sbert_model.py     # Modelo SBERT
│       └── classifier_models.py # Clasificadores
│
├── notebooks/
│   ├── TP1.ipynb              # Notebook original
│   └── EDA_Netflix.ipynb      # Análisis exploratorio de datos
│
└── docs/                      # Documentación adicional
```

## ⚡ Inicio Rápido

### 1. Clonar e Instalar

```bash
# Clonar repositorio
git clone https://github.com/Jaed69/DataMiningTools-TP.git
cd DataMiningTools-TP

# Crear entorno virtual (recomendado)
conda create -n netflix_rec python=3.10
conda activate netflix_rec

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar Modelos (Primera vez)

```bash
# Entrenar y cachear todos los modelos
python train_models.py
```

Este paso genera los archivos `.pkl` en `models_cache/` para carga rápida.

### 3. Ejecutar la Aplicación

```bash
# Iniciar aplicación Streamlit
streamlit run app_streamlit.py
```

La aplicación estará disponible en: `http://localhost:8501`

## 🧠 Modelos Disponibles

### Recomendadores

| Modelo | Dependencia | Descripción | Velocidad | Precisión |
|--------|-------------|-------------|-----------|-----------|
| **TF-IDF** | ✅ Incluido | Frecuencia de términos + Coseno | ⚡ Muy Rápido | ⭐⭐⭐ |
| **BM25** | `rank-bm25` | Okapi BM25 con normalización | ⚡ Rápido | ⭐⭐⭐⭐ |
| **Doc2Vec** | `gensim` | Embeddings de documentos | 🔄 Medio | ⭐⭐⭐⭐ |
| **SBERT** | `sentence-transformers` | Transformers pre-entrenados | 🐢 Lento (primera vez) | ⭐⭐⭐⭐⭐ |
| **Cross-Encoder** | `sentence-transformers` | Reranking de alta precisión | 🐢 Lento | ⭐⭐⭐⭐⭐ |

### Clasificadores

| Modelo | Descripción | Mejor para |
|--------|-------------|------------|
| **Logistic Regression** | Clasificador lineal | Baseline, datos balanceados |
| **Naive Bayes** | Probabilístico | Texto, alta dimensionalidad |
| **Random Forest** | Ensemble de árboles | Datos con ruido |

## 🔍 Proceso de Análisis

### 1. Exploración de Datos

- Análisis estadístico descriptivo
- Identificación de valores nulos y duplicados
- Visualización de distribuciones

### 2. Limpieza de Datos

- Tratamiento de valores faltantes
- Normalización de texto
- Eliminación de outliers

### 3. Procesamiento de Texto

- **Limpieza estructural**: Eliminación de URLs, HTML, caracteres especiales
- **Tokenización**: Separación de palabras
- **Eliminación de stop words**: Filtrado de palabras comunes
- **Lematización**: Reducción de palabras a su forma base

### 4. Preparación para Modelado

- Combinación de características textuales
- Creación de variables derivadas
- Selección de características relevantes

## 📈 Resultados Clave del Análisis

- **Dataset**: 8,790 títulos únicos
- **Distribución**: 68% películas, 32% series de TV
- **Países principales**: Estados Unidos (32%), India (11%), Reino Unido (5%)
- **Clasificación**: TV-MA (32%) es la clasificación más común
- **Géneros**: Predominan dramas y documentales

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    NETFLIX AI SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Streamlit  │  │   Plotly    │  │   Evaluación        │  │
│  │   UI/UX     │◄─┤   Graphs    │◄─┤   & Métricas        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    MultiModelEngine                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│  │  │ TF-IDF  │  │ Doc2Vec │  │  SBERT  │  Recommenders │  │
│  │  └─────────┘  └─────────┘  └─────────┘              │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│  │  │Logistic │  │  NB     │  │   RF    │  Classifiers │  │
│  │  └─────────┘  └─────────┘  └─────────┘              │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      DataLoader                             │
│         (Dataset limpio + Texto enriquecido)                │
└─────────────────────────────────────────────────────────────┘
```

## 🖥️ Tabs de la Aplicación

| Tab | Descripción |
|-----|-------------|
| 🎬 **Recomendador** | Obtén 5 títulos similares con cualquier algoritmo |
| � **Explicación** | Entiende POR QUÉ cada algoritmo recomienda diferente |
| 🔎 **Búsqueda Semántica** | Busca películas describiendo lo que quieres ver |
| 🏷️ **Clasificador** | Predice géneros para nuevas descripciones |
| 📈 **Evaluación** | Compara métricas cuantitativas de todos los modelos |
| 🎨 **Clustering** | Visualiza agrupaciones de películas con UMAP/t-SNE |
| ⚡ **Benchmark** | Tiempos de entrenamiento e inferencia |
| 📖 **¿Cómo Funciona?** | Explicación técnica de cada algoritmo |
| 📊 **Métricas** | Precision@K, Recall@K, nDCG, MAP detallados |

## 📊 Métricas de Evaluación

### Para Recomendación

- **Precision@K**: Proporción de items relevantes en top-K
- **Recall@K**: Proporción de relevantes encontrados
- **nDCG**: Normalized Discounted Cumulative Gain (considera posición)
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **ILS**: Intra-List Similarity (diversidad)
- **Coverage**: Proporción del catálogo recomendado

### Para Clasificación

- **F1-Score (Micro/Macro)**: Balance precisión-recall
- **Hamming Loss**: Fracción de etiquetas incorrectas
- **Subset Accuracy**: Coincidencia exacta de etiquetas

## 👥 Equipo

| Nombre | Código |
|--------|--------|
| Ricardo Rafael Rivas Carrillo | U202215375 |
| Ian Joaquin Sanchez Alva | U202124676 |
| Jhamil Brijan Peña Cardenas | U201714492 |

**Curso:** Data Mining Tools - Sección 2520  
**Universidad:** Universidad Peruana de Ciencias Aplicadas

## 📄 Licencia

Este proyecto está desarrollado con fines educativos.

---

<div align="center">

**🍿 Netflix AI - Sistema de Recomendación Inteligente**

Desarrollado con ❤️ usando Python, NLP y Machine Learning

</div>