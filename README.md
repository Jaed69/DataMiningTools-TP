# Sistema de Recomendación de Contenido Netflix 🎬

Un sistema de recomendación basado en contenido para películas y series de Netflix, utilizando técnicas de Minería de Textos y Procesamiento de Lenguaje Natural (PNL).

## 📋 Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar un sistema de recomendación basado en el contenido para películas y series de la plataforma Netflix. El modelo analiza las características descriptivas de cada título —como género, reparto, dirección y sinopsis— para identificar similitudes entre producciones y sugerir contenidos que se ajusten a los intereses del usuario.

## 🚀 Características Principales

- **Análisis de texto avanzado**: Procesamiento de sinopsis con técnicas de NLP
- **Múltiples características**: Consideración de directores, elenco, géneros y descripciones
- **Limpieza de datos**: Manejo de valores nulos y normalización de texto
- **Sistema de similitud**: Implementación de modelos de similitud basados en contenido

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
├── README.md                 # Documentación del proyecto
├── TP1.ipynb                 # Notebook principal con el análisis
├── netflix_titles.csv        # Dataset de títulos de Netflix
└── Data Mining Tools.pdf     # Documentación adicional del proyecto
```

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

## 🚧 Próximos Pasos

1. **Implementación del modelo de similitud** usando TF-IDF o embeddings
2. **Sistema de recomendación** basado en similitud de contenido
3. **Interfaz de usuario** para consultas y recomendaciones
4. **Evaluación del modelo** con métricas de precisión y recall

## 📝 Uso del Proyecto

1. Clona el repositorio
2. Instala las dependencias requeridas
3. Ejecuta el notebook `TP1.ipynb` para ver el análisis completo
4. El dataset `netflix_titles.csv` debe estar en el directorio raíz

## 👥 Contribuciones

Este es un proyecto académico enfocado en el aprendizaje de técnicas de Data Mining y NLP aplicadas a sistemas de recomendación.

## 📄 Licencia

Este proyecto está desarrollado con fines educativos como parte del curso de Data Mining Tools.

---

*Proyecto desarrollado utilizando técnicas de Minería de Datos y Procesamiento de Lenguaje Natural para la construcción de sistemas de recomendación inteligentes.*