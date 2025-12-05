"""
Tab de Información - Explicaciones y documentación.
"""
import streamlit as st


def render_tab_info():
    """Renderiza el tab de información y documentación."""
    
    st.markdown("### 📖 ¿Cómo Funciona el Sistema?")
    
    # Selector de sección
    info_section = st.radio(
        "Selecciona un tema:",
        ["🎯 Concepto Base", "🔄 Comparación de Modelos", "📊 Métricas", "💡 Cuándo Usar Cada Uno"],
        horizontal=True,
        key="info_section"
    )
    
    if info_section == "🎯 Concepto Base":
        _render_concept_section()
    elif info_section == "🔄 Comparación de Modelos":
        _render_models_comparison()
    elif info_section == "📊 Métricas":
        _render_metrics_explanation()
    else:
        _render_usage_guide()


def _render_concept_section():
    """Explica el concepto base del sistema."""
    
    st.markdown("""
    <div style="background: #1f1f1f; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <h4 style="color: #e50914;">🎯 El Concepto Base: Similitud del Coseno</h4>
        <p style="color: #b3b3b3;">
        Todos los modelos convierten texto en <b>vectores numéricos</b> (embeddings).
        Luego calculan qué tan "cerca" están usando la <b>similitud del coseno</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔢 ¿Qué es un Vector?
        
        Imagina que cada película es un punto en el espacio:
        
        ```
        "Stranger Things" → [0.8, 0.2, 0.9, 0.1, ...]
        "Dark"            → [0.7, 0.3, 0.8, 0.2, ...]
        "The Office"      → [0.1, 0.9, 0.2, 0.8, ...]
        ```
        
        Los números representan características semánticas 
        (terror, comedia, drama, etc.)
        """)
    
    with col2:
        st.markdown("""
        ### 📐 Similitud del Coseno
        
        Mide el ángulo entre dos vectores:
        
        | Ángulo | Similitud | Significado |
        |--------|-----------|-------------|
        | 0° | 1.0 | Idénticos |
        | 45° | 0.7 | Similares |
        | 90° | 0.0 | Nada en común |
        
        **Fórmula:** `cos(θ) = (A · B) / (||A|| × ||B||)`
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔄 Proceso de Recomendación
    
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. ENTRADA          2. VECTORIZAR        3. COMPARAR          │
    │  ──────────          ────────────         ─────────            │
    │                                                                 │
    │  "Stranger Things"   [0.8, 0.2, ...]      Calcular similitud   │
    │        │                   │              con TODAS las         │
    │        ▼                   ▼              películas             │
    │  Descripción  ─────▶  Vector 384D  ─────▶      │               │
    │  del título                                    ▼               │
    │                                          Ordenar por           │
    │  4. RESULTADO                            similitud             │
    │  ────────────                                 │                │
    │                                               ▼                │
    │  1. Dark (85%)  ◀─────────────────────── Top K                 │
    │  2. The OA (78%)                                               │
    │  3. Black Mirror (72%)                                         │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)


def _render_models_comparison():
    """Comparación detallada de los modelos."""
    
    st.markdown("## 🔄 Comparación Detallada de Modelos")
    
    # TF-IDF
    with st.expander("🟢 **TF-IDF** - Búsqueda por Palabras Exactas", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### ¿Cómo Funciona?
            
            **TF-IDF = Term Frequency × Inverse Document Frequency**
            
            1. **TF (Term Frequency):** ¿Cuántas veces aparece la palabra?
               - "zombie" aparece 3 veces → peso alto
               
            2. **IDF (Inverse Document Frequency):** ¿Qué tan rara es?
               - "the" aparece en todo → peso bajo
               - "apocalypse" aparece en pocas → peso alto
            
            ### Ejemplo Práctico
            
            ```
            Película A: "Un grupo sobrevive al apocalipsis zombie"
            Película B: "Zombies atacan una ciudad en el apocalipsis"
            
            Palabras en común: "zombie", "apocalipsis"
            → Alta similitud por TF-IDF ✓
            ```
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Ventajas
            - ⚡ Muy rápido
            - 💾 Bajo uso de memoria
            - 🔍 Fácil de interpretar
            - 📊 No requiere entrenamiento
            
            ### ❌ Limitaciones
            - No entiende sinónimos
            - "terror" ≠ "miedo"
            - Solo coincidencias exactas
            """)
    
    # Doc2Vec
    with st.expander("🔵 **Doc2Vec** - Patrones de Escritura", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### ¿Cómo Funciona?
            
            **Red neuronal que aprende el contexto**
            
            1. Entrena con todas las descripciones
            2. Aprende qué palabras aparecen juntas
            3. Cada documento → vector de 100 dimensiones
            
            ### Lo que Captura
            
            - Estructura de las oraciones
            - Patrones de vocabulario
            - "Estilo" de escritura
            
            ### Ejemplo
            
            ```
            Aprende que: "rey" - "hombre" + "mujer" ≈ "reina"
            
            Dos películas con estilo narrativo similar
            tendrán vectores cercanos aunque usen
            palabras diferentes.
            ```
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Ventajas
            - Captura contexto
            - Balance velocidad/precisión
            - Detecta patrones
            
            ### ❌ Limitaciones
            - Requiere entrenamiento
            - Menos preciso que SBERT
            - Necesita muchos datos
            """)
    
    # SBERT
    with st.expander("🔴 **SBERT** - Comprensión Semántica", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### ¿Cómo Funciona?
            
            **Modelo Transformer pre-entrenado (como GPT)**
            
            1. Entrenado con millones de textos de internet
            2. Ya "sabe" el significado de las palabras
            3. Entiende relaciones semánticas complejas
            
            ### Lo que Entiende
            
            | Texto A | Texto B | SBERT dice... |
            |---------|---------|---------------|
            | "película de terror" | "film de miedo" | ≈ Similares |
            | "viaje espacial" | "aventura intergaláctica" | ≈ Similares |
            | "comedia romántica" | "horror sangriento" | ≠ Diferentes |
            
            ### Ejemplo
            
            ```
            Búsqueda: "historia sobre un científico loco"
            
            Encuentra: "Back to the Future" 
            Aunque no contiene esas palabras exactas,
            SBERT entiende que Doc Brown es un científico loco.
            ```
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Ventajas
            - 🧠 Mejor precisión
            - 🔤 Entiende sinónimos
            - 🌐 Pre-entrenado
            - 🔍 Búsqueda semántica
            
            ### ❌ Limitaciones
            - 🐢 Más lento
            - 💾 Usa más memoria
            - 🖥️ Mejor con GPU
            """)
    
    # Tabla comparativa final
    st.markdown("---")
    st.markdown("### 📊 Tabla Comparativa")
    
    st.markdown("""
    | Característica | TF-IDF | Doc2Vec | SBERT |
    |---------------|:------:|:-------:|:-----:|
    | ⚡ Velocidad | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
    | 🎯 Precisión semántica | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    | 💾 Uso de memoria | Bajo | Medio | Alto |
    | 🔤 Entiende sinónimos | ❌ | ⚡ Parcial | ✅ |
    | 🌐 Pre-entrenado | ❌ | ❌ | ✅ |
    | 📦 Dependencias | sklearn | gensim | transformers |
    | 🎓 Curva de aprendizaje | Fácil | Media | Media |
    """)


def _render_metrics_explanation():
    """Explicación detallada de métricas."""
    
    st.markdown("## 📊 Guía Completa de Métricas")
    
    # Precision
    with st.expander("🎯 **Precision@K** - ¿Cuántas recomendaciones son buenas?", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Definición
            
            **De las K recomendaciones, ¿cuántas son relevantes?**
            
            ```
            Precision@K = Relevantes en Top-K / K
            ```
            
            ### Ejemplo
            
            Si K=5 y el modelo recomienda:
            1. ✅ Dark (relevante)
            2. ✅ The OA (relevante)
            3. ❌ Friends (no relevante)
            4. ✅ Black Mirror (relevante)
            5. ❌ The Office (no relevante)
            
            **Precision@5 = 3/5 = 0.60 (60%)**
            """)
        
        with col2:
            st.markdown("""
            ### Interpretación
            
            | Valor | Significado |
            |-------|-------------|
            | 1.0 | Perfecto - todos buenos |
            | 0.8 | Muy bueno |
            | 0.6 | Aceptable |
            | 0.4 | Mejorable |
            | 0.2 | Pobre |
            
            ### Cuándo importa más
            
            Cuando el usuario solo verá 
            las primeras recomendaciones
            (ej: homepage de Netflix)
            """)
    
    # Recall
    with st.expander("📋 **Recall@K** - ¿Cuántas buenas encontramos?", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Definición
            
            **De TODAS las películas relevantes, ¿cuántas capturamos en K?**
            
            ```
            Recall@K = Relevantes en Top-K / Total Relevantes
            ```
            
            ### Ejemplo
            
            Si hay 20 películas relevantes en total, y en K=5 encontramos 4:
            
            **Recall@5 = 4/20 = 0.20 (20%)**
            
            Encontramos el 20% de todas las películas buenas.
            """)
        
        with col2:
            st.markdown("""
            ### Trade-off con Precision
            
            ```
            K pequeño → Alta Precision
                        Bajo Recall
            
            K grande  → Baja Precision
                        Alto Recall
            ```
            
            ### Cuándo importa más
            
            Cuando no queremos perder 
            ninguna opción buena
            (ej: búsqueda exhaustiva)
            """)
    
    # nDCG
    with st.expander("📈 **nDCG@K** - ¿Están los buenos arriba?", expanded=False):
        st.markdown("""
        ### Definición
        
        **Normalized Discounted Cumulative Gain**
        
        Mide si los items relevantes están en las **primeras posiciones** del ranking.
        
        ### ¿Por qué importa el orden?
        
        | Ranking A | Ranking B |
        |-----------|-----------|
        | 1. ✅ Relevante | 1. ❌ No relevante |
        | 2. ✅ Relevante | 2. ❌ No relevante |
        | 3. ❌ No relevante | 3. ✅ Relevante |
        | 4. ❌ No relevante | 4. ✅ Relevante |
        | 5. ❌ No relevante | 5. ❌ No relevante |
        
        **Ambos tienen Precision@5 = 0.40**, pero:
        - Ranking A tiene **nDCG más alto** (buenos arriba)
        - Ranking B tiene **nDCG más bajo** (buenos abajo)
        
        ### Fórmula Simplificada
        
        ```
        DCG = Σ (relevancia / log2(posición + 1))
        
        nDCG = DCG / DCG_ideal
        ```
        
        Posiciones más altas tienen más peso (el log penaliza posiciones bajas).
        """)
    
    # MAP
    with st.expander("📊 **MAP** - Calidad general del ranking", expanded=False):
        st.markdown("""
        ### Definición
        
        **Mean Average Precision**
        
        Calcula la precisión en cada posición donde hay un "hit" y promedia.
        
        ### Ejemplo
        
        ```
        Ranking: ✅ ❌ ✅ ❌ ✅
        
        Posición 1: ✅ → Precision = 1/1 = 1.00
        Posición 3: ✅ → Precision = 2/3 = 0.67
        Posición 5: ✅ → Precision = 3/5 = 0.60
        
        AP = (1.00 + 0.67 + 0.60) / 3 = 0.76
        ```
        
        MAP es el promedio de AP para todas las consultas.
        
        ### Interpretación
        
        - Combina cantidad Y orden de relevantes
        - Más completo que Precision sola
        - Estándar en evaluación de sistemas de búsqueda
        """)
    
    # Parámetros
    st.markdown("---")
    st.markdown("### ⚙️ Cómo Afectan los Parámetros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔢 Top K
        
        | K | Precision | Recall |
        |---|-----------|--------|
        | 3 | Alta | Baja |
        | 10 | Media | Media |
        | 20 | Baja | Alta |
        
        **K pequeño:** Más exigente
        **K grande:** Más permisivo
        """)
    
    with col2:
        st.markdown("""
        #### 🎬 Películas de Prueba
        
        | N | Velocidad | Confianza |
        |---|-----------|-----------|
        | 20 | Rápido | Baja |
        | 50 | Medio | Media |
        | 100 | Lento | Alta |
        
        Más películas = menos ruido
        """)
    
    with col3:
        st.markdown("""
        #### 🎲 Semilla
        
        - **Fija (42):** Reproducible
        - **Variable:** Ver estabilidad
        
        **Tip:** Prueba varias y 
        promedia para resultados
        más robustos.
        """)


def _render_usage_guide():
    """Guía de cuándo usar cada modelo."""
    
    st.markdown("## 💡 ¿Cuándo Usar Cada Algoritmo?")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: #e50914;">🎯 Guía Rápida de Decisión</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabla de decisión
    st.markdown("""
    | Situación | Mejor Opción | Por Qué |
    |-----------|--------------|---------|
    | 🔍 Buscar secuelas/spin-offs | **TF-IDF** | Comparten nombres y términos específicos |
    | 🎨 Películas con "estilo" similar | **Doc2Vec** | Captura patrones de escritura |
    | 🧠 Buscar por tema/concepto | **SBERT** | Entiende el significado real |
    | ⚡ Máxima velocidad | **TF-IDF** | Sin redes neuronales, instantáneo |
    | 🎯 Máxima precisión | **SBERT** | Modelo pre-entrenado avanzado |
    | ⚖️ Balance velocidad/precisión | **Doc2Vec** | Punto medio |
    | 🔤 Usuarios usan sinónimos | **SBERT** | "Terror" = "Miedo" |
    | 📱 Recursos limitados | **TF-IDF** | Mínimo uso de memoria |
    """)
    
    st.markdown("---")
    
    # Casos de uso detallados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #4CAF5022; border: 2px solid #4CAF50; border-radius: 10px; padding: 1rem;">
            <h4 style="color: #4CAF50;">🟢 Usa TF-IDF cuando...</h4>
            <ul style="color: #b3b3b3;">
                <li>Buscas por título exacto</li>
                <li>Necesitas respuesta instantánea</li>
                <li>El usuario usa keywords específicos</li>
                <li>Quieres un baseline simple</li>
                <li>Tienes recursos limitados</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #2196F322; border: 2px solid #2196F3; border-radius: 10px; padding: 1rem;">
            <h4 style="color: #2196F3;">🔵 Usa Doc2Vec cuando...</h4>
            <ul style="color: #b3b3b3;">
                <li>Quieres capturar el "tono"</li>
                <li>Tienes muchos documentos</li>
                <li>Balance es importante</li>
                <li>No quieres dependencias pesadas</li>
                <li>Puedes entrenar el modelo</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #e5091422; border: 2px solid #e50914; border-radius: 10px; padding: 1rem;">
            <h4 style="color: #e50914;">🔴 Usa SBERT cuando...</h4>
            <ul style="color: #b3b3b3;">
                <li>La precisión es crítica</li>
                <li>Usuarios buscan conceptos</li>
                <li>Hay sinónimos frecuentes</li>
                <li>Tienes GPU disponible</li>
                <li>Quieres búsqueda semántica real</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Diagrama de flujo de decisión
    st.markdown("### 🔄 Árbol de Decisión")
    
    st.markdown("""
    ```
    ¿Qué es más importante?
    │
    ├─► VELOCIDAD
    │   └─► TF-IDF ✓
    │
    ├─► PRECISIÓN
    │   └─► ¿Tienes GPU?
    │       ├─► Sí → SBERT ✓
    │       └─► No → Doc2Vec ✓
    │
    └─► BALANCE
        └─► ¿Usuarios usan sinónimos?
            ├─► Sí → SBERT ✓
            └─► No → Doc2Vec ✓
    ```
    """)
    
    # Recomendación final
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #FFD700;">
        <h4 style="color: #FFD700;">💡 Recomendación Final</h4>
        <p style="color: #b3b3b3;">
        En producción, considera usar <b>ensemble</b>: combina las recomendaciones de 
        varios modelos. Si todos coinciden en una película, es muy probable que sea buena recomendación.
        </p>
    </div>
    """, unsafe_allow_html=True)
