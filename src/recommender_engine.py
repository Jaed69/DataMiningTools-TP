import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from src.data_loader import DataLoader

class NetflixSystem:
    def __init__(self, data_path):
        """
        Al inicializar la clase, se cargan los datos y se entrenan los modelos.
        Esto ocurre UNA VEZ al inicio, garantizando velocidad después.
        """
        print("Inicializando sistema... Cargando datos.")
        self.loader = DataLoader(data_path)
        self.data = self.loader.get_processed_data()
        
        # Inicializar componentes
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.cosine_sim = None
        self.classifier = None
        self.mlb = MultiLabelBinarizer()
        self.indices = None
        
        # Entrenar automáticamente al iniciar
        self._train_models()

    def _train_models(self):
        """Entrena TF-IDF, Matriz de Similitud y Clasificador."""
        print("Entrenando modelos NLP...")
        
        # 1. Vectorización (TF-IDF)
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['processed_description'])
        
        # 2. Matriz de Similitud (Recomendador)
        # Nota: Para 8000 datos es rápido. Para millones, se usaría NearestNeighbors.
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Crear índice reverso para buscar rápido por título
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()
        
        # 3. Clasificador (Regresión Logística Multietiqueta)
        print("Entrenando clasificador de géneros...")
        y = self.mlb.fit_transform(self.data['genres_list'])
        self.classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
        self.classifier.fit(self.tfidf_matrix, y)
        
        print("¡Sistema listo!")

    def recommend(self, title):
        """Lógica de recomendación."""
        if title not in self.indices:
            return [f"Error: La película '{title}' no se encuentra en el catálogo."]

        idx = self.indices[title]
        
        # Calcular similitud
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Top 5 (excluyendo el propio título)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        
        # Devolver resultados con su género para dar más contexto
        result = self.data.iloc[movie_indices][['title', 'listed_in', 'description']]
        return result.values.tolist() # Convertir a lista para Gradio

    def classify_new_content(self, synopsis):
        """Lógica de clasificación para texto nuevo."""
        # Limpiar texto nuevo
        processed_text = self.loader.clean_text(synopsis)
        # Vectorizar
        vec = self.tfidf.transform([processed_text])
        # Predecir probabilidades
        probs = self.classifier.predict_proba(vec)[0]
        
        # Obtener top 3 índices
        top_indices = probs.argsort()[-3:][::-1]
        
        # Mapear a nombres de géneros y sus confidencias
        results = {self.mlb.classes_[i]: float(probs[i]) for i in top_indices}
        return results