import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK solo una vez
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self):
        """Carga el dataset y maneja errores básicos."""
        try:
            data = pd.read_csv(self.file_path)
            # Rellenar nulos críticos para que no rompan el código
            data['description'] = data['description'].fillna('')
            data['listed_in'] = data['listed_in'].fillna('')
            data['title'] = data['title'].fillna('Untitled')
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo en: {self.file_path}")

    def clean_text(self, text):
        """Limpia, normaliza y lematiza el texto."""
        if not isinstance(text, str):
            return ""
        
        # 1. Minúsculas
        text = text.lower()
        # 2. Eliminar caracteres especiales
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 3. Tokenización y Lematización
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

    def get_processed_data(self):
        """Pipeline completo: Carga y procesa."""
        df = self.load_data()
        # Aplicamos la limpieza a la descripción
        print("Procesando textos... (esto puede tardar unos segundos)")
        df['processed_description'] = df['description'].apply(self.clean_text)
        
        # Preparamos lista de géneros para el clasificador
        df['genres_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])
        
        return df