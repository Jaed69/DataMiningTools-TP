import pandas as pd
import re
import os
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
        
        # Verificar si existe versión limpia del dataset
        clean_path = file_path.replace('.csv', '_clean.csv')
        if os.path.exists(clean_path):
            self.file_path = clean_path
            self.use_enriched = True
            print(f"✅ Usando dataset limpio y enriquecido: {clean_path}")
        else:
            self.use_enriched = False

    def load_data(self):
        """Carga el dataset y maneja errores básicos."""
        try:
            data = pd.read_csv(self.file_path)
            # Rellenar nulos críticos para que no rompan el código
            data['description'] = data['description'].fillna('')
            data['listed_in'] = data['listed_in'].fillna('')
            data['title'] = data['title'].fillna('Untitled')
            data['director'] = data['director'].fillna('')
            data['cast'] = data['cast'].fillna('')
            data['country'] = data['country'].fillna('')
            data['rating'] = data['rating'].fillna('Not Rated')
            
            # Si tiene enriched_text, asegurar que no tenga nulos
            if 'enriched_text' in data.columns:
                data['enriched_text'] = data['enriched_text'].fillna(data['description'])
            
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
    
    def create_enriched_text(self, row):
        """
        Crea un texto enriquecido combinando las mejores características.
        Formato: descripción + géneros + director + actores principales
        """
        parts = []
        
        # 1. Descripción (siempre presente)
        desc = str(row.get('description', '')).strip()
        if desc:
            parts.append(desc)
        
        # 2. Géneros (muy importante para similitud)
        genres = str(row.get('listed_in', '')).strip()
        if genres and genres != 'nan':
            parts.append(f"Género: {genres}")
        
        # 3. Director (si disponible)
        director = str(row.get('director', '')).strip()
        if director and director != 'nan':
            parts.append(f"Director: {director}")
        
        # 4. Cast - solo primeros 3 actores (si disponible)
        cast = str(row.get('cast', '')).strip()
        if cast and cast != 'nan':
            actors = [a.strip() for a in cast.split(',')][:3]  # Solo primeros 3
            parts.append(f"Actores: {', '.join(actors)}")
        
        return ' '.join(parts)

    def get_processed_data(self):
        """Pipeline completo: Carga y procesa."""
        df = self.load_data()
        
        # Si ya tiene enriched_text, usarlo directamente
        if 'enriched_text' in df.columns and self.use_enriched:
            print("📊 Usando texto enriquecido pre-calculado...")
            df['processed_description'] = df['enriched_text'].apply(self.clean_text)
        else:
            # Crear texto enriquecido si no existe
            print("📊 Creando texto enriquecido...")
            df['enriched_text'] = df.apply(self.create_enriched_text, axis=1)
            df['processed_description'] = df['enriched_text'].apply(self.clean_text)
        
        # Preparamos lista de géneros para el clasificador
        df['genres_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
        
        print(f"✅ Procesados {len(df)} títulos con texto enriquecido")
        
        return df