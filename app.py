import gradio as gr
import os
from src.recommender_engine import NetflixSystem

# --- 1. Configuración ---
CSV_PATH = os.path.join("data", "netflix_titles.csv") 

# --- 2. Instanciar el Sistema ---
print("Iniciando aplicación...")
try:
    if not os.path.exists(CSV_PATH):
        # Crear carpeta data si no existe (para evitar error inmediato)
        os.makedirs("data", exist_ok=True)
        print(f"Advertencia: No se encontró {CSV_PATH}. La app iniciará vacía.")
        system = None
    else:
        system = NetflixSystem(CSV_PATH)
        
    if system:
        titulos_disponibles = system.data['title'].tolist()
    else:
        titulos_disponibles = []

except Exception as e:
    print(f"Error al cargar el sistema: {e}")
    system = None
    titulos_disponibles = []

# --- 3. Funciones de Interfaz ---

def interfaz_recomendacion(titulo_seleccionado):
    if not system:
        return "⚠️ El sistema no está cargado. Verifica el archivo CSV."
    if not titulo_seleccionado:
        return "⚠️ Por favor selecciona un título."
    
    recommendations = system.recommend(titulo_seleccionado)
    
    if isinstance(recommendations, list) and len(recommendations) > 0 and isinstance(recommendations[0], str):
         return recommendations[0]

    output_md = f"### 🎬 Títulos similares a: *{titulo_seleccionado}*\n\n"
    for rec in recommendations:
        # Ajuste para manejar si rec tiene 2 o 3 elementos (por si acaso)
        if len(rec) >= 3:
            title, genre, desc = rec[0], rec[1], rec[2]
        else:
            title = rec[0]
            genre = "N/A"
            desc = "No description"

        output_md += f"""
        <div style='background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #e50914;'>
            <h3 style='margin-top: 0; color: white;'>{title}</h3>
            <p style='color: #ccc; font-size: 0.9em;'>🏷️ <i>{genre}</i></p>
            <p style='color: #ddd;'>{desc}</p>
        </div>
        """
    return output_md

def interfaz_clasificacion(sinopsis):
    if not system:
        return None
    if not sinopsis:
        return None
    return system.classify_new_content(sinopsis)

# --- 4. Construcción de la UI ---

# Definir el tema explícitamente
theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="zinc",
).set(
    body_background_fill="#141414",
    body_text_color="white",
    block_background_fill="#1f1f1f",
    block_label_text_color="white",
    button_primary_background_fill="#e50914"
)

# CAMBIO CLAVE: Pasamos el tema al constructor de Blocks correctamente
with gr.Blocks(title="Netflix AI") as demo:
    gr.Markdown(
        """
        # 🍿 Sistema de Recomendación Inteligente
        Este sistema utiliza **NLP** para entender el "ADN" de las películas.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("🔍 Recomendador"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_movie = gr.Dropdown(
                        choices=titulos_disponibles, 
                        label="Película", 
                        filterable=True,
                        info="Escribe para buscar..."
                    )
                    btn_rec = gr.Button("Recomendar", variant="primary")
                
                with gr.Column(scale=2):
                    output_rec = gr.HTML(label="Resultados")
            
            btn_rec.click(fn=interfaz_recomendacion, inputs=input_movie, outputs=output_rec)

        with gr.TabItem("🏷️ Clasificador"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_desc = gr.Textbox(lines=6, label="Sinopsis")
                    btn_cls = gr.Button("Predecir", variant="primary")
                
                with gr.Column(scale=1):
                    output_cls = gr.Label(num_top_classes=3, label="Géneros")
            
            btn_cls.click(fn=interfaz_clasificacion, inputs=input_desc, outputs=output_cls)

if __name__ == "__main__":
    demo.launch()