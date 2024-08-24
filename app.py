from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy
import os
import subprocess  # Importar subprocess para ejecutar comandos del sistema

# Descargar stopwords de nltk
nltk.download('stopwords')

# Comprobación y descarga del modelo SpaCy si no está instalado
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Descargar el modelo si no está disponible
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Cargar stopwords en inglés
stop_words = set(stopwords.words('english'))

# Inicializar el analizador de sentimiento
sa = SentimentIntensityAnalyzer()

# Cargar el dataset de IMDB con manejo de errores
try:
    df = pd.read_csv('IMDB Dataset.csv')

    # Verificar si la columna 'review' existe
    if 'review' not in df.columns:
        raise KeyError("La columna 'review' no se encuentra en el dataset.")

    # Preprocesar las reseñas del dataset: eliminar stopwords y calcular puntajes de sentimiento
    df['processed_review'] = df['review'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))
    df['sentiment_score'] = df['processed_review'].apply(lambda x: sa.polarity_scores(x)['compound'])
    # Calcular el sentimiento promedio
    avg_sentiment = df['sentiment_score'].mean()
except FileNotFoundError:
    # Si el archivo no se encuentra, inicializar un DataFrame vacío
    df = pd.DataFrame({'review': [], 'processed_review': [], 'sentiment_score': []})
    avg_sentiment = 0  # Sentimiento promedio por defecto si no hay datos
    print("Error: 'IMDB Dataset.csv' no encontrado. Asegúrate de que el archivo esté en el directorio correcto.")
except KeyError as e:
    # Si la columna 'review' no está en el CSV, inicializar un DataFrame vacío
    df = pd.DataFrame({'review': [], 'processed_review': [], 'sentiment_score': []})
    avg_sentiment = 0  # Sentimiento promedio por defecto si no hay datos
    print(f"Error: {e}")

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    # Capturar el texto ingresado por el usuario
    text1 = request.form['text1'].lower()

    # Preprocesamiento: eliminar stopwords
    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    # Análisis de sentimiento de la reseña del usuario
    sentiment_scores = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + sentiment_scores['compound']) / 2, 2)  # Convertir a escala 0-1

    # Comparar con el sentimiento promedio del dataset
    comparison = "más positiva" if compound > avg_sentiment else "más negativa"

    # Extracción de entidades usando SpaCy
    doc = nlp(processed_doc1)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Guardar entidades y etiquetas

    # Resumen del análisis de sentimiento
    sentiment_class = "positiva" if compound >= 0.5 else "negativa"

    # Renderizar la página con los resultados
    return render_template('form.html', 
                           final=compound, 
                           avg_sentiment=round(avg_sentiment, 2),
                           comparison=comparison,
                           text1=text1, 
                           entities=entities, 
                           sentiment_class=sentiment_class)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto proporcionado por Railway o 5000 como predeterminado
    app.run(host="0.0.0.0", port=port)

