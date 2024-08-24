from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy
import os  # Importar os para acceder a variables de entorno

if not os.path.exists(spacy.util.get_data_path() / "en_core_web_sm"):
    import spacy.cli
    spacy.cli.download("en_core_web_sm")


nltk.download('stopwords')

# Cargar stopwords en inglés
stop_words = set(stopwords.words('english'))

# Inicializar el analizador de sentimiento y SpaCy para la extracción de entidades
sa = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

# Cargar el dataset de IMDB
df = pd.read_csv('IMDB Dataset.csv')

# Preprocesar las reseñas del dataset: eliminar stopwords y calcular puntajes de sentimiento
df['processed_review'] = df['review'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))
df['sentiment_score'] = df['processed_review'].apply(lambda x: sa.polarity_scores(x)['compound'])

# Calcular el sentimiento promedio
avg_sentiment = df['sentiment_score'].mean()

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
    compound = round((1 + sentiment_scores['compound'])/2, 2)  # Convertir a escala 0-1

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
    port = int(os.environ.get('PORT', 5000))  # Usar el puerto proporcionado por Railway o 5000 como valor por defecto
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
