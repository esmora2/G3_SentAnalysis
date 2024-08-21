# Análisis de Sentimiento y Extracción de Entidades

Este proyecto es una aplicación web creada con Flask que permite a los usuarios ingresar reseñas de películas y obtener un análisis de sentimiento junto con la extracción de entidades clave. Además, el proyecto utiliza un dataset de reseñas de películas para realizar un análisis agregado del sentimiento promedio y comparar el sentimiento de la reseña del usuario con el promedio del dataset.

## Requisitos

Asegúrate de tener Python 3.6 o superior instalado. Este proyecto requiere las siguientes dependencias, que se pueden instalar usando el archivo `requirements.txt` proporcionado.

### Dependencias

- Flask
- pandas
- scikit-learn
- vaderSentiment
- spacy
- nltk
- en-core-web-sm (modelo de SpaCy)

## Instalación

1. **Clona el repositorio o descarga el proyecto:**

   ```bash
   git clone https://github.com/esmora2/G3_SentAnalysis.git
   cd G3_SentAnalysis

## Ejecucion
2. Crea un entorno virtual y activa el entorno:
    python -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate

## Dependencias

3. Instala las dependencias:
    pip install -r requirements.txt

4. Descarga el modelo SpaCy
    python -m spacy download en_core_web_sm

## Dataset de Reseñas

    Asegúrate de tener el archivo IMDB Dataset.csv en la raíz del proyecto. Puedes encontrar este dataset en Kaggle, siguiendo el enlace:

    https://www.kaggle.com/code/bhu1111/sentiment-analysis-movies-review/input?select=IMDB+Dataset.csv

## Uso

    Ejecuta la aplicación Flask:

    python app.py

    Accede a la aplicación en tu navegador:

    Abre tu navegador y visita http://127.0.0.1:5002.

## Interacción con la aplicación

    Escribe una reseña de película en el área de texto y haz clic en "Enviar".
    La aplicación mostrará el análisis de sentimiento de tu reseña y destacará las entidades clave encontradas.
    Además, la aplicación proporcionará un análisis comparativo del sentimiento de tu reseña con el promedio de las reseñas en el dataset.
