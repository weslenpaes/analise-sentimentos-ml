# =========================================
# ANÁLISE DE SENTIMENTOS - MACHINE LEARNING
# =========================================

# IMPORTS
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# =========================================
# CARREGAMENTO DO DATASET
# =========================================

# IMPORTANTE:
# NÃO subir o dataset no GitHub
# Faça o download no Kaggle:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Descomente quando tiver o arquivo local:
# df = pd.read_csv("data/IMDB Dataset.csv")


# 🔹 Dataset simulado (para teste sem arquivo)
df = pd.DataFrame({
    "review": [
        "This movie is amazing",
        "Terrible film, I hated it",
        "Great acting and story",
        "Worst movie ever",
        "I loved this film",
        "This was horrible"
    ],
    "sentiment": ["positive", "negative", "positive", "negative", "positive", "negative"]
})


# =========================================
# FEATURES E TARGET
# =========================================

X = df["review"]
y = df["sentiment"]


# =========================================
# TREINO / TESTE
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# PIPELINE
# =========================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )),
    ("model", LogisticRegression(max_iter=1000))
])


# =========================================
# TREINAMENTO
# =========================================

pipeline.fit(X_train, y_train)


# =========================================
# AVALIAÇÃO
# =========================================

y_pred = pipeline.predict(X_test)

print("\n RELATÓRIO:\n")
print(classification_report(y_test, y_pred))


# =========================================
# FUNÇÃO DE ANÁLISE
# =========================================

def analisar_review(texto):
    predicao = pipeline.predict([texto])[0]
    probabilidades = pipeline.predict_proba([texto])[0]

    print("Review:")
    print(texto)

    print("\nClassificação:", predicao)
    print("Negativo:", round(probabilidades[0], 2))
    print("Positivo:", round(probabilidades[1], 2))
    print("\n" + "=" * 50 + "\n")


# =========================================
# TESTES DO MODELO
# =========================================

print("\n INICIANDO TESTES...\n")


# POSITIVO
analisar_review("""
An absolute masterpiece of modern cinema!
Amazing performances and emotional depth.
""")


# NEGATIVO
analisar_review("""
Terrible movie, boring and poorly written.
I regret watching this.
""")


# AMBÍGUO (tende positivo)
analisar_review("""
The movie is slow in the beginning,
but becomes very engaging later.
""")


# AMBÍGUO (tende negativo)
analisar_review("""
Good visuals, but the story is predictable
and not very engaging.
""")


# =========================================
# SALVAR MODELO
# =========================================

# Opcional: não é obrigatório subir no GitHub
pickle.dump(pipeline, open("modelo_sentimento.pkl", "wb"))

print("\n OK - Modelo treinado e salvo!")