import pandas as pd
import glob
import joblib
import streamlit as st
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
from sklearn.tree import plot_tree
import numpy as np


# Definição da classe IsoForestTransformer
class IsoForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)

    def fit(self, X, y=None):
        self.iso_forest.fit(X)
        return self

    def transform(self, X):
        # Transformar os valores de -1, 1 para 0, 1
        return ((self.iso_forest.predict(X) == -1) * 1).reshape(-1, 1)

# Função para ler e concatenar todos os arquivos CSV no diretório especificado
def load_data(path):
    all_files = glob.glob(path + "/*.csv")
    if not all_files:
        raise ValueError(f"No CSV files found in directory: {path}")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df

# Caminho para os arquivos CSV
path = "amostra_PETS.csv"

# Carregar os dados
try:
    df = load_data(path)
except ValueError as e:
    st.error(e)
    st.stop()

# Verifique se os arquivos dos modelos existem
model_paths = {
    'Decision Tree': 'best_model_dt.joblib',
    'KNN': 'best_model_knn.joblib'
}

missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
if missing_models:
    st.error(f"Model files not found: {', '.join(missing_models)}")
    st.stop()

# Interface do usuário no Streamlit
st.title("Previsão de Disponibilidade de Produto")

# Seleção do modelo
model_choice = st.selectbox("Escolha o Modelo", list(model_paths.keys()))

# Carregar o modelo com tratamento de erro
try:
    model = joblib.load(model_paths[model_choice])
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# Fazer previsão com dados do DataFrame
try:
    input_data = df.drop('Available', axis=1)  # Substitua 'Available' pela coluna real do alvo
    y_true = df['Available']  # Substitua 'Available' pela coluna real do alvo
    y_pred = model.predict(input_data)

    st.write('Previsões feitas para todos os dados carregados.')
    st.write('Disponível' if y_pred[0] == 1 else 'Indisponível')
except Exception as e:
    st.error(f"Failed to make a prediction: {e}")

# Calcular e exibir precisão e acurácia
try:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1)  # Supondo que '1' é a classe positiva

    st.write(f"Acurácia do Modelo: {accuracy:.2f}")
    st.write(f"Precisão do Modelo: {precision:.2f}")
except Exception as e:
    st.error(f"Failed to calculate accuracy and precision: {e}")

# Visualizar a matriz de confusão
st.subheader("Matriz de Confusão")
try:
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Indisponível', 'Disponível'])

    fig, ax = plt.subplots()
    cmd.plot(ax=ax)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Failed to generate confusion matrix: {e}")

# Mostrar as variáveis que mais impactaram o modelo
st.subheader("Importância das Variáveis")
try:
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_

        # Obter os nomes das características após o pré-processamento
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()

        # Adicionar o nome da nova coluna criada pelo IsoForestTransformer
        feature_names = list(feature_names) + ['iso_forest']

        # Criar um DataFrame com as importâncias e os nomes das características
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        # Ordenar o DataFrame pelas importâncias
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        # Selecionar as top 5 características
        top_features = feature_importances.head(5)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 5 Características Mais Importantes no Modelo de Árvore de Decisão')
        plt.xlabel('Importância')
        plt.ylabel('Característica')
        st.pyplot(plt)
    elif hasattr(model.named_steps['classifier'], 'coef_'):
        coef = model.named_steps['classifier'].coef_[0]
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coef)})

        # Ordenar o DataFrame pelas importâncias
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        # Selecionar as top 5 características
        top_features = feature_importances.head(5)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 5 Características Mais Importantes no Modelo KNN')
        plt.xlabel('Importância')
        plt.ylabel('Característica')
        st.pyplot(plt)
except Exception as e:
    st.error(f"Failed to plot feature importances: {e}")

# Plotar a árvore de decisão, se aplicável
if model_choice == 'Decision Tree':
    st.subheader("Árvore de Decisão")
    try:
        fig, ax = plt.subplots(figsize=(50, 20))
        plot_tree(model.named_steps['classifier'], 
                  feature_names=feature_names, 
                  class_names=['Indisponível', 'Disponível'], 
                  filled=True, 
                  rounded=True,
                  ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to plot the decision tree: {e}")
