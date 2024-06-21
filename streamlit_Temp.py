import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import glob















# Função para carregar dados
def load_data(path):
    all_files = glob.glob(path + "/*.csv")
    if not all_files:
        raise ValueError(f"No CSV files found in directory: {path}")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df

# Função para ajustar modelo SARIMA
def fit_sarima(df, order, seasonal_order):
    model = SARIMAX(df['target'], order=order, seasonal_order=seasonal_order)
    fit_model = model.fit(disp=False)
    return fit_model

# Função para fazer previsões
def forecast_sarima(model, steps):
    forecast = model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    return forecast_mean, forecast_ci

# Função para plotar gráfico
def plot_forecast(train, test, forecast_index, forecast_mean, forecast_ci):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['target'], label='Treino', color='green')
    ax.plot(test.index, test['target'], label='Observado', color='blue')
    ax.plot(forecast_index, forecast_mean, label='Previsão', color='orange')
    ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.2)
    ax.set_title('Previsão de FinalPrice com SARIMA')
    ax.set_xlabel('Data')
    ax.set_ylabel('FinalPrice')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Configurar Streamlit para ignorar o aviso de depreciação de pyplot global
st.set_option('deprecation.showPyplotGlobalUse', False)

# Carregar dados
path = "amostra_PETS.csv"
df_original = load_data(path)

# Verificar colunas necessárias
required_columns = ['FinalPrice', 'Ano', 'Mês', 'Dia']
for col in required_columns:
    if col not in df_original.columns:
        raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")

# Filtrar e preparar dados
df = df_original[required_columns].copy()
df.rename(columns={'Ano': 'year', 'Mês': 'month', 'Dia': 'day', 'FinalPrice': 'target'}, inplace=True)
df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])
df.set_index('ds', inplace=True)
df = df.resample('D').mean().fillna(method='ffill')

# Separar dados de treino e teste
train = df.iloc[:-30]
test = df.iloc[-30:]

# Ajustar modelo SARIMA
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)
model = fit_sarima(train, order, seasonal_order)

# Fazer previsões
forecast_steps = 30
forecast_mean, forecast_ci = forecast_sarima(model, forecast_steps)
forecast_index = pd.date_range(start=test.index[0], periods=forecast_steps, freq='D')

# Calcular métricas de erro para o atual mês
y_pred = forecast_mean.reindex(test.index)
mae = mean_absolute_error(test['target'], y_pred)
mse = mean_squared_error(test['target'], y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test['target'] - y_pred) / test['target'])) * 100

# Interface do Streamlit
st.title('Previsão de FinalPrice com SARIMA')
st.write("### Dados de Treino e Teste")
st.write(train)
st.write(test)

# Checkbox para exibir previsão do próximo mês
show_next_month_forecast = st.checkbox("Exibir Previsão para o Próximo Mês")

if show_next_month_forecast:
    # Ajustar modelo SARIMA para o próximo mês
    model_next_month = SARIMAX(df['target'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    fit_model_next_month = model_next_month.fit(disp=False)
    
    # Fazer previsões para o próximo mês
    forecast_next_month = fit_model_next_month.get_forecast(steps=31)  # 31 dias para todo o mês de julho
    forecast_mean_next_month = forecast_next_month.predicted_mean
    forecast_ci_next_month = forecast_next_month.conf_int()
    forecast_index_next_month = pd.date_range(start='2023-07-01', periods=len(forecast_mean_next_month), freq='D')
    
    # Mostrar gráfico para o próximo mês
    st.write("### Previsão para o Próximo Mês")
    plot_forecast(train, test, forecast_index_next_month, forecast_mean_next_month, forecast_ci_next_month)

    # Calcular métricas de erro para o próximo mês
    # Ajuste para utilizar apenas os dados onde há correspondência
    if len(test) == len(forecast_mean_next_month):
        mae_next_month = mean_absolute_error(test['target'], forecast_mean_next_month)
        mse_next_month = mean_squared_error(test['target'], forecast_mean_next_month)
        rmse_next_month = np.sqrt(mse_next_month)
        mape_next_month = np.mean(np.abs((test['target'] - forecast_mean_next_month) / test['target'])) * 100
    
        st.write("### Métricas de Erro para o Próximo Mês")
        st.write(f"Erro Médio Absoluto (MAE): {mae_next_month:.2f}")
        st.write(f"Erro Quadrático Médio (MSE): {mse_next_month:.2f}")
        st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse_next_month:.2f}")
        st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape_next_month:.2f}%")
    else:
        st.write("Não foi possível calcular as métricas de erro para o próximo mês devido à falta de dados correspondentes.")

# Métricas de erro para o atual mês (já exibidas acima)
st.write("### Métricas de Erro para o Mês Atual")
st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
st.write(f"Erro Quadrático Médio (MSE): {mse:.2f}")
st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape:.2f}%")
