import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import statsmodels.api as sm

# Função para ler e concatenar todos os arquivos CSV no diretório especificado
def load_data(path):
    all_files = glob.glob(path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df

# Caminho para os arquivos CSV
path = "amostra_PETS.csv"

# Carregar os dados
df = load_data(path)


# Função para filtrar valores negativos
def filter_negative_prices(df):
    df = df[(df['FinalPrice'] >= 0) & (df['SuggestedPrice'] >= 0)]
    return df

df = filter_negative_prices(df)



# FILTROS

# Filtrar por mês e dia
st.sidebar.title('Filtros')
meses = st.sidebar.multiselect('Selecione o(s) Mês(es)', sorted(df['Mês'].unique()))
dias = st.sidebar.multiselect('Selecione o(s) Dia(s)', sorted(df['Dia'].unique()))

# Aplicar filtros
if meses:
    df_filtrado = df[df['Mês'].isin(meses)]
else:
    df_filtrado = df.copy()  # Se nenhum mês selecionado, usar o DataFrame completo
    
if dias:
    df_filtrado = df_filtrado[df_filtrado['Dia'].isin(dias)]

# Filtro de seleção de retailer
retailer = st.sidebar.selectbox('Selecione o Retailer', ['Todos'] + sorted(df['Retailer'].unique()))

if retailer != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Retailer'] == retailer]

# Filtro de seleção de produto
produto = st.sidebar.selectbox('Selecione o Produto', ['Todos'] + sorted(df['Product'].unique()))

if produto != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Product'] == produto]


# Filtro para escolher entre Disponível e Indisponível
disponibilidade = st.sidebar.selectbox('Escolha a Disponibilidade', ['Todos', 'Disponível', 'Indisponível'])

# Mapear valores de 'Available' para 'Disponível' e 'Indisponível'
df_filtrado['Disponibilidade'] = df_filtrado['Available'].map({1: 'Disponível', 0: 'Indisponível'})

# Aplicar filtro de disponibilidade
if disponibilidade != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Disponibilidade'] == disponibilidade]


# Converter 'Dia' para tipo inteiro
df_filtrado['Dia'] = pd.to_numeric(df_filtrado['Dia'], errors='coerce')  # Convertendo para numérico


# Título do painel
st.title('Análise de Dados de Produtos')

# Subtítulo e visualização dos dados
st.subheader('Visualização dos Dados')
st.write(df_filtrado.head())  # Exibir os primeiros registros para verificação




#Graficos



# Gráfico de dispersão (scatter plot) para SuggestedPrice versus FinalPrice
fig_scatter = px.scatter(df_filtrado, x='SuggestedPrice', y='FinalPrice', color='Available',
                         hover_data=['RandomPrecosNegativos', 'RandomPrecosDiscrepantes'],
                         labels={'SuggestedPrice': 'Preço Sugerido', 'FinalPrice': 'Preço Final'})
st.plotly_chart(fig_scatter)






# Gerar gráfico de barras
df_filtrado['Disponibilidade'] = df_filtrado['Available'].map({1: 'Disponível', 0: 'Indisponível'})
df_estoque = df_filtrado.groupby(['Dia', 'Mês', 'Product', 'Disponibilidade']).size().reset_index(name='Contagem')
fig_estoque = px.bar(df_estoque, x='Dia', y='Contagem', color='Disponibilidade', barmode='stack',
                     labels={'Contagem': 'Quantidade', 'Dia': 'Dia do Mês'},
                     title='Estoque por Dia e Disponibilidade')
st.plotly_chart(fig_estoque)




# Calcular o estoque disponível por retailer
df_estoque_retailer = df_filtrado.groupby(['Retailer', 'Available']).size().reset_index(name='Contagem')

# Mapear valores de 'Available' para 'Disponível' e 'Indisponível'
df_estoque_retailer['Disponibilidade'] = df_estoque_retailer['Available'].map({1: 'Disponível', 0: 'Indisponível'})

# Ordenar pela contagem em ordem decrescente para facilitar a visualização
df_estoque_retailer = df_estoque_retailer.sort_values(by='Contagem', ascending=False)

# Gráfico de barras empilhadas para estoque por retailer e disponibilidade
fig_estoque_retailer = px.bar(df_estoque_retailer, x='Retailer', y='Contagem', color='Disponibilidade',
                              title='Estoque Disponível por Retailer',
                              labels={'Contagem': 'Quantidade de Produtos', 'Retailer': 'Retailer', 'Disponibilidade': 'Disponibilidade'},
                              barmode='stack')  # Empilhamento das barras

# Exibir o gráfico
st.plotly_chart(fig_estoque_retailer)





# Verificar e limpar dados não numéricos em 'FinalPrice'
df_filtrado['FinalPrice'] = pd.to_numeric(df_filtrado['FinalPrice'], errors='coerce')

# Verificar se há NaNs após a conversão
if df_filtrado['FinalPrice'].isnull().any():
    st.error('Existem valores não numéricos na coluna FinalPrice. Verifique e ajuste seus dados.')

# Calcular a média dos preços finais por dia e produto
try:
    df_media_final_price = df_filtrado.groupby(['Dia', 'Product'])['FinalPrice'].mean().reset_index()
    
    # Gráfico de linha para Média de Preço Final por Dia
    fig_media_final_price = px.line(df_media_final_price, x='Dia', y='FinalPrice', color='Product',
                                   labels={'Dia': 'Dia do Mês', 'FinalPrice': 'Média de Preço Final'},
                                   title='Média de Preço Final por Dia')
    
    st.plotly_chart(fig_media_final_price)

except Exception as e:
    st.error(f"Erro ao calcular a média dos preços finais: {str(e)}")




# Calcular vendas por produto e dia
df_vendas = df_filtrado.groupby(['Dia', 'Product']).size().reset_index(name='Vendas')

# Ordenar produtos por número de vendas decrescente
df_vendas_sorted = df_vendas.sort_values(by='Vendas', ascending=False)

# Gráfico de barras para produtos mais vendidos
fig_vendas = px.bar(df_vendas_sorted, x='Product', y='Vendas', color='Product',
                    labels={'Product': 'Produto', 'Vendas': 'Número de Vendas'},
                    title='Produtos Mais Vendidos por Dia')
st.plotly_chart(fig_vendas)







# Gráfico de barras para vendas totais por retailer
# Ordenar por vendas totais em ordem decrescente
vendas_totais_retailer = df_filtrado.groupby('Retailer').size().reset_index(name='Vendas Totais')

vendas_totais_retailer = vendas_totais_retailer.sort_values(by='Vendas Totais', ascending=False)
fig_vendas_totais = px.bar(vendas_totais_retailer, x='Retailer', y='Vendas Totais',
                           labels={'Retailer': 'Retailer', 'Vendas Totais': 'Número de Vendas'},
                           title='Vendas Totais por Retailer')
st.plotly_chart(fig_vendas_totais)




# Gerar gráfico de barras empilhadas para produtos mais vendidos por retailer
vendas_por_retailer_produto = df_filtrado.groupby(['Retailer', 'Product']).size().reset_index(name='Vendas')
produtos_mais_vendidos = vendas_por_retailer_produto.loc[vendas_por_retailer_produto.groupby('Retailer')['Vendas'].idxmax()]
produtos_mais_vendidos = produtos_mais_vendidos.sort_values(by='Vendas', ascending=False)

fig_produtos_mais_vendidos = px.bar(produtos_mais_vendidos, x='Retailer', y='Vendas', color='Product',
                                   labels={'Retailer': 'Retailer', 'Vendas': 'Número de Vendas', 'Product': 'Produto'},
                                   title='Produtos Mais Vendidos por Retailer',
                                   barmode='stack')  # Empilhamento das barras
st.plotly_chart(fig_produtos_mais_vendidos)






# Calcular a quantidade de produtos vendidos por retailer
vendas_por_retailer = df_filtrado.groupby(['Retailer', 'Product']).size().reset_index(name='Quantidade de Produtos Vendidos')

# Calcular a quantidade total de produtos vendidos por retailer
vendas_totais_retailer = vendas_por_retailer.groupby('Retailer')['Quantidade de Produtos Vendidos'].sum().reset_index()
vendas_totais_retailer = vendas_totais_retailer.sort_values(by='Quantidade de Produtos Vendidos', ascending=False)

# Ordenar o DataFrame original pela quantidade total de produtos vendidos
vendas_por_retailer = vendas_por_retailer.merge(vendas_totais_retailer[['Retailer']], on='Retailer')
vendas_por_retailer = vendas_por_retailer.sort_values(by=['Quantidade de Produtos Vendidos', 'Retailer'], ascending=[False, True])

# Gráfico de barras empilhadas para quantidade de produtos vendidos por retailer
fig_vendas_por_retailer = px.bar(vendas_por_retailer, x='Retailer', y='Quantidade de Produtos Vendidos', color='Product',
                                 labels={'Quantidade de Produtos Vendidos': 'Quantidade de Produtos Vendidos', 'Product': 'Produto'},
                                 title='Quantidade de Produtos Vendidos por Retailer',
                                 barmode='stack')  # Empilhamento das barras
st.plotly_chart(fig_vendas_por_retailer)


















