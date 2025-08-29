import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Acessa o arquivo 'ecommerce_preparados.csv' do ambiente
df = pd.read_csv("C:/Users/flavi/Downloads/ecommerce_preparados.csv")

# Renomeia as colunas para evitar KeyErrors
df = df.rename(columns={'N_Avaliações': 'N_Avaliacoes'})

print('DataFrame inicial:')
print(df.head().to_string())
print(df.dtypes)

# Tratamento de dados
# Mapeamento dos valores para que a conversão de 'Qtd_Vendidos' funcione
qtd_vendidos_cod = {
    'Nenhum': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '+5': 5, '+25': 25, '+50': 50, '+100': 100,
    '+1000': 1000, '+10mil': 10000, '+50mil': 50000
}
df['Qtd_Vendidos'] = df['Qtd_Vendidos'].map(qtd_vendidos_cod)
df['Qtd_Vendidos'] = pd.to_numeric(df['Qtd_Vendidos'], errors='coerce').fillna(0)

# A coluna 'Marca' é convertida para código numérico para outros gráficos (pairplot, regressão)
df['Marca_encoded'] = df['Marca'].astype('category').cat.codes

print('\nDataFrame após tratamento de dados:')
print(df.head().to_string())

# --- Geração dos Gráficos ---

# Gráfico de Pairplot - Dispersão e Histograma
plt.figure()
sns.pairplot(df[['Qtd_Vendidos', 'Marca_encoded', 'N_Avaliacoes']])
plt.suptitle('Pairplot - Quantidade de Vendas, Marcas e Avaliações', y=1.02)
plt.show()

# Gráfico de Dispersão entre Marca e Quantidade de Vendas
plt.figure()
plt.scatter(df['Marca_encoded'], df['Qtd_Vendidos'])
plt.title('Dispersão - Marca e Quantidade de Vendas')
plt.xlabel('Marca')
plt.ylabel('Quantidade de Vendas')
plt.show()

# Mapa de Calor entre Marca e Quantidade de Vendas
plt.figure()
corr = df[['Marca_encoded', 'Qtd_Vendidos']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação Marca e Quantidade de Vendas')
plt.show()

# Gráfico de Barras Correlacionando Marca e Total de Avaliações
plt.figure(figsize=(12, 8))
# Agrupa por 'Marca' (os nomes originais)
top_marcas_avaliacoes = df.groupby('Marca')['N_Avaliacoes'].sum().sort_values(ascending=False).head(5)
top_marcas_avaliacoes.plot(kind='barh', color='#90ee70')
plt.title('Top 5 Marcas por Total de Avaliações')
plt.ylabel('Marcas') # O rótulo agora se refere aos nomes
plt.xlabel('Total de Avaliações')
plt.tight_layout()
plt.show()

# Gráfico de pizza
plt.figure(figsize=(10, 8))
# Usa o `index` (nomes das marcas) para os rótulos do gráfico de pizza
plt.pie(top_marcas_avaliacoes.values, labels=top_marcas_avaliacoes.index, autopct='%.1f%%', startangle=90)
plt.title('Top 5 Marcas - Distribuição de Avaliações')
plt.tight_layout()
plt.show()

# Gráfico de Densidade
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Marca_encoded'], fill=True, color='#863e9c')
plt.title('Densidade de Marcas')
plt.xlabel('Marca')
plt.ylabel('Densidade')
plt.show()

# Gráfico de Regressão
plt.figure(figsize=(10, 6))
sns.regplot(x='N_Avaliacoes', y='Marca_encoded', data=df, color='#278f65', scatter_kws={'alpha': 0.5, 'color': '#34c289'})
plt.title('Regressão de Marca por Avaliações')
plt.xlabel('Avaliações')
plt.ylabel('Marca')
plt.show()
