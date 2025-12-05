import streamlit as st
import pandas as pd
import kagglehub
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Google Play Store Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------
# 1. FUNÇÕES DE LIMPEZA E CARREGAMENTO
# ------------------------------------
@st.cache_data
def load_data():
    """
    Baixa, carrega e pré-processa o dataset do Google Play Store do Kaggle.
    Contém a limpeza de dados robusta para resolver o erro 'ValueError: could not convert string to float'.
    Requer as variáveis KAGGLE_USERNAME e KAGGLE_KEY configuradas nos Secrets do Streamlit Cloud.
    """
    try:
        st.write("Baixando dataset do Kaggle Hub...")
        
        # Tenta baixar o dataset usando as credenciais configuradas no ambiente
        dataset_path = kagglehub.dataset_download("lava18/google-play-store-apps")
        file_path = f"{dataset_path}/googleplaystore.csv"
        
        # Carrega o CSV
        df = pd.read_csv(file_path)
        
    except Exception as e:
        # Mensagem de erro robusta para falha de autenticação
        st.error(f"Erro ao carregar os dados do Kaggle Hub. Verifique se as chaves KAGGLE_USERNAME e KAGGLE_KEY estão configuradas corretamente nos Secrets do Streamlit Cloud.")
        st.error(f"Detalhe do erro: {e}")
        return pd.DataFrame()

    # Limpeza e Pré-processamento
    
    # 1. TRATAMENTO DE LINHAS CORROMPIDAS
    # Remove as linhas conhecidas que corrompem o dataset:
    # - 'Life Made Better' (linha com Category=1.9 que desalinhou colunas)
    # - A linha onde 'Category' é '1.9' (segunda ocorrência do problema)
    df.drop(df[df['App'] == 'Life Made Better'].index, inplace=True)
    df.drop(df[df['Category'] == '1.9'].index, inplace=True)
    
    # 2. LIMPEZA E CONVERSÃO ROBUSTA DE 'Installs'
    # Remove caracteres (+ e ,)
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=True)
    # CORREÇÃO CRÍTICA: pd.to_numeric com errors='coerce' converte strings inválidas (como 'Free' desalinhado) em NaN
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # 3. LIMPEZA E CONVERSÃO DE 'Reviews'
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # 4. LIMPEZA E CONVERSÃO DE 'Price'
    # Remove o '$' antes de converter
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=True)
    # CORREÇÃO: Converte strings (incluindo possíveis desalinhamentos) em NaN se não forem números
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # 5. LIMPEZA DE 'Size' (para evitar problemas futuros)
    # Esta é uma etapa extra de robustez
    df['Size'] = df['Size'].astype(str).str.replace('M', '', regex=False).str.replace('k', '', regex=False).str.replace(',', '', regex=True)
    df['Size'] = df['Size'].replace('Varies with device', np.nan)
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    
    # 6. REMOÇÃO DE VALORES NULOS ESSENCIAIS
    # Remove linhas onde os valores essenciais (incluindo os NaN gerados pelas conversões de strings inválidas acima) são nulos.
    df.dropna(subset=['Rating', 'Installs', 'Reviews', 'Price', 'Category', 'Type'], inplace=True)
    
    # Cria uma métrica de popularidade: Engajamento/Instalações
    df['Popularity_Score'] = df['Reviews'] / df['Installs']
    
    return df

# Carrega os dados
df = load_data()


# ------------------------------------
# 2. SIDEBAR E FILTROS
# ------------------------------------
st.sidebar.header("Filtros de Análise")

# Garante que os dados foram carregados antes de aplicar filtros
if df.empty:
    st.error("O aplicativo foi interrompido devido a um erro na autenticação ou carregamento de dados.")
    st.stop()

# Filtro 1: Categoria
all_categories = ['Todas'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox('Categoria', all_categories)

# Filtro 2: Tipo (Free/Paid)
all_types = ['Ambos'] + df['Type'].unique().tolist()
selected_type = st.sidebar.selectbox('Tipo de Aplicativo', all_types)

# Aplica os filtros
df_filtered = df.copy()

if selected_category != 'Todas':
    df_filtered = df_filtered[df_filtered['Category'] == selected_category]

if selected_type != 'Ambos':
    df_filtered = df_filtered[df_filtered['Type'] == selected_type]

# Garante que o DataFrame não está vazio após os filtros
if df_filtered.empty:
    st.error("Nenhum dado corresponde aos filtros selecionados. Tente ajustar os filtros.")
    st.stop()


# ------------------------------------
# 3. CABEÇALHO PRINCIPAL
# ------------------------------------
st.title("🤖 Google Play Store: Desvendando o Sucesso")
st.markdown("Análise de Métricas, Engajamento e Receita dos Aplicativos (Dataset 2018/2019)")


# ------------------------------------
# 4. KPIs DINÂMICOS (Linha superior)
# ------------------------------------
st.header("Métricas Chave")
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Aplicativos Analisados
col1.metric("Aplicativos Analisados", f"{df_filtered.shape[0]:,}", help="Número de linhas após a limpeza e filtros.")

# KPI 2: Média de Avaliação (Rating)
avg_rating = df_filtered['Rating'].mean()
col2.metric("Avaliação Média", f"{avg_rating:.2f} / 5.0", help="Nota média dos aplicativos no filtro.")

# KPI 3: Aplicativo de Maior Sucesso
# Garante que o KPI não quebre em filtros vazios
if not df_filtered.empty and 'Popularity_Score' in df_filtered and df_filtered['Popularity_Score'].max() >= 0:
    app_mais_popular = df_filtered.loc[df_filtered['Popularity_Score'].idxmax()]
    col3.metric("App Mais Popular (Score)", f"{app_mais_popular['Popularity_Score']:.4f}", help=f"Baseado na relação Reviews/Installs. App: {app_mais_popular['App']}")
else:
    col3.metric("App Mais Popular (Score)", "N/A", help="Dados insuficientes para calcular o score de popularidade.")


# KPI 4: Aplicativo mais caro
if not df_filtered.empty and 'Price' in df_filtered and df_filtered['Price'].max() > 0:
    app_mais_caro = df_filtered.loc[df_filtered['Price'].idxmax()]
    col4.metric("Preço Máximo", f"${app_mais_caro['Price']:.2f}", help=f"App: {app_mais_caro['App']}")
else:
    col4.metric("Preço Máximo", "$0.00", help="Nenhum aplicativo pago no filtro.")

st.markdown("---")


# ------------------------------------
# 5. ANÁLISE CENTRAL (Gráficos Criativos)
# ------------------------------------
st.header("Visuais Criativos")
col_chart_1, col_chart_2 = st.columns([2, 1])

# GRÁFICO 1: Instalações vs. Categoria (Plotly com Barras Ordenadas)
with col_chart_1:
    st.subheader("Instalações Totais por Categoria (Top 15)")
    
    # Agrupa e soma, e reseta o índice para usar no Plotly
    df_cat_installs = df_filtered.groupby('Category')['Installs'].sum().nlargest(15).reset_index()
    
    fig_installs = px.bar(
        df_cat_installs, 
        x='Category', 
        y='Installs', 
        color='Installs',
        color_continuous_scale=px.colors.sequential.Plasma,
        title='Volume de Instalações (Escala Logarítmica)',
        log_y=True # Uso de escala logarítmica para visualização de grandes variações
    )
    fig_installs.update_layout(xaxis_title="", yaxis_title="Instalações (Escala Log)")
    st.plotly_chart(fig_installs, use_container_width=True)

# GRÁFICO 2: Distribuição de Preço (Histograma/Violin Plot)
with col_chart_2:
    st.subheader("Distribuição de Preço")
    
    # Filtra apenas apps pagos
    df_paid = df_filtered[df_filtered['Type'] == 'Paid']
    
    # Condição para desenhar o gráfico apenas se houver dados pagos válidos
    if not df_paid.empty and 'Price' in df_paid and df_paid['Price'].max() > 0:
        # Usa um histograma interativo para mostrar a concentração de preços
        fig_price_hist = px.histogram(
            df_paid, 
            x='Price', 
            nbins=30, 
            title=f'Preços em {selected_category or "Todos"} (Apenas Pagos)',
            labels={'Price': 'Preço ($)'}
        )
        # Limita o eixo X para melhor visualização (preços mais altos distorcem)
        fig_price_hist.update_xaxes(range=[0, df_paid['Price'].quantile(0.95)])
        st.plotly_chart(fig_price_hist, use_container_width=True)
    else:
        st.info("Nenhum aplicativo pago encontrado no filtro atual para o gráfico de preços.")

st.markdown("---")

# GRÁFICO 3: Gráfico de Dispersão (Dispersão de Desempenho)
st.header("Dispersão de Desempenho (Avaliação vs. Revisões)")

fig_scatter = px.scatter(
    df_filtered,
    x='Reviews',
    y='Rating',
    size='Installs', # O tamanho da bolha representa as instalações
    color='Category',
    hover_name='App',
    log_x=True, # Log para Reviews para visualizar melhor
    title="Avaliação (Rating) vs. Volume de Revisões (Reviews)",
    labels={'Reviews': 'Revisões (Log Scale)', 'Rating': 'Avaliação'}
)
fig_scatter.update_layout(showlegend=True)
st.plotly_chart(fig_scatter, use_container_width=True)

# NOVO GRÁFICO: Média de Instalações por Categoria (Gráfico de Rosca/Donut)
st.header("Distribuição Média de Instalações por Categoria")
col_ranking, col_donut = st.columns(2)

with col_donut:
    df_avg_installs = df_filtered.groupby('Category')['Installs'].mean().nlargest(10).reset_index()
    fig_donut = px.pie(
        df_avg_installs, 
        values='Installs', 
        names='Category', 
        title='Top 10 Categorias por Média de Instalações',
        hole=.4
    )
    fig_donut.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_donut, use_container_width=True)

with col_ranking:
    st.subheader("Top 10 Aplicativos por Installs")
    df_top_apps = df_filtered.nlargest(10, 'Installs')[['App', 'Installs', 'Category']]
    # Garante que 'Installs' é um float para a formatação
    df_top_apps['Installs Formatado'] = df_top_apps['Installs'].apply(lambda x: f'{x:,.0f}')
    st.dataframe(df_top_apps[['App', 'Category', 'Installs Formatado']], use_container_width=True, hide_index=True)


# ------------------------------------
# 6. EXPORTAR DEPENDÊNCIAS (para Deploy)
# ------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Para o deploy no Streamlit Cloud, você precisa dos arquivos `app.py` e `requirements.txt` no seu repositório.")
