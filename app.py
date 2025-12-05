import streamlit as st
import pandas as pd
import kagglehub
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # Adicionando a importação do numpy

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
    """Baixa, carrega e pré-processa o dataset do Google Play Store do Kaggle."""
    try:
        st.write("Baixando dataset do Kaggle Hub...")
        
        # Baixa o dataset
        dataset_path = kagglehub.dataset_download("lava18/google-play-store-apps")
        file_path = f"{dataset_path}/googleplaystore.csv"
        
        # Carrega o CSV
        df = pd.read_csv(file_path)
        
    except Exception as e:
        st.error(f"Erro ao carregar os dados do Kaggle Hub: {e}")
        return pd.DataFrame()

    # Limpeza e Pré-processamento
    
    # Remove a linha com dados incorretos ou outliers conhecidos
    df.drop(df[df['App'] == 'Life Made Better'].index, inplace=True)
    
    # Limpa e converte 'Installs' para numérico
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '').str.replace(',', '', regex=True).astype(float)
    
    # Converte 'Reviews' para numérico
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Limpa e converte 'Price' para numérico
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=True).astype(float)
    
    # Remove valores nulos essenciais
    df.dropna(subset=['Rating', 'Installs', 'Reviews', 'Price'], inplace=True)
    
    # Cria uma métrica de popularidade: Engajamento/Instalações
    df['Popularity_Score'] = df['Reviews'] / df['Installs']
    
    return df

# Carrega os dados
df = load_data()


# ------------------------------------
# 2. SIDEBAR E FILTROS
# ------------------------------------
st.sidebar.header("Filtros de Análise")

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

# Garante que o DataFrame não está vazio
if df_filtered.empty:
    st.error("Nenhum dado corresponde aos filtros selecionados.")
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
# Usa a métrica Popularity_Score (Reviews/Installs) para evitar vieses de apps muito antigos
app_mais_popular = df_filtered.loc[df_filtered['Popularity_Score'].idxmax()]
col3.metric("App Mais Popular (Score)", f"{app_mais_popular['Popularity_Score']:.4f}", help=f"Baseado na relação Reviews/Installs. App: {app_mais_popular['App']}")

# KPI 4: Aplicativo mais caro
app_mais_caro = df_filtered.loc[df_filtered['Price'].idxmax()]
col4.metric("Preço Máximo", f"${app_mais_caro['Price']:.2f}", help=f"App: {app_mais_caro['App']}")

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
    
    if not df_paid.empty:
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

# Usa o `Size` do App no eixo Z (tamanho da bolha)
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
    df_top_apps['Installs Formatado'] = df_top_apps['Installs'].apply(lambda x: f'{x:,.0f}')
    st.dataframe(df_top_apps[['App', 'Category', 'Installs Formatado']], use_container_width=True, hide_index=True)


# ------------------------------------
# 6. EXPORTAR DEPENDÊNCIAS (para Deploy)
# ------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Para o deploy no Streamlit Cloud, você precisa dos arquivos `app.py` e `requirements.txt` no seu repositório.")