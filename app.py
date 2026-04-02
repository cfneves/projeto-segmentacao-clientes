# --- IMPORTS ---
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="TCC Segmentação de Clientes",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ TCC: Segmentação de Clientes de Supermercado")
st.write(
    """
Este App apresenta os resultados do TCC, aplicando **K-Means** para segmentar clientes
com base em **Renda Anual** e **Pontuação de Gastos** (*Spending Score*).
"""
)

# =========================
# CONSTANTES
# =========================
DATA_LINK = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
)

LOCAL_DATA_PATH = Path("data") / "Mall_Customers.csv"

COLS_PADRAO = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]

LABELS_TRADUZIDOS = {
    "Annual Income (k$)": "Renda Anual (k$)",
    "Spending Score (1-100)": "Pontuação de Gastos (1-100)",
    "Age": "Idade",
    "Genre": "Gênero",
}

# Cores por PERSONA (mantém consistência com sidebar e gráficos)
MAPA_CORES_PERSONA = {
    "Econômicos Ricos": "blue",
    "Cautelosos": "red",
    "Clientes Alvo (VIPs)": "green",
    "Gastadores (Impulsivos)": "orange",
    "Clientes Padrão": "purple",
}

# Blocos de estilo do Streamlit para sidebar
TIPO_CAIXA_PERSONA = {
    "Econômicos Ricos": "info",
    "Cautelosos": "error",
    "Clientes Alvo (VIPs)": "success",
    "Gastadores (Impulsivos)": "warning",
    "Clientes Padrão": "info",
}


# =========================
# FUNÇÕES
# =========================
@st.cache_data
def carregar_dados() -> pd.DataFrame | None:
    """
    Carrega os dados do arquivo local (se existir) ou da internet.
    Retorna DataFrame com colunas padronizadas.
    """
    try:
        if LOCAL_DATA_PATH.exists():
            df = pd.read_csv(LOCAL_DATA_PATH)
        else:
            df = pd.read_csv(DATA_LINK)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

    df.columns = COLS_PADRAO
    return df


def classificar_persona(income: float, spend: float, income_median: float, spend_median: float) -> str:
    """
    Classifica a persona com base em renda e gasto (alto/baixo) e casos intermediários.
    A lógica é simples e interpretável (boa para banca).
    """
    renda_alta = income >= income_median
    gasto_alto = spend >= spend_median

    if renda_alta and (not gasto_alto):
        return "Econômicos Ricos"  # alta renda, baixo gasto
    if (not renda_alta) and (not gasto_alto):
        return "Cautelosos"  # baixa renda, baixo gasto
    if renda_alta and gasto_alto:
        return "Clientes Alvo (VIPs)"  # alta renda, alto gasto
    if (not renda_alta) and gasto_alto:
        return "Gastadores (Impulsivos)"  # baixa renda, alto gasto

    # fallback (caso raro por mediana)
    return "Clientes Padrão"


def descrever_niveis(valor: float, p33: float, p66: float, rotulo: str) -> str:
    """
    Converte um valor em rótulo qualitativo (Baixo/Médio/Alto) com base em tercis.
    """
    if valor <= p33:
        return f"{rotulo} Baixo"
    if valor <= p66:
        return f"{rotulo} Médio"
    return f"{rotulo} Alto"


@st.cache_data
def modelar_kmeans(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """
    Aplica scaler + KMeans e devolve:
    - df_clusterizado (com Cluster, Persona)
    - centers_df (centroides na escala original)
    - silhouette (escala padronizada)
    """
    features = df[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(X)

    df_out = df.copy()
    df_out["Cluster"] = labels

    # Centroides na escala original
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=FEATURE_COLS)
    centers_df["Cluster"] = np.arange(k)

    # Persona baseada em renda/gasto (com medianas do dataset)
    income_median = df_out[FEATURE_COLS[0]].median()
    spend_median = df_out[FEATURE_COLS[1]].median()

    centers_df["Persona"] = centers_df.apply(
        lambda r: classificar_persona(r[FEATURE_COLS[0]], r[FEATURE_COLS[1]], income_median, spend_median),
        axis=1,
    )

    # Atribuir persona a cada ponto conforme cluster
    map_cluster_persona = dict(zip(centers_df["Cluster"], centers_df["Persona"]))
    df_out["Persona"] = df_out["Cluster"].map(map_cluster_persona)

    # Silhouette Score (somente se fizer sentido)
    sil = None
    if k >= 2 and k < len(df_out):
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    return df_out, centers_df, sil


@st.cache_data
def calcular_elbow(df: pd.DataFrame, k_max: int = 10) -> pd.DataFrame:
    """
    Calcula WCSS (inércia) para K=1..k_max (método do cotovelo).
    """
    features = df[FEATURE_COLS].copy()
    X = StandardScaler().fit_transform(features)

    wcss = []
    K_range = range(1, k_max + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    return pd.DataFrame({"Número de Clusters (K)": list(K_range), "WCSS (Inércia)": wcss})


def caixa_sidebar_por_persona(persona: str, texto: str) -> None:
    """
    Renderiza a caixa adequada (info/success/warning/error) conforme a persona.
    """
    tipo = TIPO_CAIXA_PERSONA.get(persona, "info")
    if tipo == "success":
        st.sidebar.success(texto)
    elif tipo == "warning":
        st.sidebar.warning(texto)
    elif tipo == "error":
        st.sidebar.error(texto)
    else:
        st.sidebar.info(texto)


# =========================
# CONTROLES (SIDEBAR)
# =========================
st.sidebar.header("Configurações do Modelo")

k_escolhido = st.sidebar.slider(
    "Número de clusters (K)",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.header("Sobre o Projeto")
st.sidebar.markdown(
    """
**Autor:** Cláudio Ferreira Neves  
**Cargo:** Especialista em Dados II  
**Curso:** Pós-Graduação em Ciência de Dados e Inteligência Artificial (UniSENAI)  
**Professor:** Willian Daniel de Mattos  
"""
)


# =========================
# EXECUÇÃO
# =========================
with st.spinner("Carregando dados e treinando o modelo..."):
    df = carregar_dados()

if df is None:
    st.stop()

with st.spinner("Modelando clusters e calculando métricas..."):
    df_clusterizado, centers_df, sil = modelar_kmeans(df, k_escolhido)
    df_elbow = calcular_elbow(df, k_max=10)

# Métricas resumidas no topo
colA, colB, colC = st.columns(3)
colA.metric("Total de clientes", f"{len(df_clusterizado):,}".replace(",", "."))
colB.metric("K (clusters)", str(k_escolhido))
colC.metric("Silhouette Score", "-" if sil is None else f"{sil:.3f}")

# =========================
# ABAS (TABS)
# =========================
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Segmentação de Clientes (Gráfico 2D)",
        "📊 Perfis Detalhados (Boxplots)",
        "📈 Método do Cotovelo (Elbow)",
    ]
)

# --- ABA 1: Cluster 2D ---
with tab1:
    st.header(f"Segmentação de Clientes (K={k_escolhido})")
    st.markdown(
        """
O gráfico apresenta os grupos (clusters) identificados pelo K-Means.
As cores representam **personas de negócio** inferidas a partir dos centroides (renda e gasto),
o que mantém consistência mesmo se a numeração do cluster variar.
"""
    )

    fig_cluster = px.scatter(
        data_frame=df_clusterizado,
        x=FEATURE_COLS[0],
        y=FEATURE_COLS[1],
        color="Persona",
        title="Segmentação de Clientes (Gráfico Interativo)",
        hover_data=["Age", "Genre"],
        labels=LABELS_TRADUZIDOS,
        color_discrete_map=MAPA_CORES_PERSONA,
    )

    # Centroides (pontos X)
    fig_cluster.add_scatter(
        x=centers_df[FEATURE_COLS[0]],
        y=centers_df[FEATURE_COLS[1]],
        mode="markers",
        marker_symbol="x",
        marker_color="black",
        marker_size=12,
        name="Centroides",
        hovertext=centers_df["Persona"],
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

# --- ABA 2: Perfis Detalhados ---
with tab2:
    st.header("Análise Detalhada dos Perfis")
    st.write(
        "Nesta seção, analisamos características por persona (Idade, Renda, Gasto e Gênero), "
        "apoiando a interpretação e a construção de recomendações de negócio."
    )

    # Boxplots por persona
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Renda e Idade")

        fig_renda = px.box(
            df_clusterizado,
            x="Persona",
            y=FEATURE_COLS[0],
            color="Persona",
            title="Distribuição da Renda Anual por Persona",
            labels=LABELS_TRADUZIDOS,
            color_discrete_map=MAPA_CORES_PERSONA,
        )
        st.plotly_chart(fig_renda, use_container_width=True)

        fig_idade = px.box(
            df_clusterizado,
            x="Persona",
            y="Age",
            color="Persona",
            title="Distribuição da Idade por Persona",
            labels=LABELS_TRADUZIDOS,
            color_discrete_map=MAPA_CORES_PERSONA,
        )
        st.plotly_chart(fig_idade, use_container_width=True)

    with col2:
        st.subheader("Gasto e Gênero")

        fig_gasto = px.box(
            df_clusterizado,
            x="Persona",
            y=FEATURE_COLS[1],
            color="Persona",
            title="Distribuição da Pontuação de Gastos por Persona",
            labels=LABELS_TRADUZIDOS,
            color_discrete_map=MAPA_CORES_PERSONA,
        )
        st.plotly_chart(fig_gasto, use_container_width=True)

        df_counts = (
            df_clusterizado.groupby(["Persona", "Genre"]).size().reset_index(name="Contagem")
        )
        fig_genero = px.bar(
            df_counts,
            x="Persona",
            y="Contagem",
            color="Genre",
            title="Distribuição de Gênero por Persona",
            labels=LABELS_TRADUZIDOS,
            barmode="group",
        )
        st.plotly_chart(fig_genero, use_container_width=True)

# --- ABA 3: Elbow ---
with tab3:
    st.header("Método do Cotovelo (Justificativa do K)")
    st.info(
        """
**Como ler o gráfico:** A linha mostra a Inércia (WCSS). Procuramos o “cotovelo”, isto é,
o ponto em que a redução da inércia passa a ser pequena ao aumentar K.
"""
    )

    fig_elbow = px.line(
        df_elbow,
        x="Número de Clusters (K)",
        y="WCSS (Inércia)",
        title="Método do Cotovelo (Interativo)",
        markers=True,
    )
    fig_elbow.update_xaxes(dtick=1)
    st.plotly_chart(fig_elbow, use_container_width=True)

# =========================
# SIDEBAR: PERSONAS (estável e explicável)
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Conclusões (Personas)")

# Estatísticas por persona para descrever automaticamente
income_p33, income_p66 = np.percentile(df_clusterizado[FEATURE_COLS[0]], [33, 66])
spend_p33, spend_p66 = np.percentile(df_clusterizado[FEATURE_COLS[1]], [33, 66])

resumo = (
    df_clusterizado.groupby("Persona")
    .agg(
        idade_media=("Age", "mean"),
        renda_media=(FEATURE_COLS[0], "mean"),
        gasto_medio=(FEATURE_COLS[1], "mean"),
        total=("Persona", "size"),
    )
    .reset_index()
    .sort_values("total", ascending=False)
)

# Ações recomendadas fixas (negócio)
acoes = {
    "Econômicos Ricos": "Ativar com produtos premium, cross-sell e experiências exclusivas.",
    "Cautelosos": "Focar em promoções de itens essenciais e comunicação de economia.",
    "Clientes Alvo (VIPs)": "Priorizar retenção, fidelidade e benefícios personalizados.",
    "Gastadores (Impulsivos)": "Marketing de tendências e campanhas rápidas (alto apelo).",
    "Clientes Padrão": "Nutrir com ofertas para elevar frequência e ticket médio.",
}

for _, row in resumo.iterrows():
    persona = row["Persona"]
    idade = row["idade_media"]
    renda = row["renda_media"]
    gasto = row["gasto_medio"]
    total = int(row["total"])

    renda_nivel = descrever_niveis(renda, income_p33, income_p66, "Renda")
    gasto_nivel = descrever_niveis(gasto, spend_p33, spend_p66, "Gasto")

    texto = (
        f"**{persona}**\n"
        f"* **Tamanho do grupo:** {total}\n"
        f"* **Perfil:** {renda_nivel}, {gasto_nivel}, Idade média ~{idade:.0f}\n"
        f"* **Ação recomendada:** {acoes.get(persona, 'Ações segmentadas conforme perfil.')}"
    )
    caixa_sidebar_por_persona(persona, texto)
