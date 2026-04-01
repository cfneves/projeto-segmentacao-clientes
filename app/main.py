"""
app/main.py
Aplicação Streamlit — Segmentação de Clientes com K-Means.

Arquitetura: camada EXCLUSIVAMENTE de apresentação.
Nenhum import de sklearn aqui — toda lógica de ML em src/pipeline.py.

Execução:
    streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Adiciona a raiz do projeto ao sys.path ────────────────────────────────────
# Necessário para imports de src/ quando chamado com "streamlit run app/main.py"
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.evaluation import (
    calinski_harabasz_label,
    davies_bouldin_label,
    silhouette_label,
)
from src.pipeline import CustomerSegmentationPipeline, PipelineConfig, PipelineResults
from src.preprocessing import (
    get_numeric_columns,
    load_default_data,
    load_uploaded_data,
    validate_features,
)
from src.utils import (
    PERSONA_ACTIONS,
    PERSONA_BOX_TYPE,
    PERSONA_COLORS,
    build_resumo_table,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Segmentação de Clientes",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — tema escuro corporativo ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div:not([data-testid]) { color: #eaeaea !important; }

    /* KPI Cards */
    .kpi-box {
        background: #16213e;
        border-left: 4px solid #e94560;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .kpi-box.green  { border-left-color: #4ecca3; }
    .kpi-box.blue   { border-left-color: #4895ef; }
    .kpi-box.orange { border-left-color: #f7b731; }
    .kpi-label { font-size: 11px; color: #8888aa; letter-spacing: 1px; text-transform: uppercase; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #ffffff; line-height: 1.3; }
    .kpi-sub   { font-size: 11px; color: #4ecca3; margin-top: 2px; }

    /* Section labels */
    .section-label {
        font-size: 11px; font-weight: 700;
        color: #4ecca3; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 6px;
    }

    /* Metric badge */
    .metric-badge {
        display: inline-block;
        background: #0f3460;
        border: 1px solid #4ecca3;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 12px;
        color: #eaeaea;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS: dict = {
    "df": None,              # DataFrame carregado
    "results": None,         # PipelineResults (único objeto de resultado)
    "features": [],          # Features selecionadas
    "k": 5,                  # K atual
    "model_ready": False,    # Flag: pipeline executou com sucesso
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE UI
# ══════════════════════════════════════════════════════════════════════════════

def kpi(label: str, value: str, sub: str, col, accent: str = "") -> None:
    """Renderiza card KPI estilizado em uma coluna Streamlit."""
    accent_class = f" {accent}" if accent else ""
    col.markdown(
        f'<div class="kpi-box{accent_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _dark(fig: go.Figure) -> go.Figure:
    """Aplica tema escuro padrão a figuras Plotly."""
    fig.update_layout(
        plot_bgcolor="#0f3460",
        paper_bgcolor="#16213e",
        font_color="#eaeaea",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=20),
    )
    return fig


def _persona_box(persona: str, msg: str) -> None:
    """Renderiza caixa Streamlit (success/warning/error/info) pela persona."""
    box_type = PERSONA_BOX_TYPE.get(persona, "info")
    getattr(st, box_type)(msg)


def _results() -> Optional[PipelineResults]:
    """Retorna PipelineResults do session_state ou None."""
    return st.session_state.get("results")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛍️ Segmentação\n### de Clientes")
    st.caption("TCC UniSENAI — Ciência de Dados e IA")
    st.markdown("---")

    # ── Navegação ──────────────────────────────────────────
    nav = st.radio(
        "Navegação",
        ["🏠 Início", "📊 Exploração", "🎯 Modelagem", "👥 Perfis"],
        key="nav",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">📂 Fonte de Dados</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "",
        ["Dataset padrão (Mall Customers)", "Upload CSV"],
        key="data_source",
        label_visibility="collapsed",
    )

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Arquivo CSV", type="csv", label_visibility="collapsed")
        if uploaded:
            df_new = load_uploaded_data(uploaded)
            if df_new is not None:
                st.session_state.df = df_new
                st.session_state.model_ready = False
                st.session_state.results = None
                st.success(f"✅ {len(df_new):,} registros")
    else:
        if st.button("Carregar Dataset Padrão", use_container_width=True):
            with st.spinner("Buscando dados..."):
                df_new = load_default_data()
            if df_new is not None:
                st.session_state.df = df_new
                st.session_state.model_ready = False
                st.session_state.results = None
                st.success(f"✅ {len(df_new):,} registros")

    # ── Parâmetros do modelo (aparece só com dados carregados) ──
    if st.session_state.df is not None:
        _df_side = st.session_state.df
        num_cols = get_numeric_columns(_df_side)

        st.markdown("---")
        st.markdown('<div class="section-label">⚙️ Parâmetros</div>', unsafe_allow_html=True)

        _default_feats = (
            ["Annual Income (k$)", "Spending Score (1-100)"]
            if all(c in num_cols for c in ["Annual Income (k$)", "Spending Score (1-100)"])
            else num_cols[:2]
        )

        selected_features = st.multiselect(
            "Features para clustering",
            options=num_cols,
            default=_default_feats,
            key="feature_select",
        )
        st.session_state.features = selected_features

        k_val = st.slider(
            "Número de clusters (K)",
            min_value=2,
            max_value=10,
            value=st.session_state.k,
            key="k_slider",
        )
        st.session_state.k = k_val

        st.markdown("---")
        run_btn = st.button("🚀 Treinar Modelo", use_container_width=True, type="primary")

        if run_btn:
            if len(selected_features) < 2:
                st.warning("Selecione pelo menos 2 features.")
            elif not validate_features(_df_side, selected_features):
                pass  # mensagem exibida por validate_features
            else:
                with st.spinner("Executando pipeline de ML..."):
                    try:
                        pipeline = CustomerSegmentationPipeline(
                            PipelineConfig(k=k_val)
                        )
                        results = pipeline.run(_df_side, selected_features)
                        st.session_state.results = results
                        st.session_state.model_ready = True
                        st.success("✅ Modelo treinado!")
                    except Exception as exc:
                        st.error(f"Erro no pipeline: {exc}")
                        st.session_state.model_ready = False

    # ── Rodapé do sidebar ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <small>
        <b>Autor:</b> Cláudio F. Neves<br>
        <b>Orientador:</b> Prof. Willian D. de Mattos<br>
        <b>Curso:</b> Pós-Grad. Ciência de Dados e IA<br>
        <b>UniSENAI</b> | 2025
        </small>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ══════════════════════════════════════════════════════════════════════════════
page: str = st.session_state.nav


# ────────────────────────────────────────────────────────────────────────
# 🏠  INÍCIO
# ────────────────────────────────────────────────────────────────────────
if page == "🏠 Início":
    st.title("🛍️ Segmentação de Clientes com K-Means")
    st.markdown(
        "Aplicação interativa de **aprendizado não supervisionado** para identificar "
        "grupos naturais de clientes e gerar estratégias de marketing personalizadas."
    )

    r = _results()
    df_home = st.session_state.df
    m = r.metrics if r else {}

    # ── KPIs ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpi("Total de Clientes",    f"{len(df_home):,}" if df_home is not None else "—",
        "registros carregados", c1)
    kpi("Clusters (K)",         str(st.session_state.k), "valor selecionado", c2, "green")
    kpi("Silhouette Score",
        silhouette_label(m.get("silhouette")) if r else "—",
        "[-1, 1] maior = melhor", c3, "blue")
    kpi("Davies-Bouldin",
        davies_bouldin_label(m.get("davies_bouldin")) if r else "—",
        "[0, ∞) menor = melhor", c4, "orange")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📌 Como usar")
        st.markdown(
            """
            1. **Carregue os dados** na barra lateral (dataset padrão ou CSV próprio)
            2. **Selecione as features** numéricas para o clustering
            3. **Ajuste o K** usando o gráfico do cotovelo como guia
            4. Clique em **🚀 Treinar Modelo**
            5. Explore os resultados em **Modelagem** e **Perfis**
            """
        )

    with col_b:
        st.subheader("📐 Sobre o Algoritmo")
        st.markdown(
            """
            **K-Means** agrupa clientes minimizando a variância intra-cluster (WCSS).

            - Inicialização `k-means++` — evita mínimos locais
            - `StandardScaler` — normaliza escala antes de calcular distâncias
            - `random_state=42` — resultado 100% reproduzível
            - **3 métricas complementares** — Silhouette · Davies-Bouldin · Calinski-Harabasz
            """
        )

    st.markdown("---")
    if st.session_state.model_ready:
        st.success(
            "✅ Modelo treinado! Acesse **Modelagem** para ver os clusters "
            "ou **Perfis** para as recomendações de negócio."
        )
    elif df_home is not None:
        st.info("ℹ️ Dados carregados. Configure os parâmetros e clique em **🚀 Treinar Modelo**.")
    else:
        st.info("ℹ️ Comece carregando um dataset na barra lateral.")


# ────────────────────────────────────────────────────────────────────────
# 📊  EXPLORAÇÃO
# ────────────────────────────────────────────────────────────────────────
elif page == "📊 Exploração":
    st.title("📊 Exploração dos Dados")

    df_exp = st.session_state.df
    if df_exp is None:
        st.warning("⚠️ Carregue um dataset na barra lateral antes de explorar.")
        st.stop()

    # KPIs de qualidade do dataset
    with st.container():
        ce1, ce2, ce3, ce4 = st.columns(4)
        ce1.metric("Registros", f"{len(df_exp):,}")
        ce2.metric("Colunas", str(df_exp.shape[1]))
        ce3.metric("Colunas numéricas", str(len(get_numeric_columns(df_exp))))
        ce4.metric("Valores nulos", str(int(df_exp.isnull().sum().sum())))

    st.markdown("---")

    tab_data, tab_dist, tab_corr = st.tabs(
        ["📋 Dados Brutos", "📈 Distribuições", "🔗 Correlação"]
    )

    # ── Dados Brutos ───────────────────────────────────────
    with tab_data:
        n_rows = st.slider("Linhas a exibir", 10, min(500, len(df_exp)), 50)
        st.dataframe(df_exp.head(n_rows), use_container_width=True, height=420)
        st.caption(f"Exibindo {n_rows} de {len(df_exp):,} registros")

        with st.expander("📊 Estatísticas Descritivas"):
            st.dataframe(df_exp.describe().T.round(3), use_container_width=True)

    # ── Distribuições ─────────────────────────────────────
    with tab_dist:
        num_cols_exp = get_numeric_columns(df_exp)
        col_hist = st.selectbox("Variável:", num_cols_exp, key="hist_col")

        fig_hist = px.histogram(
            df_exp, x=col_hist, nbins=30, marginal="box",
            title=f"Distribuição — {col_hist}",
            color_discrete_sequence=["#e94560"],
        )
        st.plotly_chart(_dark(fig_hist), use_container_width=True)

        if len(num_cols_exp) >= 2:
            st.markdown("#### Relações entre Variáveis (Pairplot)")
            pair_cols = st.multiselect(
                "Selecione as variáveis:",
                num_cols_exp,
                default=num_cols_exp[:min(4, len(num_cols_exp))],
                key="pair_cols",
            )
            if len(pair_cols) >= 2:
                fig_pair = px.scatter_matrix(
                    df_exp,
                    dimensions=pair_cols,
                    title="Pairplot",
                    color_discrete_sequence=["#4ecca3"],
                    opacity=0.55,
                )
                fig_pair.update_traces(diagonal_visible=True, showupperhalf=False)
                st.plotly_chart(_dark(fig_pair), use_container_width=True)

    # ── Correlação ────────────────────────────────────────
    with tab_corr:
        num_cols_corr = get_numeric_columns(df_exp)
        if len(num_cols_corr) < 2:
            st.info("São necessárias pelo menos 2 colunas numéricas.")
        else:
            corr = df_exp[num_cols_corr].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Mapa de Correlação de Pearson",
                zmin=-1, zmax=1,
            )
            st.plotly_chart(_dark(fig_corr), use_container_width=True)
            st.caption(
                "Correlação próxima de +1 = relação positiva forte. "
                "-1 = negativa forte. 0 = sem correlação linear."
            )


# ────────────────────────────────────────────────────────────────────────
# 🎯  MODELAGEM
# ────────────────────────────────────────────────────────────────────────
elif page == "🎯 Modelagem":
    st.title("🎯 Modelagem — K-Means Clustering")

    if st.session_state.df is None:
        st.warning("⚠️ Carregue um dataset na barra lateral.")
        st.stop()

    tab_elbow, tab_scatter, tab_metricas = st.tabs(
        ["📈 Método do Cotovelo", "🔵 Clusters (Scatter)", "📊 Métricas de Avaliação"]
    )

    # ── Método do Cotovelo ────────────────────────────────
    with tab_elbow:
        st.subheader("Método do Cotovelo — Escolha do K Ideal")
        st.info(
            "**Como ler:** Identifique o ponto em que a queda da Inércia (WCSS) "
            "passa de acentuada para suave — esse é o **K ideal**. "
            "Adicionar mais clusters além desse ponto traz ganho marginal pequeno."
        )

        r = _results()
        feats = st.session_state.features

        if not feats or len(feats) < 2:
            st.warning("Selecione pelo menos 2 features na barra lateral.")
        else:
            # Usa o elbow do resultado do pipeline se disponível
            elbow_df = r.elbow_df if r else None

            if elbow_df is None:
                st.info("Treine o modelo para gerar o gráfico do cotovelo.")
            else:
                k_atual = st.session_state.k
                fig_elbow = px.line(
                    elbow_df, x="K", y="WCSS",
                    markers=True,
                    title="Curva do Cotovelo — Inércia (WCSS) × Número de Clusters",
                    labels={"K": "Número de Clusters (K)", "WCSS": "Inércia (WCSS)"},
                    color_discrete_sequence=["#e94560"],
                )
                fig_elbow.add_vline(
                    x=k_atual,
                    line_dash="dash",
                    line_color="#4ecca3",
                    annotation_text=f"  K = {k_atual}",
                    annotation_font_color="#4ecca3",
                    annotation_position="top right",
                )
                fig_elbow.update_xaxes(dtick=1)
                st.plotly_chart(_dark(fig_elbow), use_container_width=True)

    # ── Scatter dos Clusters ──────────────────────────────
    with tab_scatter:
        st.subheader("Visualização dos Clusters no Espaço de Features")

        r = _results()
        if not st.session_state.model_ready or r is None:
            st.warning("⚠️ Treine o modelo primeiro usando a barra lateral.")
        else:
            df_c = r.df_clustered
            centers_df = r.centers_df
            feats = r.feature_cols

            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("K (clusters)", str(r.k))
            ms2.metric("Silhouette", silhouette_label(r.metrics.get("silhouette")))
            ms3.metric("Total de pontos", f"{len(df_c):,}")

            st.markdown("---")

            sc1, sc2 = st.columns(2)
            x_col = sc1.selectbox("Eixo X:", feats, index=0, key="sx")
            y_col = sc2.selectbox(
                "Eixo Y:", feats, index=min(1, len(feats) - 1), key="sy"
            )

            color_col = "Persona" if "Persona" in df_c.columns else "Cluster"
            color_map = PERSONA_COLORS if color_col == "Persona" else None
            hover_cols = [c for c in ["Age", "Genre", "CustomerID"] if c in df_c.columns]

            fig_sc = px.scatter(
                df_c,
                x=x_col, y=y_col,
                color=color_col,
                title=f"Segmentação de Clientes — K = {r.k}",
                hover_data=hover_cols,
                color_discrete_map=color_map,
                opacity=0.78,
            )

            # Centroides marcados com X
            fig_sc.add_trace(
                go.Scatter(
                    x=centers_df[x_col].tolist(),
                    y=centers_df[y_col].tolist(),
                    mode="markers",
                    marker=dict(
                        symbol="x", color="white", size=15,
                        line=dict(width=2.5, color="#000000"),
                    ),
                    name="Centroides",
                    hovertext=(
                        centers_df["Persona"].tolist()
                        if "Persona" in centers_df.columns
                        else centers_df["Cluster"].astype(str).tolist()
                    ),
                    hoverinfo="text",
                )
            )
            st.plotly_chart(_dark(fig_sc), use_container_width=True)

    # ── Métricas de Avaliação ─────────────────────────────
    with tab_metricas:
        st.subheader("Avaliação Quantitativa do Clustering")

        r = _results()
        if not st.session_state.model_ready or r is None:
            st.warning("⚠️ Treine o modelo primeiro.")
        else:
            m = r.metrics

            # Cards de métricas
            cm1, cm2, cm3 = st.columns(3)

            with cm1:
                st.markdown("#### Silhouette Score")
                st.markdown(f"**Valor:** `{silhouette_label(m.get('silhouette'))}`")
                st.caption(
                    "Mede o quão bem cada ponto pertence ao seu cluster "
                    "em comparação com o cluster vizinho. "
                    "**Escala: [-1, 1] — maior é melhor.**"
                )

            with cm2:
                st.markdown("#### Davies-Bouldin Index")
                st.markdown(f"**Valor:** `{davies_bouldin_label(m.get('davies_bouldin'))}`")
                st.caption(
                    "Razão entre dispersão intra-cluster e distância entre centroides. "
                    "**Escala: [0, ∞) — menor é melhor.**"
                )

            with cm3:
                st.markdown("#### Calinski-Harabasz Index")
                st.markdown(f"**Valor:** `{calinski_harabasz_label(m.get('calinski_harabasz'))}`")
                st.caption(
                    "Razão entre dispersão inter-cluster e intra-cluster. "
                    "Análogo ao R² do clustering. "
                    "**Escala: (0, ∞) — maior é melhor.**"
                )

            st.markdown("---")
            st.subheader("Estatísticas por Cluster")
            st.dataframe(r.cluster_stats, use_container_width=True, hide_index=True)
            st.caption(
                "Médias calculadas sobre as features usadas no clustering + Idade (se disponível)."
            )


# ────────────────────────────────────────────────────────────────────────
# 👥  PERFIS
# ────────────────────────────────────────────────────────────────────────
elif page == "👥 Perfis":
    st.title("👥 Análise de Perfis por Cluster")

    r = _results()
    if not st.session_state.model_ready or r is None:
        st.warning("⚠️ Treine o modelo primeiro usando a barra lateral.")
        st.stop()

    df_p = r.df_clustered
    feats_p = r.feature_cols
    color_col = "Persona" if "Persona" in df_p.columns else "Cluster"
    color_map = PERSONA_COLORS if color_col == "Persona" else None

    tab_box, tab_genero, tab_resumo = st.tabs(
        ["📦 Boxplots por Grupo", "⚧ Distribuição de Gênero", "📋 Resumo & Recomendações"]
    )

    # ── Boxplots ──────────────────────────────────────────
    with tab_box:
        st.subheader("Distribuição das Features por Grupo")

        for feat in feats_p:
            fig_box = px.box(
                df_p, x=color_col, y=feat,
                color=color_col,
                title=f"Distribuição de {feat} por {color_col}",
                color_discrete_map=color_map,
                points="outliers",
            )
            st.plotly_chart(_dark(fig_box), use_container_width=True)

        if "Age" in df_p.columns:
            fig_age = px.box(
                df_p, x=color_col, y="Age",
                color=color_col,
                title=f"Distribuição de Idade por {color_col}",
                color_discrete_map=color_map,
                labels={"Age": "Idade"},
                points="outliers",
            )
            st.plotly_chart(_dark(fig_age), use_container_width=True)

    # ── Distribuição de Gênero ────────────────────────────
    with tab_genero:
        if "Genre" in df_p.columns:
            df_cnt = (
                df_p.groupby([color_col, "Genre"])
                .size()
                .reset_index(name="Contagem")
            )

            fig_gen = px.bar(
                df_cnt, x=color_col, y="Contagem", color="Genre",
                barmode="group",
                title=f"Distribuição de Gênero por {color_col}",
                color_discrete_sequence=["#e94560", "#4ecca3"],
            )
            st.plotly_chart(_dark(fig_gen), use_container_width=True)

            fig_pct = px.bar(
                df_cnt, x=color_col, y="Contagem", color="Genre",
                barmode="relative",
                title="Proporção de Gênero por Grupo (100%)",
                color_discrete_sequence=["#e94560", "#4ecca3"],
            )
            st.plotly_chart(_dark(fig_pct), use_container_width=True)
        else:
            st.info("ℹ️ Coluna 'Genre' não encontrada no dataset atual.")

    # ── Resumo & Recomendações ────────────────────────────
    with tab_resumo:
        st.subheader("Tabela Resumo por Grupo")
        resumo_df = build_resumo_table(df_p, color_col, feats_p)
        st.dataframe(resumo_df, use_container_width=True, hide_index=True)

        if "Persona" in df_p.columns:
            st.markdown("---")
            st.subheader("🎯 Recomendações Estratégicas de Negócio")

            for persona, acao in PERSONA_ACTIONS.items():
                grupo = df_p[df_p["Persona"] == persona]
                if grupo.empty:
                    continue
                perc = len(grupo) / len(df_p) * 100
                msg = (
                    f"**{persona}** — {len(grupo)} clientes ({perc:.1f}% da base)  \n"
                    f"➡️  {acao}"
                )
                _persona_box(persona, msg)
