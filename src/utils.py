"""
src/utils.py
Funções auxiliares: classificação de personas, descrições qualitativas e mapeamentos de UI.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Mapeamentos de UI — Personas
# ──────────────────────────────────────────────

# Cores usadas em todos os gráficos (mantém consistência entre tabs)
PERSONA_COLORS: Dict[str, str] = {
    "Clientes Alvo (VIPs)":      "#4ecca3",   # verde
    "Econômicos Ricos":          "#4895ef",   # azul
    "Gastadores (Impulsivos)":   "#f7b731",   # laranja
    "Cautelosos":                "#e94560",   # vermelho
    "Clientes Padrão":           "#a29bfe",   # roxo
}

# Tipo de caixa Streamlit para cada persona (st.success / warning / error / info)
PERSONA_BOX_TYPE: Dict[str, str] = {
    "Clientes Alvo (VIPs)":      "success",
    "Econômicos Ricos":          "info",
    "Gastadores (Impulsivos)":   "warning",
    "Cautelosos":                "error",
    "Clientes Padrão":           "info",
}

# Ações de negócio recomendadas por persona
PERSONA_ACTIONS: Dict[str, str] = {
    "Clientes Alvo (VIPs)":    (
        "Priorizar retenção, programa de fidelidade premium e benefícios personalizados."
    ),
    "Econômicos Ricos":        (
        "Ativar com produtos premium, estratégias de cross-sell e experiências exclusivas."
    ),
    "Gastadores (Impulsivos)": (
        "Campanhas de tendências, lançamentos relâmpago e marketing de urgência."
    ),
    "Cautelosos":              (
        "Focar em promoções de itens essenciais e comunicação de custo-benefício."
    ),
    "Clientes Padrão":         (
        "Nutrir com ofertas segmentadas para elevar frequência e ticket médio."
    ),
}

# Features padrão do Mall Customers usadas na classificação de personas
_INCOME_FEATURE: str = "Annual Income (k$)"
_SPEND_FEATURE: str = "Spending Score (1-100)"


# ──────────────────────────────────────────────
# Lógica de classificação
# ──────────────────────────────────────────────
def _classify_persona(
    income: float,
    spend: float,
    income_median: float,
    spend_median: float,
) -> str:
    """
    Classifica um centroide em uma das 5 personas com base em sua posição
    em relação às medianas do dataset (renda e gasto).
    """
    renda_alta = income >= income_median
    gasto_alto = spend >= spend_median

    if renda_alta and gasto_alto:
        return "Clientes Alvo (VIPs)"
    if renda_alta and not gasto_alto:
        return "Econômicos Ricos"
    if not renda_alta and gasto_alto:
        return "Gastadores (Impulsivos)"
    if not renda_alta and not gasto_alto:
        return "Cautelosos"
    return "Clientes Padrão"


def assign_personas(
    df_clustered: pd.DataFrame,
    centers_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Atribui personas a cada cluster com base na posição do centroide.

    Funciona apenas quando as features padrão do Mall Customers estão presentes.
    Caso contrário, usa rótulos genéricos ("Cluster 0", "Cluster 1", …).

    Retorna (df_clustered_com_persona, centers_df_com_persona).
    """
    df_out = df_clustered.copy()
    ctr_out = centers_df.copy()

    if _INCOME_FEATURE not in feature_cols or _SPEND_FEATURE not in feature_cols:
        # Fallback genérico para qualquer combinação de features
        label_map = {i: f"Cluster {i}" for i in ctr_out["Cluster"]}
        ctr_out["Persona"] = ctr_out["Cluster"].map(label_map)
        df_out["Persona"] = df_out["Cluster"].map(label_map)
        return df_out, ctr_out

    income_median: float = float(df_out[_INCOME_FEATURE].median())
    spend_median: float = float(df_out[_SPEND_FEATURE].median())

    ctr_out["Persona"] = ctr_out.apply(
        lambda r: _classify_persona(
            r[_INCOME_FEATURE], r[_SPEND_FEATURE], income_median, spend_median
        ),
        axis=1,
    )

    cluster_to_persona: Dict[int, str] = dict(
        zip(ctr_out["Cluster"], ctr_out["Persona"])
    )
    df_out["Persona"] = df_out["Cluster"].map(cluster_to_persona)

    return df_out, ctr_out


# ──────────────────────────────────────────────
# Descrição qualitativa
# ──────────────────────────────────────────────
def describe_level(value: float, p33: float, p66: float, label: str) -> str:
    """
    Converte um valor numérico em rótulo qualitativo (Baixo / Médio / Alto)
    usando os percentis 33 e 66 como limiares.
    """
    if value <= p33:
        return f"{label} Baixo"
    if value <= p66:
        return f"{label} Médio"
    return f"{label} Alto"


def build_resumo_table(
    df_clustered: pd.DataFrame,
    color_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Gera tabela resumo com média das features e contagem por grupo.
    """
    cols_to_avg = [f for f in feature_cols if f in df_clustered.columns]
    if "Age" in df_clustered.columns:
        cols_to_avg.append("Age")

    resumo = df_clustered.groupby(color_col)[cols_to_avg].mean().round(2)
    resumo.insert(0, "Total", df_clustered.groupby(color_col).size())
    resumo = resumo.reset_index().sort_values("Total", ascending=False)

    rename_map = {
        "Annual Income (k$)": "Renda Média (k$)",
        "Spending Score (1-100)": "Gasto Médio",
        "Age": "Idade Média",
    }
    resumo = resumo.rename(columns=rename_map)
    return resumo
