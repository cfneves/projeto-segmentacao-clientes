"""
src/evaluation.py
Avaliação quantitativa e qualitativa dos clusters K-Means.

Métricas implementadas:
    Silhouette Score      [-1, 1]   — maior é melhor (separação inter-cluster)
    Davies-Bouldin Index  [0, ∞)    — menor é melhor (compactação intra-cluster)
    Calinski-Harabasz     (0, ∞)    — maior é melhor (razão dispersão inter/intra)
    Inércia (WCSS)        (0, ∞)    — usada no método do cotovelo

Separar métricas de treinamento é fundamental em produção:
o pipeline de avaliação pode rodar de forma independente, inclusive
em dados de validação holdout, sem retreinar o modelo.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


# ══════════════════════════════════════════════
# CÁLCULO DAS MÉTRICAS
# ══════════════════════════════════════════════

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: float,
) -> Dict[str, Optional[float]]:
    """
    Calcula o conjunto completo de métricas de avaliação do clustering.

    Args:
        X       : array normalizado (n_samples × n_features)
        labels  : rótulos de cluster por amostra (n_samples,)
        inertia : WCSS retornado pelo KMeans.fit().inertia_

    Returns:
        Dicionário com chaves:
            silhouette, davies_bouldin, calinski_harabasz, inertia
        Valores None quando o número de clusters é inválido (< 2).
    """
    result: Dict[str, Optional[float]] = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "inertia": float(inertia),
    }

    n_clusters = len(set(labels))
    # Métricas requerem pelo menos 2 clusters e menos clusters que amostras
    if n_clusters < 2 or n_clusters >= len(X):
        return result

    try:
        result["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        pass

    try:
        result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        pass

    try:
        result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        pass

    return result


def compute_silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calcula o coeficiente de silhueta individual para cada ponto.
    Útil para visualizações avançadas de análise por cluster.

    Returns:
        Array (n_samples,) com score individual, ou None em caso de erro.
    """
    try:
        return silhouette_samples(X, labels)
    except Exception:
        return None


# ══════════════════════════════════════════════
# RÓTULOS QUALITATIVOS
# ══════════════════════════════════════════════

def silhouette_label(score: Optional[float]) -> str:
    """
    Converte Silhouette Score em rótulo qualitativo.
    Escala: [-1, 1] — quanto MAIOR, melhor separação entre clusters.
    """
    if score is None:
        return "—"
    if score >= 0.70:
        return f"{score:.3f}  ★ Excelente"
    if score >= 0.50:
        return f"{score:.3f}  ✓ Bom"
    if score >= 0.25:
        return f"{score:.3f}  ~ Moderado"
    return f"{score:.3f}  ✗ Fraco"


def davies_bouldin_label(score: Optional[float]) -> str:
    """
    Converte Davies-Bouldin Index em rótulo qualitativo.
    Escala: [0, ∞) — quanto MENOR, melhor compactação intra-cluster.
    """
    if score is None:
        return "—"
    if score <= 0.50:
        return f"{score:.3f}  ★ Excelente"
    if score <= 1.00:
        return f"{score:.3f}  ✓ Bom"
    if score <= 1.50:
        return f"{score:.3f}  ~ Moderado"
    return f"{score:.3f}  ✗ Fraco"


def calinski_harabasz_label(score: Optional[float]) -> str:
    """
    Converte Calinski-Harabasz Index em rótulo qualitativo.
    Escala: (0, ∞) — quanto MAIOR, melhor definição dos clusters.
    Valores de referência variam com o dataset; limiares são orientativos.
    """
    if score is None:
        return "—"
    if score >= 500:
        return f"{score:.0f}  ★ Excelente"
    if score >= 200:
        return f"{score:.0f}  ✓ Bom"
    if score >= 100:
        return f"{score:.0f}  ~ Moderado"
    return f"{score:.0f}  ✗ Fraco"


# ══════════════════════════════════════════════
# ESTATÍSTICAS POR CLUSTER
# ══════════════════════════════════════════════

def compute_cluster_stats(
    df_clustered: pd.DataFrame,
    feature_cols: List[str],
    group_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Gera tabela de estatísticas descritivas por cluster.

    Inclui automaticamente 'Age' se presente no DataFrame.
    Colunas geradas: n_clientes, % base, <feature>_mean, <feature>_median, <feature>_std

    Args:
        df_clustered : DataFrame com coluna de grupo (Cluster ou Persona)
        feature_cols : features usadas no clustering
        group_col    : coluna de agrupamento ("Persona" ou "Cluster")

    Returns:
        DataFrame ordenado por n_clientes decrescente.
    """
    cols_to_analyze = [f for f in feature_cols if f in df_clustered.columns]
    if "Age" in df_clustered.columns and "Age" not in cols_to_analyze:
        cols_to_analyze.append("Age")

    if not cols_to_analyze:
        return pd.DataFrame()

    # Estatísticas descritivas
    stats = (
        df_clustered.groupby(group_col)[cols_to_analyze]
        .agg(["mean", "median", "std"])
        .round(2)
    )

    # Flatten multi-index: "Annual Income (k$)_mean" etc.
    stats.columns = [f"{col} ({stat})" for col, stat in stats.columns]

    # Contagem e percentual
    counts = df_clustered.groupby(group_col).size()
    stats.insert(0, "n_clientes", counts)
    stats.insert(
        1,
        "% da base",
        (counts / len(df_clustered) * 100).round(1).astype(str) + "%",
    )

    return stats.reset_index().sort_values("n_clientes", ascending=False)
