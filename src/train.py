"""
src/train.py
Responsável por: treinamento do K-Means, método do cotovelo e métricas de avaliação.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# Método do Cotovelo
# ──────────────────────────────────────────────
def compute_elbow(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_max: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula a inércia (WCSS) para K = 1 … k_max.

    Retorna DataFrame com colunas 'K' e 'WCSS', pronto para plotar.
    """
    X: np.ndarray = StandardScaler().fit_transform(df[feature_cols].dropna().values)

    wcss: List[float] = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init=10)
        km.fit(X)
        wcss.append(float(km.inertia_))

    return pd.DataFrame({"K": list(range(1, k_max + 1)), "WCSS": wcss})


# ──────────────────────────────────────────────
# Treinamento do modelo final
# ──────────────────────────────────────────────
def train_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Treina o K-Means e retorna:
        labels          — rótulo de cluster para cada ponto (n_samples,)
        cluster_centers — centroides no espaço normalizado (k × n_features)
        inertia         — WCSS final (soma das distâncias ao centroide)
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=random_state,
        n_init=10,
    )
    labels: np.ndarray = km.fit_predict(X)
    return labels, km.cluster_centers_, float(km.inertia_)


# ──────────────────────────────────────────────
# Métricas de avaliação
# ──────────────────────────────────────────────
def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Calcula o Coeficiente de Silhueta.
    Retorna None se o número de clusters for inválido (< 2 ou ≥ n_amostras).

    Interpretação:
        ≥ 0.70 — excelente separação
        0.50–0.69 — boa separação
        0.25–0.49 — separação moderada
        < 0.25 — clusters sobrepostos
    """
    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(X):
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def silhouette_label(score: Optional[float]) -> str:
    """Converte o silhouette score em rótulo qualitativo."""
    if score is None:
        return "—"
    if score >= 0.70:
        return f"{score:.3f} (Excelente)"
    if score >= 0.50:
        return f"{score:.3f} (Bom)"
    if score >= 0.25:
        return f"{score:.3f} (Moderado)"
    return f"{score:.3f} (Fraco)"
