"""
src/pipeline.py
Orquestrador do pipeline de Machine Learning — Segmentação de Clientes.

Implementa o padrão FACADE: encapsula todas as etapas CRISP-DM em uma única
interface consistente. O chamador (app/main.py) não precisa conhecer a
implementação interna de nenhum módulo de ML.

Etapas executadas em sequência:
    1. Preparação e normalização das features (StandardScaler)
    2. Treinamento K-Means com k-means++ initialization
    3. Avaliação quantitativa (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    4. Cálculo do Método do Cotovelo (K=1..k_max)
    5. Atribuição de personas de negócio por centroide
    6. Geração de estatísticas descritivas por cluster

Uso:
    >>> from src.pipeline import CustomerSegmentationPipeline, PipelineConfig
    >>> pipeline = CustomerSegmentationPipeline(PipelineConfig(k=5))
    >>> results = pipeline.run(df, feature_cols=["Annual Income (k$)", "Spending Score (1-100)"])
    >>> print(f"Silhouette: {results.metrics['silhouette']:.3f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation import compute_cluster_stats, evaluate_clustering
from src.preprocessing import prepare_features
from src.train import compute_elbow, train_kmeans
from src.utils import assign_personas


# ══════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Parâmetros do pipeline K-Means.

    Todos os valores têm defaults funcionais — instanciar sem argumentos
    reproduz o resultado do TCC (K=5, Mall Customers).
    """
    k: int = 5                  # Número de clusters
    random_state: int = 42      # Semente para reprodutibilidade
    n_init: int = 10            # Execuções com inicializações diferentes
    k_max_elbow: int = 10       # K máximo para o gráfico do cotovelo


# ══════════════════════════════════════════════
# RESULTADO
# ══════════════════════════════════════════════

@dataclass
class PipelineResults:
    """
    Contêiner com todos os artefatos produzidos pelo pipeline.

    Projetado para ser armazenado em st.session_state e consumido
    diretamente pela camada de visualização — sem necessidade de
    recalcular qualquer coisa na UI.

    Atributos:
        df_clustered  : DataFrame original + colunas "Cluster" e "Persona"
        centers_df    : Centroides na escala original + "Cluster" + "Persona"
        X_scaled      : Features normalizadas (para métricas e visualizações)
        labels        : Rótulo de cluster por amostra (n_samples,)
        metrics       : Dict com silhouette, davies_bouldin, calinski_harabasz, inertia
        elbow_df      : DataFrame K × WCSS para o gráfico do cotovelo
        cluster_stats : Estatísticas descritivas por cluster/persona
        feature_cols  : Lista de features utilizadas
        k             : K efetivamente usado
        scaler        : StandardScaler ajustado (para inverse_transform futuro)
    """
    df_clustered: pd.DataFrame
    centers_df: pd.DataFrame
    X_scaled: np.ndarray
    labels: np.ndarray
    metrics: Dict[str, Optional[float]]
    elbow_df: pd.DataFrame
    cluster_stats: pd.DataFrame
    feature_cols: List[str]
    k: int
    scaler: StandardScaler


# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

class CustomerSegmentationPipeline:
    """
    Pipeline completo de segmentação de clientes com K-Means.

    Interface mínima:
        pipeline = CustomerSegmentationPipeline()
        results  = pipeline.run(df, feature_cols, k=5)

    O pipeline é stateful: após run(), os resultados ficam disponíveis
    em pipeline.results sem precisar executar novamente.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config: PipelineConfig = config or PipelineConfig()
        self._results: Optional[PipelineResults] = None

    # ── Interface pública ────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        k: Optional[int] = None,
    ) -> PipelineResults:
        """
        Executa o pipeline completo e retorna PipelineResults.

        Args:
            df           : DataFrame original (não normalizado)
            feature_cols : colunas numéricas para clustering (mínimo 2)
            k            : sobrescreve PipelineConfig.k se fornecido

        Returns:
            PipelineResults com todos os artefatos produzidos.

        Raises:
            ValueError: se feature_cols tiver menos de 2 elementos.
        """
        if len(feature_cols) < 2:
            raise ValueError(
                f"São necessárias pelo menos 2 features para clustering. "
                f"Recebido: {feature_cols}"
            )

        if k is not None:
            self.config.k = k

        # ── 1. Preparação: seleção + normalização ──────────
        X, scaler = prepare_features(df, feature_cols)

        # ── 2. Treinamento K-Means ─────────────────────────
        labels, centers_scaled, inertia = train_kmeans(
            X,
            k=self.config.k,
            random_state=self.config.random_state,
        )

        # ── 3. Avaliação quantitativa ──────────────────────
        metrics = evaluate_clustering(X, labels, inertia)

        # ── 4. Método do cotovelo ──────────────────────────
        elbow_df = compute_elbow(
            df,
            feature_cols=feature_cols,
            k_max=self.config.k_max_elbow,
            random_state=self.config.random_state,
        )

        # ── 5. Centroides → escala original ───────────────
        centers_orig = scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_orig, columns=feature_cols)
        centers_df["Cluster"] = np.arange(self.config.k)

        # ── 6. Atribuir clusters e personas ao df ─────────
        df_c = df.copy()
        df_c["Cluster"] = labels
        df_c, centers_df = assign_personas(df_c, centers_df, feature_cols)

        # ── 7. Estatísticas descritivas por cluster ────────
        group_col = "Persona" if "Persona" in df_c.columns else "Cluster"
        cluster_stats = compute_cluster_stats(df_c, feature_cols, group_col)

        # ── 8. Persistir resultado ─────────────────────────
        self._results = PipelineResults(
            df_clustered=df_c,
            centers_df=centers_df,
            X_scaled=X,
            labels=labels,
            metrics=metrics,
            elbow_df=elbow_df,
            cluster_stats=cluster_stats,
            feature_cols=feature_cols,
            k=self.config.k,
            scaler=scaler,
        )

        return self._results

    # ── Propriedades ─────────────────────────────────────────
    @property
    def results(self) -> Optional[PipelineResults]:
        """Último PipelineResults produzido. None se run() nunca foi chamado."""
        return self._results

    @property
    def is_fitted(self) -> bool:
        """True após ao menos uma execução bem-sucedida de run()."""
        return self._results is not None

    def __repr__(self) -> str:
        status = f"K={self.config.k}" if self.is_fitted else "não treinado"
        return f"CustomerSegmentationPipeline({status})"
