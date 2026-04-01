"""
src/preprocessing.py
Responsável por: carga, validação e preparação das features para o modelo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
DATA_URL: str = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
)

LOCAL_DATA_PATH: Path = Path("data") / "Mall_Customers.csv"

# Nomes padronizados para o dataset Mall Customers
COLS_STANDARD: List[str] = [
    "CustomerID",
    "Genre",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


# ──────────────────────────────────────────────
# Carga de dados
# ──────────────────────────────────────────────
def load_default_data() -> Optional[pd.DataFrame]:
    """
    Carrega o dataset padrão (Mall Customers).
    Tenta o arquivo local primeiro; faz fallback para a URL pública.
    """
    try:
        if LOCAL_DATA_PATH.exists():
            df = pd.read_csv(LOCAL_DATA_PATH)
        else:
            df = pd.read_csv(DATA_URL)

        # Forçar nomes padronizados se o número de colunas bater
        if df.shape[1] == len(COLS_STANDARD):
            df.columns = COLS_STANDARD

        return df
    except Exception as exc:
        st.error(f"Erro ao carregar dataset padrão: {exc}")
        return None


def load_uploaded_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Lê um arquivo CSV enviado via st.file_uploader.
    Tenta inferir encoding e separador automaticamente.
    """
    try:
        # Tenta UTF-8 primeiro, depois latin-1 como fallback
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")

        if df.empty:
            st.error("O arquivo enviado está vazio.")
            return None

        return df
    except Exception as exc:
        st.error(f"Erro ao processar arquivo: {exc}")
        return None


# ──────────────────────────────────────────────
# Inspeção de colunas
# ──────────────────────────────────────────────
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de colunas numéricas do DataFrame."""
    return df.select_dtypes(include=np.number).columns.tolist()


def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Verifica se todas as colunas selecionadas existem e são numéricas.
    Exibe um aviso no Streamlit em caso negativo.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.warning(f"Colunas não encontradas no dataset: {missing}")
        return False

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        st.warning(f"Colunas não numéricas selecionadas: {non_numeric}")
        return False

    return True


# ──────────────────────────────────────────────
# Preparação de features
# ──────────────────────────────────────────────
def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Seleciona as colunas, remove NaN e aplica StandardScaler.

    Retorna:
        X       — array normalizado (n_samples × n_features)
        scaler  — instância ajustada, para inverse_transform dos centroides
    """
    X_raw: np.ndarray = df[feature_cols].dropna().values
    scaler = StandardScaler()
    X: np.ndarray = scaler.fit_transform(X_raw)
    return X, scaler
