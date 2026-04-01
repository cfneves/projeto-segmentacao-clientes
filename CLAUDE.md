# CLAUDE.md — Memória e Guia Central do Projeto

> Este arquivo é a fonte de verdade para qualquer instância de IA (ou desenvolvedor)
> que trabalhe neste repositório. Leia antes de qualquer modificação.

---

## Identidade do Projeto

| Campo | Detalhe |
|---|---|
| **Nome** | Segmentação de Clientes com K-Means |
| **Tipo** | TCC de Pós-Graduação + Portfólio Profissional |
| **Autor** | Cláudio Ferreira Neves |
| **Instituição** | UniSENAI — Pós-Graduação em Ciência de Dados e IA |
| **Orientador** | Prof. Willian Daniel de Mattos |
| **Metodologia** | CRISP-DM (6 etapas completas) |
| **Versão** | 2.0.0 |
| **Status** | Produção / Portfólio |

---

## Contexto de Negócio

Um supermercado possui dados de clientes via programa de fidelidade.
Sem segmentação, todos recebem o mesmo tratamento de marketing — reduzindo eficácia e ROI.

**Problema:** Ausência de segmentação estruturada.
**Solução:** K-Means para identificar grupos naturais de clientes.
**Entrega:** Personas de negócio acionáveis + aplicação Streamlit interativa.

**Dataset:** Mall Customers (Kaggle público) — 200 clientes, 5 variáveis.
**Resultado:** K=5, Silhouette ≈ 0.55.

---

## Arquitetura do Projeto

```
projeto-segmentacao-clientes/
│
├── app/
│   └── main.py              ← UI Streamlit — ÚNICA camada de apresentação
│
├── src/
│   ├── preprocessing.py     ← Carga, validação e normalização de dados
│   ├── train.py             ← Treinamento K-Means e cálculo do cotovelo
│   ├── evaluation.py        ← Métricas: Silhouette, Davies-Bouldin, Calinski-Harabasz
│   ├── pipeline.py          ← Orquestrador Facade — ponto de entrada do ML
│   └── utils.py             ← Personas, cores, ações de negócio, funções auxiliares
│
├── config/
│   └── settings.yaml        ← Configuração central (sem hardcode no código-fonte)
│
├── data/
│   ├── Mall_Customers.csv   ← Dataset real (gerado na primeira execução — gitignored)
│   └── sample_customers.csv ← Dataset de exemplo sintético (200 clientes — versionado)
│
├── notebooks/
│   └── proj_pipeline_claudiofneves.ipynb  ← Pipeline CRISP-DM documentado
│
├── reports/                 ← Artefatos visuais gerados (PNGs, HTML EDA)
│
├── app.py                   ← Versão original (raiz) — mantida para referência do TCC
├── CLAUDE.md                ← ESTE ARQUIVO
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Decisão Arquitetural Central

> **`app/main.py` NUNCA importa `sklearn` diretamente.**

Toda lógica de ML fica encapsulada em `src/pipeline.py:CustomerSegmentationPipeline`.
A UI consome apenas `PipelineResults` — um dataclass com todos os artefatos prontos.

```
┌──────────────┐     run()      ┌──────────────────────────────┐
│  app/main.py │ ─────────────► │ CustomerSegmentationPipeline │
│  (UI only)   │ ◄───────────── │  preprocessing → train →     │
└──────────────┘  PipelineResults  evaluation → personas      │
                                └──────────────────────────────┘
```

---

## Padrões Adotados

### Código Python
- **`from __future__ import annotations`** — compatibilidade com Python 3.9+
- **Type hints** em todas as funções públicas
- **Docstrings** Google Style nas classes e funções não-triviais
- **Dataclasses** para configs e resultados (`PipelineConfig`, `PipelineResults`)
- Sem `print()` em produção — usar `logging` ou feedback via Streamlit

### Machine Learning
- Sempre usar `StandardScaler` antes do K-Means (algoritmo sensível a escala)
- `random_state=42` em **todos** os modelos para reprodutibilidade total
- `init="k-means++"` — converge mais rápido e com menor risco de mínimos locais
- Avaliar com **3 métricas complementares** (não só silhouette):
  - `silhouette_score`: [-1, 1] — separação inter-cluster (maior = melhor)
  - `davies_bouldin_score`: [0, ∞) — compactação intra-cluster (menor = melhor)
  - `calinski_harabasz_score`: (0, ∞) — razão dispersão inter/intra (maior = melhor)

### UI / Streamlit
- Tema escuro com paleta: `sidebar=#1a1a2e`, `plot_bg=#0f3460`, `accent=#e94560`
- `st.session_state` para **toda** persistência de estado entre rerenders
- Nunca chamar pipeline dentro de loops de renderização — apenas no botão "Treinar"
- `_dark_layout(fig)` — helper centralizado para tema dos gráficos Plotly

### Personas de Negócio
- Classificação baseada na posição do **centroide** vs. **medianas** do dataset
- Lógica exclusivamente em `src/utils.py:_classify_persona()` — não duplicar
- Cores, ações e tipo de caixa em `PERSONA_COLORS`, `PERSONA_ACTIONS`, `PERSONA_BOX_TYPE`
- Se features não forem os padrões Mall Customers → fallback genérico ("Cluster N")

---

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---|---|---|
| Algoritmo | K-Means | Requisito do TCC; interpretável para banca |
| K padrão | 5 | Cotovelo claro com Mall Customers |
| Features principais | Renda + Gasto | Maior correlação visual; Age/Genre só em análise |
| Normalização | StandardScaler | K-Means usa distância euclidiana — escalas diferentes distorcem clusters |
| Config | YAML | Centraliza parâmetros; alterável sem modificar código |
| Pattern pipeline | Facade | Isola UI de ML; facilita troca de algoritmo |
| State Streamlit | PipelineResults no session_state | Evita recomputação a cada interação |

---

## Como Evoluir o Projeto

### Adicionar nova métrica de avaliação
1. Adicionar cálculo em `src/evaluation.py:evaluate_clustering()`
2. Criar função `*_label()` para o rótulo qualitativo
3. Exibir no KPI card em `app/main.py` → página Modelagem

### Adicionar novo algoritmo (ex: DBSCAN)
1. Criar `train_dbscan()` em `src/train.py`
2. Criar `DbscanPipeline` em `src/pipeline.py` seguindo mesmo padrão de interface
3. Adicionar seletor de algoritmo no sidebar — UI não muda, apenas qual pipeline instanciar

### Suportar dataset customizado
- Personas retornam "Cluster N" automaticamente quando features padrão ausentes
- Sem hardcode de colunas fora de `src/utils.py` e `src/preprocessing.py`

### Deploy em produção
```bash
# Streamlit Community Cloud (recomendado para portfólio)
# 1. Push para GitHub
# 2. Acessar share.streamlit.io
# 3. Apontar para app/main.py

# Docker (produção corporativa)
# streamlit run app/main.py --server.port=8501 --server.headless=true
```

---

## Dependências Críticas

| Lib | Versão mínima | Motivo |
|---|---|---|
| scikit-learn | 1.4 | `davies_bouldin_score` na API estável |
| streamlit | 1.35 | `st.dataframe(hide_index=True)` |
| plotly | 5.20 | `add_vline` com annotation text |
| pandas | 2.0 | `pd.api.types.is_numeric_dtype` estável |
| pyyaml | 6.0 | Leitura do `config/settings.yaml` |

---

## Instruções para Claude / IA

- **NÃO** importar `sklearn` em `app/main.py` — lógica de ML vai em `src/`
- **NÃO** duplicar constantes de personas fora de `src/utils.py`
- **NÃO** usar `st.experimental_*` — somente APIs estáveis
- **SEMPRE** usar `CustomerSegmentationPipeline.run()` para treinar
- **SEMPRE** testar com Mall Customers (K=5) antes de qualquer commit
- **SEMPRE** manter retrocompatibilidade com `app.py` raiz (versão original do TCC)
- **MCP State:** ao retomar o projeto, verificar `st.session_state` keys e `PipelineResults`

---

## Skills Definidas no Projeto

| Skill | Módulo | Responsabilidade |
|---|---|---|
| `data_preprocessing` | `src/preprocessing.py` | Carga, validação, StandardScaler |
| `clustering_analysis` | `src/train.py` | K-Means, cotovelo, k-means++ |
| `cluster_evaluation` | `src/evaluation.py` | Silhouette, DB, CH, stats por cluster |
| `ml_pipeline` | `src/pipeline.py` | Orquestração Facade, PipelineResults |
| `persona_logic` | `src/utils.py` | Classificação, cores, ações de negócio |
| `streamlit_ui` | `app/main.py` | UI/AX, navegação, visualizações |

---

*Última atualização: 2025 | Versão 2.0.0*
