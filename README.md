<div align="center">

# 🛍️ Segmentação de Clientes com K-Means

**TCC — Pós-Graduação em Ciência de Dados com Inteligência Artificial**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Metodologia](https://img.shields.io/badge/Metodologia-CRISP--DM-9B59B6)](https://www.datascience-pm.com/crisp-dm-2/)

</div>

---

## 📋 Sobre o Projeto

> Aplicação de **Aprendizado Não Supervisionado** para segmentação de clientes de supermercado, gerando personas de negócio acionáveis a partir de dados de renda e comportamento de consumo.

**Contexto:** Um supermercado coleta dados de clientes via programa de fidelidade, mas trata todos uniformemente. Isso reduz a eficácia do marketing e o ROI de campanhas.

**Solução:** Usar o algoritmo **K-Means** para identificar grupos naturais de clientes e criar estratégias personalizadas para cada perfil.

**Resultado:** 5 clusters com **Silhouette Score ≈ 0.55**, traduzidos em personas de negócio com ações recomendadas.

---

## 🎯 Objetivos

| Tipo | Descrição |
|---|---|
| **Negócio** | Identificar segmentos de clientes para personalização de campanhas de marketing |
| **Técnico** | Aplicar K-Means com validação quantitativa (3 métricas) e qualitativa (personas) |
| **Acadêmico** | Seguir rigorosamente a metodologia CRISP-DM em todas as 6 etapas |

---

## 🧠 O Algoritmo K-Means

O **K-Means** é um algoritmo de clusterização que agrupa pontos de dados minimizando a **Inércia (WCSS — Within-Cluster Sum of Squares)**:

```
WCSS = Σ Σ ||x_i - μ_k||²
```

**Como funciona em 4 passos:**

```
1. Inicialização  →  K centroides posicionados com k-means++
2. Atribuição     →  Cada ponto vai para o centroide mais próximo (distância euclidiana)
3. Atualização    →  Centroides recalculados como média dos pontos do cluster
4. Convergência   →  Repetir 2 e 3 até centroides estabilizarem
```

**Por que StandardScaler é obrigatório?**
K-Means usa distância euclidiana. Variáveis em escalas diferentes (ex: Renda em k$ vs. Pontuação 1-100) distorcem os clusters. O `StandardScaler` normaliza para média=0, desvio=1.

**Escolha do K — Método do Cotovelo:**

```
Inércia alta         Cotovelo (K ideal)    Ganho marginal pequeno
     |                      |                      |
K=1 ───────────────────── K=5 ─────────────────── K=10
```

---

## 📊 Métricas de Avaliação

| Métrica | Escala | Interpretação | Resultado |
|---|---|---|---|
| **Silhouette Score** | [-1, 1] | Maior = melhor separação | ≈ 0.55 ✓ Bom |
| **Davies-Bouldin Index** | [0, ∞) | Menor = melhor compactação | ≈ 0.65 ✓ Bom |
| **Calinski-Harabasz** | (0, ∞) | Maior = melhor definição | ≈ 280 ✓ Bom |

---

## 👥 Personas de Negócio

| Persona | Renda | Gasto | Estratégia Recomendada |
|---|---|---|---|
| 🟢 **Clientes Alvo (VIPs)** | Alta | Alto | Retenção, fidelidade premium, benefícios exclusivos |
| 🔵 **Econômicos Ricos** | Alta | Baixo | Cross-sell, produtos premium, experiências exclusivas |
| 🟡 **Gastadores (Impulsivos)** | Baixa | Alto | Campanhas relâmpago, tendências, urgência |
| 🔴 **Cautelosos** | Baixa | Baixo | Promoções de essenciais, custo-benefício |
| 🟣 **Clientes Padrão** | Média | Médio | Elevar frequência e ticket médio |

---

## 🏗️ Arquitetura do Projeto

```
projeto-segmentacao-clientes/
│
├── 📁 app/
│   └── main.py                  # Aplicação Streamlit (UI/AX)
│
├── 📁 src/                      # Lógica de ML — zero sklearn na UI
│   ├── preprocessing.py         # Carga, validação, StandardScaler
│   ├── train.py                 # K-Means + Método do Cotovelo
│   ├── evaluation.py            # Silhouette, Davies-Bouldin, Calinski-Harabasz
│   ├── pipeline.py              # Orquestrador Facade (PipelineResults)
│   └── utils.py                 # Personas, cores, ações de negócio
│
├── 📁 config/
│   └── settings.yaml            # Parâmetros centralizados
│
├── 📁 data/
│   └── sample_customers.csv     # Dataset sintético (200 clientes)
│
├── 📁 notebooks/
│   └── proj_pipeline_claudiofneves.ipynb   # Pipeline CRISP-DM completo
│
├── 📁 reports/                  # Artefatos visuais gerados
│   ├── cluster_plot_2d.png
│   ├── elbow_plot.png
│   ├── pairplot.png
│   ├── perfis_detalhados_clusters.png
│   └── eda_mall_customers.html
│
├── 📁 docs/                     # Documentação técnica
├── 📁 models/                   # Modelos serializados (futuro)
│
├── app.py                       # Versão original do TCC (raiz)
├── CLAUDE.md                    # Memória e guia do projeto
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

**Padrão arquitetural:** O `app/main.py` usa exclusivamente `CustomerSegmentationPipeline.run()` — nunca importa sklearn diretamente. Separação estrita entre camada de ML (`src/`) e camada de apresentação (`app/`).

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- pip

### 1. Clonar o repositório

```bash
git clone https://github.com/cfneves/projeto-segmentacao-clientes.git
cd projeto-segmentacao-clientes
```

### 2. Criar ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar a aplicação Streamlit

```bash
streamlit run app/main.py
```

Acesse em: **http://localhost:8501**

### 5. Executar o notebook (opcional)

```bash
jupyter notebook notebooks/proj_pipeline_claudiofneves.ipynb
```

---

## 📱 Demonstração da Aplicação

A aplicação possui 4 seções de navegação:

| Seção | Conteúdo |
|---|---|
| 🏠 **Início** | KPIs do modelo, como usar, sobre o algoritmo |
| 📊 **Exploração** | Preview dos dados, distribuições, correlação |
| 🎯 **Modelagem** | Cotovelo, scatter dos clusters, 3 métricas |
| 👥 **Perfis** | Boxplots, distribuição de gênero, recomendações |

**Funcionalidades:**
- Upload de CSV próprio ou uso do dataset padrão (Mall Customers)
- Seleção interativa de features para clustering
- Slider para escolha do K (2 a 10)
- Visualizações interativas com Plotly

---

## 🛠️ Stack Tecnológica

| Categoria | Tecnologia | Versão |
|---|---|---|
| Linguagem | Python | 3.10+ |
| ML | scikit-learn | ≥ 1.4 |
| Dados | pandas, numpy | ≥ 2.0, ≥ 1.26 |
| Visualização | Plotly, Matplotlib, Seaborn | ≥ 5.20 |
| Web App | Streamlit | ≥ 1.35 |
| EDA | ydata-profiling | ≥ 4.6 |
| Configuração | PyYAML | ≥ 6.0 |
| Metodologia | CRISP-DM | — |
| IDE | VS Code | — |

---

## 📐 Metodologia CRISP-DM

```
┌─────────────────────────────────────────────────────────┐
│                      CRISP-DM                           │
│                                                         │
│  1. Entendimento    →  Problema de negócio definido     │
│     do Negócio                                          │
│                                                         │
│  2. Entendimento    →  Mall Customers, EDA,             │
│     dos Dados          ydata-profiling, pairplot        │
│                                                         │
│  3. Preparação      →  Seleção de features,             │
│     dos Dados          StandardScaler                   │
│                                                         │
│  4. Modelagem       →  K-Means (k-means++, K=5)         │
│                        Método do Cotovelo               │
│                                                         │
│  5. Avaliação       →  Silhouette ≈ 0.55                │
│                        Davies-Bouldin, Calinski-Harabasz│
│                        Análise qualitativa por persona  │
│                                                         │
│  6. Deployment      →  Streamlit App                    │
│                        Personas + Recomendações         │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Fonte dos Dados

- **Dataset:** Mall Customers (público — Kaggle)
- **Registros:** 200 clientes
- **Features para modelagem:** `Annual Income (k$)`, `Spending Score (1-100)`
- **Features para análise de perfil:** `Age`, `Genre`

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE) para mais informações.

---

## 👨‍💻 Autor

**Cláudio Ferreira Neves**

- Cargo: Analista de Dados Sênior
- Curso: Pós-Graduação em Ciência de Dados com IA — UniSENAI
- Orientador: Prof. Willian Daniel de Mattos

---

<div align="center">
<sub>Projeto desenvolvido como TCC da Pós-Graduação em Ciência de Dados e IA — UniSENAI | 2025</sub>
</div>
