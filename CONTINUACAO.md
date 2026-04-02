# CONTINUAÇÃO DO PROJETO — Guia Completo de Retomada

> **Para quem:** qualquer desenvolvedor (ou instância de IA) que precise continuar este projeto sem ter acompanhado o desenvolvimento anterior.
> **Última atualização:** 2025 | Versão 2.0.0

---

## 1. CONTEXTO DO PROJETO

**O que é:**
TCC de Pós-Graduação em Ciência de Dados com IA (UniSENAI).
Segmentação de clientes de supermercado com K-Means, entregue como:
1. Notebook Jupyter com pipeline CRISP-DM completo (documentação acadêmica)
2. Aplicação Streamlit interativa (deployment/portfólio)

**Autor:** Cláudio Ferreira Neves — Especialista em Dados II
**Orientador:** Prof. Willian Daniel de Mattos
**GitHub:** https://github.com/cfneves/projeto-segmentacao-clientes
**App ao vivo:** https://projeto-segmentacao-clientes-unisenai.streamlit.app/

---

## 2. ESTADO ATUAL DO PROJETO

### ✅ O que está completo e funcionando

| Componente | Status | Observação |
|---|---|---|
| `app/main.py` | ✅ Completo | 4 páginas, tema light, pipeline integrado |
| `src/preprocessing.py` | ✅ Completo | Carga, validação, StandardScaler |
| `src/train.py` | ✅ Completo | K-Means + cotovelo |
| `src/evaluation.py` | ✅ Completo | 3 métricas + labels qualitativos |
| `src/pipeline.py` | ✅ Completo | Facade + PipelineConfig + PipelineResults |
| `src/utils.py` | ✅ Completo | Personas, cores, ações de negócio |
| `config/settings.yaml` | ✅ Completo | Todos os parâmetros centralizados |
| `CLAUDE.md` | ✅ Completo | Memória e guia do projeto |
| `README.md` | ✅ Completo | Profissional com badges, Streamlit link |
| `.streamlit/config.toml` | ✅ Completo | Tema light (#ffffff / #36454f) |
| `.github/workflows/ci.yml` | ✅ Completo | CI que testa importações + pipeline |
| `.github/ISSUE_TEMPLATE/` | ✅ Completo | Templates bug + feature |
| `data/sample_customers.csv` | ✅ Completo | 200 clientes sintéticos (5 clusters) |
| `reports/` | ✅ Completo | 4 PNGs + HTML EDA |
| GitHub repo "About" | ✅ Atualizado | Descrição + URL + 12 topics/tags |
| Streamlit Cloud deploy | ✅ Ativo | Redeploy automático a cada push |

### ⚠️ O que está pendente

| Item | Arquivo | O que falta |
|---|---|---|
| **Conclusão do notebook** | `notebooks/proj_pipeline_claudiofneves.ipynb` — célula 23 | Texto de encerramento escrito pelo aluno (ver Seção 6) |
| `app.py` (raiz) | Versão original do TCC | Mantida como referência — não integrada com `src/` |
| `models/` | Pasta criada, vazia | Poderia salvar o modelo treinado (.pkl) para deploy sem retreinar |
| `docs/` | Apenas `architecture.md` | Poderia ter diagrama visual da arquitetura |

---

## 3. ARQUITETURA — ENTENDA EM 2 MINUTOS

```
ENTRADA (dados)
     │
     ▼
src/preprocessing.py          ← carga CSV (local ou URL), validação, StandardScaler
     │
     ▼
src/train.py                  ← KMeans (k-means++), compute_elbow
     │
     ▼
src/evaluation.py             ← silhouette, davies_bouldin, calinski_harabasz
     │
     ▼
src/utils.py                  ← assign_personas (centroide vs mediana)
     │
     ▼
src/pipeline.py               ← FACADE: orquestra tudo, retorna PipelineResults
     │
     ▼
app/main.py                   ← UI STREAMLIT: consome PipelineResults, zero sklearn
```

**Regra de ouro:** `app/main.py` **nunca importa sklearn**. Qualquer lógica de ML vai em `src/`.

**Estado Streamlit:** tudo fica em `st.session_state.results` (um único `PipelineResults`).

---

## 4. COMO RODAR LOCALMENTE

```bash
# 1. Clonar
git clone https://github.com/cfneves/projeto-segmentacao-clientes.git
cd projeto-segmentacao-clientes

# 2. Ambiente virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Dependências
pip install -r requirements.txt

# 4. Rodar o app novo (v2)
streamlit run app/main.py

# 5. Rodar o app original do TCC (versão raiz)
streamlit run app.py

# 6. Abrir notebook
jupyter notebook notebooks/proj_pipeline_claudiofneves.ipynb
```

---

## 5. DESIGN SYSTEM — TOKENS DE COR

### Tema atual: Light

| Token | Valor | Uso |
|---|---|---|
| `--bg-page` | `#ffffff` | Fundo da página |
| `--bg-secondary` | `#f4f6f8` | Sidebar, cards, expanders |
| `--bg-chart` | `#f8fafb` | plot_bgcolor dos gráficos Plotly |
| `--text-primary` | `#36454f` | Texto principal (charcoal) |
| `--text-secondary` | `#8a9ba8` | Labels, captions |
| `--text-strong` | `#1a2a33` | Títulos, valores KPI |
| `--border` | `#e1e4e8` | Bordas de containers |
| `--accent-red` | `#e94560` | CTA, aba ativa, alerta |
| `--accent-teal` | `#1ab394` | Sucesso, KPI verde |
| `--accent-blue` | `#3b82f6` | Info, KPI azul |
| `--accent-amber` | `#f59e0b` | Aviso, KPI laranja |
| `--accent-purple` | `#8b5cf6` | Neutro, KPI roxo |

### Personas — cores (usadas nos gráficos Plotly)

| Persona | Cor Hex |
|---|---|
| Clientes Alvo (VIPs) | `#4ecca3` |
| Econômicos Ricos | `#4895ef` |
| Gastadores (Impulsivos) | `#f7b731` |
| Cautelosos | `#e94560` |
| Clientes Padrão | `#a29bfe` |

> ⚠️ Se mudar o tema para dark novamente, atualizar: `config.toml` (base, backgroundColor, textColor) + CSS inteiro em `app/main.py` + `_dark()` function.

---

## 6. PENDÊNCIA CRÍTICA — CONCLUSÃO DO NOTEBOOK

**Arquivo:** `notebooks/proj_pipeline_claudiofneves.ipynb`
**Célula:** 23 (última) — tipo `markdown`
**Conteúdo atual:** `"**ALUNOS**: Façam um encerramento/conclusão do projeto. Repassem os principais aspectos e resultados."`

### O que precisa ser escrito ali:

A conclusão deve cobrir, com as palavras do próprio Cláudio:
1. Resumo dos resultados (K=5, Silhouette ≈ 0.55)
2. Descrição das 5 personas encontradas
3. Qual persona representa maior valor de negócio e por quê
4. Limitações do modelo (dataset pequeno, apenas 2 features)
5. Possíveis próximos passos (DBSCAN, mais features, dados reais)

**Como editar:** abrir o notebook, clicar na célula 23, editar o markdown e salvar. Depois commitar o `.ipynb` atualizado.

---

## 7. HISTÓRICO DE DECISÕES TÉCNICAS

| Decisão | O que foi escolhido | Por que não outra coisa |
|---|---|---|
| Algoritmo | K-Means | Requisito do TCC; interpretável para banca |
| K=5 | Padrão no slider | Cotovelo claro no Mall Customers nesse ponto |
| 2 features de clustering | Renda + Gasto | Maior correlação visual; Age/Genre só em análise de perfil |
| Normalização | StandardScaler | K-Means usa distância euclidiana — escalas diferentes distorcem |
| Padrão de arquitetura | Facade (`pipeline.py`) | UI nunca toca sklearn; troca de algoritmo não exige mudar app |
| Estado Streamlit | 1 objeto `PipelineResults` | Evita 6+ variáveis separadas no session_state |
| Classificação de personas | Centroide vs. mediana | Funciona para qualquer K; não depende de número de cluster fixo |
| Tema | Light (`#ffffff` / `#36454f`) | Decisão do cliente após teste com dark theme |
| Config | YAML externo | Parâmetros alteráveis sem modificar código |

---

## 8. INTEGRAÇÃO CONTÍNUA (GitHub Actions)

**Arquivo:** `.github/workflows/ci.yml`
**Trigger:** push para `main` ou PR para `main`

**O que o CI faz:**
1. Instala `requirements.txt`
2. Verifica que todos os módulos `src/` importam sem erro
3. Roda `CustomerSegmentationPipeline` com `data/sample_customers.csv` e valida que `silhouette is not None`

**Se o CI falhar:** verifique se algum import em `src/` quebrou, ou se `data/sample_customers.csv` foi removido do repositório.

---

## 9. DEPLOY — STREAMLIT CLOUD

| Campo | Valor |
|---|---|
| URL | https://projeto-segmentacao-clientes-unisenai.streamlit.app/ |
| Repositório | cfneves/projeto-segmentacao-clientes |
| Branch | main |
| Entry point | app/main.py |
| Redeploy | automático a cada push na main |
| Secrets | nenhum configurado atualmente |

**Se o app travar no deploy:** verificar se `requirements.txt` tem versões compatíveis. O `ydata-profiling` é a dependência mais pesada — pode causar timeout no primeiro build.

---

## 10. PRÓXIMAS EVOLUÇÕES SUGERIDAS (prioridade)

### Alta prioridade
- [ ] **Escrever a conclusão** da célula 23 do notebook (bloqueador acadêmico)
- [ ] **Salvar modelo treinado** em `models/kmeans_k5.pkl` para não retreinar no deploy

### Média prioridade
- [ ] **Adicionar DBSCAN** como algoritmo alternativo (sem precisar definir K)
- [ ] **Predição de novos clientes** — input manual na UI para classificar um cliente novo
- [ ] **Exportar resultados** — botão para baixar o CSV clusterizado

### Baixa prioridade
- [ ] **Mais features de clustering** (incluir Age como 3ª dimensão, visualização 3D)
- [ ] **Autenticação** na app (se for usar dados reais/sensíveis)
- [ ] **Testes unitários** em `tests/` para as funções de `src/`

---

## 11. COMANDOS GIT ÚTEIS

```bash
# Ver histórico completo
git log --oneline

# Criar branch para nova feature
git checkout -b feature/nova-funcionalidade

# Commitar e publicar
git add <arquivos>
git commit -m "tipo: descrição curta"
git push origin main

# Ver diferenças antes de commitar
git diff

# Voltar para versão anterior (cuidado — destrói alterações locais)
git checkout -- <arquivo>
```

**Convenção de commits usada neste projeto:**
- `feat:` — nova funcionalidade
- `fix:` — correção de bug
- `design:` — mudança visual/UX
- `docs:` — documentação
- `refactor:` — refatoração sem mudar comportamento

---

## 12. REFERÊNCIAS E LINKS RÁPIDOS

| Recurso | Link |
|---|---|
| App ao vivo | https://projeto-segmentacao-clientes-unisenai.streamlit.app/ |
| Repositório GitHub | https://github.com/cfneves/projeto-segmentacao-clientes |
| Dataset Mall Customers (Kaggle) | https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python |
| Dataset URL RAW | https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv |
| Streamlit Community Cloud | https://share.streamlit.io |
| Documentação scikit-learn KMeans | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html |
| Documentação CRISP-DM | https://www.datascience-pm.com/crisp-dm-2/ |

---

*Este arquivo deve ser atualizado sempre que uma decisão importante for tomada ou um componente for concluído.*
