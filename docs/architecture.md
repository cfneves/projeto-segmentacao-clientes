# Arquitetura do Projeto

## Decisão Central: Padrão Facade no Pipeline

O `app/main.py` **nunca importa sklearn**. Toda lógica de ML fica encapsulada em `src/pipeline.py`.

```
app/main.py  ──run()──►  CustomerSegmentationPipeline
                 ◄──────  PipelineResults (dataclass)
```

## Fluxo de Dados

```
CSV / URL
   │
   ▼
src/preprocessing.py
   load_default_data() / load_uploaded_data()
   prepare_features() → StandardScaler
   │
   ▼
src/train.py
   train_kmeans() → labels, centers, inertia
   compute_elbow() → K × WCSS
   │
   ▼
src/evaluation.py
   evaluate_clustering() → silhouette, davies_bouldin, calinski_harabasz
   compute_cluster_stats() → tabela descritiva
   │
   ▼
src/utils.py
   assign_personas() → Cluster → Persona (negócio)
   │
   ▼
src/pipeline.py
   PipelineResults (dataclass) ← todos os artefatos
   │
   ▼
app/main.py
   st.session_state.results = PipelineResults
   Renderização das 4 páginas
```

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---|---|---|
| Inicialização KMeans | `k-means++` | Evita mínimos locais; converge mais rápido |
| Normalização | `StandardScaler` | KMeans sensível a escala |
| Métricas | 3 complementares | Silhouette sozinho não é suficiente |
| Config | YAML externo | Sem hardcode; alterável sem modificar código |
| Estado Streamlit | `PipelineResults` único | Um objeto tipado com tudo |
| Personas | Baseadas em medianas | Interpretável para qualquer K |
