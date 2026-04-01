## Descrição

Descreva o que foi alterado e por quê.

## Tipo de Mudança

- [ ] 🐛 Bug fix
- [ ] ✨ Nova funcionalidade
- [ ] 📚 Documentação
- [ ] 🔧 Refatoração
- [ ] 📊 Dados / Notebook

## Checklist

- [ ] Código segue os padrões do projeto (type hints, docstrings)
- [ ] Importações de sklearn estão em `src/` (nunca em `app/main.py`)
- [ ] Testado com dataset `data/sample_customers.csv`
- [ ] `CLAUDE.md` atualizado se houver mudança arquitetural
- [ ] README atualizado se necessário

## Teste

Descreva como testar as mudanças:

```bash
streamlit run app/main.py
# Carregar dataset padrão → Treinar com K=5 → Verificar Silhouette Score
```
