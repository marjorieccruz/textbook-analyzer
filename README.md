---
title: Textbook Analyzer
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: Análise quali-quantitativa de livros didáticos em PDF
tags:
  - nlp
  - education
  - pdf
  - lda
  - maxqda
  - portuguese
  - textbook-analysis
---

# 📚 Textbook Analyzer

Ferramenta open source para análise quali-quantitativa de livros didáticos em PDF, com exportação para MAXQDA.

## Funcionalidades

- 📊 Métricas de legibilidade (Flesch, Gunning Fog, TTR)
- 🗺️ Modelagem de tópicos LDA por capítulo
- 🤖 Análise qualitativa via LLM (Claude ou OpenAI — configure em Settings)
- 📤 Exportação para MAXQDA (.mx20)

## Configuração do LLM

Para usar análise via IA, vá em **Settings → Repository secrets** e adicione:
- `ANTHROPIC_API_KEY` — para usar Claude
- `OPENAI_API_KEY` — para usar GPT-4o

Sem chaves configuradas, todas as outras análises funcionam normalmente.

## Uso local

```bash
git clone https://huggingface.co/spaces/marjorieccruz/textbook-analyzer
cd textbook-analyzer
pip install -r requirements.txt
streamlit run app.py
```
