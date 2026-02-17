"""
app.py — Textbook Analyzer
Entry point para Hugging Face Spaces.
Multi-page app em arquivo único para compatibilidade máxima com HF.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Textbook Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Bootstrap: downloads NLTK/spaCy na primeira execução ─────────────────────
@st.cache_resource(show_spinner="⚙️ Configurando modelos NLP (apenas na primeira vez)...")
def bootstrap_nlp():
    import nltk
    import subprocess
    for resource in ["stopwords", "punkt", "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    # Baixa modelos spaCy se não existirem
    for model in ["pt_core_news_sm", "en_core_web_sm"]:
        try:
            import spacy
            spacy.load(model)
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", model],
                           capture_output=True)
    return True

bootstrap_nlp()

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#1f6feb; }
    .page-nav button { width:100%; margin-bottom:4px; }
    .metric-card {
        background:#f8f9fa; border:1px solid #dee2e6;
        border-radius:8px; padding:1rem; text-align:center;
    }
    .topic-pill {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:0.82rem; margin:2px;
    }
    div[data-testid="stSidebar"] { min-width: 220px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Navegação via sidebar ────────────────────────────────────────────────────
PAGES = {
    "🏠 Início": "home",
    "📤 1. Upload PDF": "upload",
    "📊 2. Análise Quantitativa": "quantitative",
    "🗺️ 3. Tópicos LDA": "topics",
    "🤖 4. Análise LLM": "llm",
    "📤 5. Exportar MAXQDA": "export",
}

with st.sidebar:
    st.markdown("## 📚 Textbook Analyzer")
    st.markdown("---")
    page = st.radio("Navegação", list(PAGES.keys()), label_visibility="collapsed")

    # Status da sessão
    st.markdown("---")
    st.markdown("**Status da sessão**")
    if "doc_structure" in st.session_state:
        doc = st.session_state["doc_structure"]
        st.success(f"✅ PDF carregado\n{doc.total_pages}p · {len(doc.chapters)}cap")
    else:
        st.info("○ Nenhum PDF")

    if "doc_metrics" in st.session_state:
        st.success("✅ Quantitativo OK")
    if "topic_result" in st.session_state:
        st.success("✅ Tópicos OK")
    if "llm_result" in st.session_state:
        st.success("✅ LLM OK")

    st.markdown("---")
    st.caption("MIT License · [GitHub](https://github.com/seu-usuario/textbook-analyzer)")

current = PAGES[page]

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: HOME
# ══════════════════════════════════════════════════════════════════════════════
if current == "home":
    st.markdown('<div class="main-title">📚 Textbook Analyzer</div>', unsafe_allow_html=True)
    st.markdown("**Análise quali-quantitativa de livros didáticos em PDF · Open Source · Exportação para MAXQDA**")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Análise Quantitativa")
        st.markdown("""
        - Índices Flesch, Flesch-Kincaid, Gunning Fog
        - Type-Token Ratio e densidade lexical
        - TF-IDF por capítulo
        - Progressão de complexidade ao longo do livro
        """)
        st.subheader("🗺️ Modelagem de Tópicos (LDA)")
        st.markdown("""
        - Identificação automática de temas
        - Detecção do número ideal de tópicos
        - Distribuição por capítulo
        - Visualização interativa
        """)

    with col2:
        st.subheader("🤖 Análise via LLM *(opcional)*")
        st.markdown("""
        - Resumos por capítulo
        - Objetivos de aprendizado
        - Conceitos-chave
        - Análise pedagógica geral
        
        *Configure `ANTHROPIC_API_KEY` ou `OPENAI_API_KEY` nos Secrets do Space.*
        """)
        st.subheader("📤 Exportação MAXQDA")
        st.markdown("""
        - Arquivo `.mx20` compatível com MAXQDA 2020+
        - Capítulos como documentos separados
        - Códigos automáticos por tópico LDA
        - Variáveis de documento com métricas
        """)

    st.divider()
    st.info("👈 **Comece pelo passo 1: Upload PDF** no menu lateral.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif current == "upload":
    st.title("📤 Upload e Processamento do PDF")

    uploaded = st.file_uploader(
        "Selecione o PDF do livro didático",
        type=["pdf"],
        help="PDFs nativos e escaneados (OCR automático). Máx: 200MB no Spaces.",
    )

    if uploaded:
        size_mb = uploaded.size / 1_048_576
        st.info(f"📄 **{uploaded.name}** · {size_mb:.1f} MB")

        col1, col2, col3 = st.columns(3)
        with col1:
            extract_images = st.checkbox("Extrair imagens", value=True)
        with col2:
            extract_tables = st.checkbox("Extrair tabelas", value=True)
        with col3:
            lang = st.selectbox("Idioma", ["pt", "en"],
                                format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English")

        if st.button("🚀 Processar PDF", type="primary", use_container_width=True):
            with st.spinner("Processando..."):
                try:
                    from core.ingestion import ingest_pdf

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    assets_dir = Path(tmp_path).parent / "assets"
                    prog = st.progress(0)
                    status = st.empty()

                    def cb(cur, tot, msg):
                        prog.progress(cur / max(tot, 1))
                        status.text(msg)

                    structure = ingest_pdf(
                        tmp_path, output_dir=assets_dir,
                        extract_images=extract_images,
                        extract_tables=extract_tables,
                        progress_callback=cb,
                    )

                    st.session_state["doc_structure"] = structure
                    st.session_state["doc_lang"] = lang
                    # Limpa análises anteriores ao carregar novo doc
                    for key in ["doc_metrics", "topic_result", "llm_result"]:
                        st.session_state.pop(key, None)

                    prog.progress(1.0)
                    status.text("✅ Concluído!")
                    os.unlink(tmp_path)
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro: {e}")
                    st.exception(e)

    if "doc_structure" in st.session_state:
        doc = st.session_state["doc_structure"]
        st.divider()
        st.subheader("✅ Documento carregado")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Páginas", doc.total_pages)
        c2.metric("Capítulos", len(doc.chapters))
        c3.metric("Imagens", sum(len(p.images) for p in doc.pages))
        c4.metric("Tabelas", sum(len(p.tables) for p in doc.pages))

        with st.expander("📋 Metadados"):
            st.json({"Título": doc.title, "Autor": doc.author, **doc.metadata})

        if doc.chapters:
            with st.expander(f"📚 {len(doc.chapters)} capítulos detectados"):
                for ch in doc.chapters:
                    st.markdown(f"- **{ch['title']}** · pp. {ch['start_page']}–{ch.get('end_page','?')}")

        with st.expander("👁️ Preview (3 primeiras páginas)"):
            for p in doc.pages[:3]:
                st.markdown(f"**Página {p.page_number}**")
                st.text((p.text[:400] + "...") if len(p.text) > 400 else p.text)
                st.divider()

        st.success("➡️ Continue para **Análise Quantitativa**.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: QUANTITATIVO
# ══════════════════════════════════════════════════════════════════════════════
elif current == "quantitative":
    st.title("📊 Análise Quantitativa")

    if "doc_structure" not in st.session_state:
        st.warning("⚠️ Faça o upload do PDF primeiro.")
        st.stop()

    structure = st.session_state["doc_structure"]
    lang = st.session_state.get("doc_lang", "pt")

    if "doc_metrics" not in st.session_state:
        if st.button("▶️ Executar análise", type="primary", use_container_width=True):
            with st.spinner("Analisando..."):
                try:
                    from core.quantitative import analyze_document
                    import plotly.express as px
                    import pandas as pd

                    prog = st.progress(0)
                    status = st.empty()

                    def cb(cur, tot, msg):
                        prog.progress(cur / max(tot, 1))
                        status.text(msg)

                    metrics = analyze_document(structure, lang=lang, progress_callback=cb)
                    st.session_state["doc_metrics"] = metrics
                    prog.progress(1.0)
                    status.text("✅ Concluído!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
                    st.exception(e)
        st.stop()

    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    metrics = st.session_state["doc_metrics"]

    # Métricas gerais
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Palavras", f"{metrics.total_words:,}")
    c2.metric("Sentenças", f"{metrics.total_sentences:,}")
    c3.metric("Parágrafos", f"{metrics.total_paragraphs:,}")
    c4.metric("Imagens", metrics.total_images)
    c5.metric("Tabelas", metrics.total_tables)

    st.divider()

    # Legibilidade
    if metrics.readability:
        r = metrics.readability
        st.subheader("📖 Legibilidade")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r.flesch_reading_ease,
                title={"text": "Flesch Reading Ease"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f6feb"},
                    "steps": [
                        {"range": [0, 30], "color": "#FADBD8"},
                        {"range": [30, 70], "color": "#FEF9E7"},
                        {"range": [70, 100], "color": "#D5F5E3"},
                    ],
                },
            ))
            fig.update_layout(height=220, margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**{r.interpretation}**")

        with col2:
            st.metric("Flesch-Kincaid Grade", f"{r.flesch_kincaid_grade:.1f}")
            st.metric("Gunning Fog", f"{r.gunning_fog:.1f}")
            st.metric("Média palavras/sentença", f"{r.avg_sentence_length:.1f}")

        with col3:
            if metrics.vocabulary:
                v = metrics.vocabulary
                st.metric("TTR", f"{v.ttr:.3f}", help="Type-Token Ratio")
                st.metric("Hapax Legomena", f"{v.hapax_legomena:,}")
                st.metric("Densidade Lexical", f"{v.lexical_density:.1%}")

    # Progressão
    if len(metrics.complexity_progression) > 1:
        st.divider()
        st.subheader("📈 Complexidade por Capítulo")
        labels = [ch.chapter_title[:25] for ch in metrics.chapters[:len(metrics.complexity_progression)]]
        df = pd.DataFrame({"Capítulo": range(1, len(metrics.complexity_progression)+1),
                           "Flesch": metrics.complexity_progression,
                           "Título": labels})
        fig = px.line(df, x="Capítulo", y="Flesch", hover_name="Título",
                      markers=True, color_discrete_sequence=["#1f6feb"],
                      title="Flesch por capítulo (maior = mais fácil)")
        fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.05)
        fig.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.05)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Tabela de capítulos
    if metrics.chapters:
        st.divider()
        st.subheader("📚 Por Capítulo")
        df_ch = pd.DataFrame([{
            "Capítulo": ch.chapter_title[:35],
            "Palavras": ch.word_count,
            "Flesch": round(ch.readability.flesch_reading_ease, 1) if ch.readability else "-",
            "TTR": round(ch.vocabulary.ttr, 3) if ch.vocabulary else "-",
            "Top palavras (TF-IDF)": ", ".join([kw for kw, _ in ch.tfidf_keywords[:4]]) if ch.tfidf_keywords else "",
        } for ch in metrics.chapters])
        st.dataframe(df_ch, use_container_width=True)

    # Top palavras
    if metrics.vocabulary and metrics.vocabulary.top_words:
        st.divider()
        st.subheader("🔤 Top 20 Palavras")
        df_w = pd.DataFrame(metrics.vocabulary.top_words[:20], columns=["Palavra", "Freq"])
        fig = px.bar(df_w, x="Freq", y="Palavra", orientation="h",
                     color="Freq", color_continuous_scale="Blues",
                     title="Palavras mais frequentes (sem stopwords)")
        fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Refazer análise"):
        del st.session_state["doc_metrics"]
        st.rerun()

    st.success("➡️ Continue para **Tópicos LDA**.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: TÓPICOS
# ══════════════════════════════════════════════════════════════════════════════
elif current == "topics":
    st.title("🗺️ Modelagem de Tópicos (LDA)")

    if "doc_structure" not in st.session_state:
        st.warning("⚠️ Faça o upload do PDF primeiro.")
        st.stop()

    structure = st.session_state["doc_structure"]
    lang = st.session_state.get("doc_lang", "pt")

    col1, col2, col3 = st.columns(3)
    with col1:
        auto_tune = st.checkbox("Detectar nº de tópicos automaticamente", value=True)
    with col2:
        num_topics = st.slider("Nº de tópicos (manual)", 3, 20, 8, disabled=auto_tune)
    with col3:
        passes = st.slider("Passagens (qualidade vs. velocidade)", 5, 20, 10)

    if "topic_result" not in st.session_state:
        if st.button("▶️ Executar LDA", type="primary", use_container_width=True):
            with st.spinner("Treinando modelo LDA..."):
                try:
                    from core.ingestion import iter_chapters
                    from core.topics import train_lda

                    chapters_list = list(iter_chapters(structure))
                    texts = [t for _, t in chapters_list]
                    titles = [ch.get("title", f"Cap.{i+1}") for i, (ch, _) in enumerate(chapters_list)]

                    if len(texts) < 3:
                        texts = [p.text for p in structure.pages if len(p.text.split()) > 40]
                        titles = [f"Pág. {p.page_number}" for p in structure.pages if len(p.text.split()) > 40]

                    prog = st.progress(0)
                    status = st.empty()
                    def cb(cur, tot, msg):
                        prog.progress(min(cur / max(tot, 1), 1.0))
                        status.text(msg)

                    result = train_lda(
                        texts, num_topics=None if auto_tune else num_topics,
                        lang=lang, passes=passes, auto_tune=auto_tune,
                        progress_callback=cb,
                    )
                    st.session_state["topic_result"] = result
                    st.session_state["topic_titles"] = titles
                    prog.progress(1.0)
                    status.text("✅ Concluído!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
                    st.exception(e)
        st.stop()

    import plotly.express as px
    import pandas as pd

    result = st.session_state["topic_result"]
    titles = st.session_state.get("topic_titles", [f"Cap.{i+1}" for i in range(len(result.chapter_topic_distribution))])

    COLORS = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
              "#DDA0DD","#98D8C8","#F7DC6F","#BB8FCE","#85C1E9",
              "#F0B27A","#82E0AA","#F1948A","#A9CCE3","#FAD7A0"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Tópicos", result.num_topics)
    c2.metric("Coerência c_v", f"{result.coherence_score:.3f}")
    c3.metric("Documentos", len(result.chapter_topic_distribution))

    st.divider()
    st.subheader("🏷️ Tópicos")
    ncols = min(3, result.num_topics)
    cols = st.columns(ncols)
    for i, topic in enumerate(result.topics):
        with cols[i % ncols]:
            c = COLORS[i % len(COLORS)]
            kws = " · ".join([kw for kw, _ in topic.keywords[:7]])
            st.markdown(f"""
            <div style="background:{c}22;border-left:4px solid {c};
                        padding:0.7rem;border-radius:6px;margin-bottom:0.6rem">
                <strong>T{i+1}: {topic.label}</strong><br>
                <small>{kws}</small><br>
                <small>Prevalência: {topic.prevalence:.1%}</small>
            </div>""", unsafe_allow_html=True)

    # Heatmap
    st.divider()
    st.subheader("🔥 Distribuição por Capítulo")
    t_labels = [f"T{i+1}:{t.label[:12]}" for i, t in enumerate(result.topics)]
    short_titles = [t[:22]+"…" if len(t)>22 else t for t in titles]
    df_h = pd.DataFrame(result.document_topic_matrix[:len(short_titles)],
                        index=short_titles, columns=t_labels)
    fig = px.imshow(df_h, color_continuous_scale="Blues", aspect="auto",
                    title="Proporção de cada tópico por capítulo")
    fig.update_layout(height=max(280, len(short_titles) * 26))
    st.plotly_chart(fig, use_container_width=True)

    # Tópico dominante
    st.subheader("📌 Tópico dominante por capítulo")
    df_dom = pd.DataFrame([{
        "Capítulo": titles[i] if i < len(titles) else f"Cap.{i+1}",
        "Tópico dominante": f"T{d['dominant_topic']+1}: {result.topics[d['dominant_topic']].label}",
        "Proporção": round(d["distribution"][d["dominant_topic"]], 3),
    } for i, d in enumerate(result.chapter_topic_distribution)])
    st.dataframe(df_dom, use_container_width=True)

    if st.button("🔄 Refazer com novos parâmetros"):
        del st.session_state["topic_result"]
        st.rerun()

    st.success("➡️ Continue para **Análise LLM** (opcional) ou **Exportar MAXQDA**.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: LLM
# ══════════════════════════════════════════════════════════════════════════════
elif current == "llm":
    st.title("🤖 Análise Qualitativa via LLM")
    st.markdown("Resumos automáticos e análise pedagógica. Requer chave de API configurada.")

    if "doc_structure" not in st.session_state:
        st.warning("⚠️ Faça o upload do PDF primeiro.")
        st.stop()

    structure = st.session_state["doc_structure"]

    # Detecta providers disponíveis
    from core.llm import LLMClient

    available = {}
    for prov in ["anthropic", "openai"]:
        client = LLMClient(provider=prov)
        ok, msg = client.is_available()
        available[prov] = (ok, msg, client)

    has_any = any(ok for ok, _, _ in available.values())

    if not has_any:
        st.error("Nenhum provedor LLM configurado.")
        st.markdown("""
        **Como configurar no Hugging Face Spaces:**
        1. Vá em **Settings** do seu Space
        2. Clique em **Repository secrets**
        3. Adicione `ANTHROPIC_API_KEY` ou `OPENAI_API_KEY`
        4. Recarregue o Space

        **Para uso local:** crie um arquivo `.env` com as chaves.
        """)
        st.info("💡 As análises quantitativas e de tópicos funcionam sem LLM!")
        st.stop()

    # Seleciona provider
    prov_opts = {prov: f"{'✅' if ok else '❌'} {prov.title()} — {msg}"
                 for prov, (ok, msg, _) in available.items() if ok}
    selected_prov = st.selectbox("Provedor", list(prov_opts.keys()),
                                 format_func=lambda k: prov_opts[k])
    client = available[selected_prov][2]

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        do_chapters = st.checkbox("Analisar capítulos", value=True)
        max_ch = st.slider("Máx. capítulos", 1, min(15, max(len(structure.chapters), 1)),
                           min(5, max(len(structure.chapters), 1)), disabled=not do_chapters)
    with col2:
        do_overview = st.checkbox("Análise geral do livro", value=True)

    if "llm_result" not in st.session_state:
        if st.button("▶️ Executar análise LLM", type="primary", use_container_width=True):
            with st.spinner("Analisando com LLM..."):
                try:
                    from core.ingestion import iter_chapters
                    from core.llm import analyze_chapter, analyze_document_overview

                    chapters_list = list(iter_chapters(structure))
                    results = {"chapters": [], "overview": None}
                    doc_metrics = st.session_state.get("doc_metrics")

                    total = (max_ch if do_chapters else 0) + (1 if do_overview else 0)
                    prog = st.progress(0)
                    status = st.empty()
                    step = 0

                    if do_chapters:
                        for i, (ch_info, ch_text) in enumerate(chapters_list[:max_ch]):
                            step += 1
                            prog.progress(step / max(total, 1))
                            title = ch_info.get("title", f"Capítulo {i+1}")
                            status.text(f"Analisando: {title}...")
                            try:
                                analysis = analyze_chapter(client, title, ch_text)
                                results["chapters"].append(analysis)
                            except Exception as e:
                                st.warning(f"Pulando '{title}': {e}")

                    if do_overview:
                        step += 1
                        prog.progress(step / max(total, 1))
                        status.text("Análise geral...")
                        om = {}
                        if doc_metrics:
                            om = {
                                "total_words": doc_metrics.total_words,
                                "flesch": doc_metrics.readability.flesch_reading_ease if doc_metrics.readability else "N/A",
                                "flesch_interp": doc_metrics.readability.interpretation if doc_metrics.readability else "",
                            }
                        try:
                            ov = analyze_document_overview(
                                client, structure.title, structure.author,
                                [ch.summary for ch in results["chapters"]], om)
                            results["overview"] = ov
                        except Exception as e:
                            st.warning(f"Erro na análise geral: {e}")

                    prog.progress(1.0)
                    status.text("✅ Concluído!")
                    st.session_state["llm_result"] = results
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
                    st.exception(e)
        st.stop()

    results = st.session_state["llm_result"]

    if results.get("overview"):
        ov = results["overview"]
        st.subheader("📖 Análise Geral")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Resumo**")
            st.markdown(ov.overall_summary)
            st.markdown("**Abordagem Pedagógica**")
            st.markdown(ov.pedagogical_approach)
        with col2:
            st.info(f"**Público-alvo:** {ov.target_audience}")
            st.markdown("**Temas:**")
            for t in ov.main_themes:
                st.markdown(f"• {t}")
        col1, col2 = st.columns(2)
        with col1:
            if ov.strengths:
                st.markdown("**✅ Pontos Fortes**")
                for s in ov.strengths: st.markdown(f"• {s}")
        with col2:
            if ov.suggestions:
                st.markdown("**💡 Sugestões**")
                for s in ov.suggestions: st.markdown(f"• {s}")

    if results.get("chapters"):
        st.divider()
        st.subheader(f"📚 Capítulos ({len(results['chapters'])})")
        for ch in results["chapters"]:
            with st.expander(f"📖 {ch.chapter_title}  ·  {ch.difficulty_level}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Resumo**")
                    st.markdown(ch.summary)
                    if ch.pedagogical_notes:
                        st.markdown("**Notas Pedagógicas**")
                        st.markdown(ch.pedagogical_notes)
                with col2:
                    if ch.learning_objectives:
                        st.markdown("**Objetivos:**")
                        for o in ch.learning_objectives: st.markdown(f"✓ {o}")
                    if ch.key_concepts:
                        st.markdown("**Conceitos:**")
                        st.markdown(" · ".join([f"`{c}`" for c in ch.key_concepts]))

    if st.button("🔄 Refazer análise LLM"):
        del st.session_state["llm_result"]
        st.rerun()

    st.success("➡️ Continue para **Exportar MAXQDA**.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: EXPORTAR
# ══════════════════════════════════════════════════════════════════════════════
elif current == "export":
    st.title("📤 Exportação para MAXQDA")

    if "doc_structure" not in st.session_state:
        st.warning("⚠️ Faça o upload do PDF primeiro.")
        st.stop()

    structure = st.session_state["doc_structure"]
    doc_metrics = st.session_state.get("doc_metrics")
    topic_result = st.session_state.get("topic_result")
    llm_result = st.session_state.get("llm_result")

    # Status
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 PDF", "✅ OK")
    c2.metric("📊 Quantitativo", "✅ OK" if doc_metrics else "⚪ Opcional")
    c3.metric("🗺️ Tópicos", f"✅ {topic_result.num_topics} tópicos" if topic_result else "⚪ Opcional")
    c4.metric("🤖 LLM", "✅ OK" if llm_result else "⚪ Opcional")

    st.divider()
    project_name = st.text_input("Nome do projeto MAXQDA", value=structure.title[:50])
    col1, col2 = st.columns(2)
    with col1:
        include_text = st.checkbox("Incluir texto completo", value=True)
        include_metrics = st.checkbox("Incluir variáveis (métricas)", value=True)
    with col2:
        include_lda = st.checkbox("Incluir códigos LDA", value=True, disabled=not topic_result)
        include_llm = st.checkbox("Incluir temas LLM", value=True, disabled=not llm_result)

    if st.button("🔨 Gerar .mx20", type="primary", use_container_width=True):
        with st.spinner("Gerando arquivo..."):
            try:
                from core.ingestion import iter_chapters
                from core.maxqda_export import (
                    export_to_maxqda,
                    build_chapters_for_export,
                    build_topics_for_export,
                )

                prog = st.progress(0)
                status = st.empty()
                def cb(cur, tot, msg):
                    prog.progress(cur / max(tot, 1))
                    status.text(msg)

                chapters_exp = build_chapters_for_export(structure, doc_metrics, topic_result)
                if not include_text:
                    for ch in chapters_exp:
                        ch["text"] = ch["text"][:800] + "\n[...]"
                if not include_metrics:
                    for ch in chapters_exp:
                        ch.update({"flesch": "", "ttr": "", "word_count": ""})

                topics_exp = build_topics_for_export(topic_result) if (topic_result and include_lda) else []

                doc_analysis = None
                if llm_result and include_llm and llm_result.get("overview"):
                    doc_analysis = llm_result["overview"]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mx20") as tmp:
                    out_path = tmp.name

                export_to_maxqda(
                    document_title=project_name,
                    chapters=chapters_exp,
                    topics=topics_exp,
                    output_path=out_path,
                    document_analysis=doc_analysis,
                    progress_callback=cb,
                )

                with open(out_path, "rb") as f:
                    bytes_data = f.read()
                os.unlink(out_path)

                st.session_state["mx20_bytes"] = bytes_data
                st.session_state["mx20_name"] = f"{project_name[:30].replace(' ','_')}.mx20"
                prog.progress(1.0)
                status.text("✅ Arquivo gerado!")
                st.rerun()

            except Exception as e:
                st.error(f"Erro: {e}")
                st.exception(e)

    if "mx20_bytes" in st.session_state:
        st.success("✅ Arquivo pronto!")
        kb = len(st.session_state["mx20_bytes"]) / 1024
        st.markdown(f"📦 **{st.session_state['mx20_name']}** · {kb:.1f} KB")

        st.download_button(
            "⬇️ Baixar arquivo .mx20",
            data=st.session_state["mx20_bytes"],
            file_name=st.session_state["mx20_name"],
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        st.subheader("📖 Como abrir no MAXQDA")
        st.markdown("""
        1. Abra **MAXQDA 2020** ou superior
        2. **Projeto → Abrir projeto** → selecione o arquivo `.mx20`
        3. O projeto abrirá com:
           - 📁 Documentos: um por capítulo
           - 🏷️ Códigos: tópicos LDA + categorias qualitativas pré-estruturadas
           - 📊 Variáveis: métricas quantitativas por capítulo
        4. Os códigos automáticos são **sugestões** — revise e ajuste!
        """)
