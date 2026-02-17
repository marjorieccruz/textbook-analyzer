"""
core/topics.py
──────────────
Modelagem de tópicos com LDA (Latent Dirichlet Allocation) via Gensim.
Inclui detecção automática do número ideal de tópicos via coerência.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk
from gensim import corpora, models
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Topic:
    id: int
    label: str                                    # Rótulo automático (top palavra)
    keywords: list[tuple[str, float]]             # [(palavra, peso)]
    prevalence: float                             # % de presença no corpus


@dataclass
class TopicModelResult:
    num_topics: int
    coherence_score: float
    topics: list[Topic]
    chapter_topic_distribution: list[dict]        # [{chapter, dominant_topic, distribution}]
    document_topic_matrix: list[list[float]]      # Matriz completa capítulo x tópico


# ─── Pré-processamento ────────────────────────────────────────────────────────

def preprocess_texts(
    texts: list[str],
    lang: str = "pt",
    min_word_len: int = 3,
) -> list[list[str]]:
    """
    Tokeniza e limpa textos para LDA.
    Remove stopwords, números e palavras muito curtas.
    """
    lang_map = {"pt": "portuguese", "en": "english"}
    stops = set(stopwords.words(lang_map.get(lang, "portuguese")))

    processed = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^a-záàâãéêíóôõúüçñ\s]", " ", text)
        tokens = text.split()
        tokens = [
            t for t in tokens
            if len(t) >= min_word_len and t not in stops and not t.isdigit()
        ]
        if tokens:
            processed.append(tokens)
        else:
            processed.append(["vazio"])  # Garante que nunca retornamos lista vazia

    return processed


# ─── Detecção automática do número de tópicos ────────────────────────────────

def find_optimal_topics(
    texts: list[list[str]],
    dictionary: corpora.Dictionary,
    corpus: list,
    min_topics: int = 3,
    max_topics: int = 20,
    step: int = 2,
    progress_callback=None,
) -> int:
    """
    Testa diferentes números de tópicos e retorna o que maximiza a coerência.
    Usa métrica c_v (mais confiável para textos em português).
    """
    best_score = -1.0
    best_k = min_topics
    results = []

    topic_range = range(min_topics, min(max_topics + 1, len(texts) + 1), step)

    for i, k in enumerate(topic_range):
        if progress_callback:
            progress_callback(i, len(topic_range), f"Testando {k} tópicos...")

        try:
            model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=5,
                alpha="auto",
                random_state=42,
            )
            coherence = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v",
            ).get_coherence()

            results.append((k, coherence))

            if coherence > best_score:
                best_score = coherence
                best_k = k
        except Exception:
            continue

    return best_k


# ─── Treinamento do modelo LDA ────────────────────────────────────────────────

def train_lda(
    chapter_texts: list[str],
    num_topics: int | None = None,
    lang: str = "pt",
    passes: int = 10,
    auto_tune: bool = True,
    progress_callback=None,
) -> TopicModelResult:
    """
    Treina modelo LDA e retorna distribuição de tópicos por capítulo.

    Args:
        chapter_texts: Lista de textos (um por capítulo).
        num_topics: Número de tópicos (None = detecção automática).
        lang: Idioma ('pt' ou 'en').
        passes: Número de passagens pelo corpus.
        auto_tune: Se True, busca número ideal de tópicos automaticamente.
        progress_callback: Função (current, total, msg).

    Returns:
        TopicModelResult com tópicos e distribuições.
    """
    if not chapter_texts:
        raise ValueError("Lista de textos vazia.")

    if progress_callback:
        progress_callback(0, 4, "Pré-processando textos...")

    processed = preprocess_texts(chapter_texts, lang=lang)

    # Dicionário e corpus
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=2, no_above=0.85, keep_n=5000)
    corpus = [dictionary.doc2bow(tokens) for tokens in processed]

    # Número de tópicos
    if num_topics is None:
        if auto_tune and len(chapter_texts) >= 5:
            if progress_callback:
                progress_callback(1, 4, "Detectando número ideal de tópicos...")
            num_topics = find_optimal_topics(
                processed, dictionary, corpus,
                max_topics=min(20, len(chapter_texts)),
                progress_callback=progress_callback,
            )
        else:
            num_topics = min(8, max(3, len(chapter_texts) // 2))

    if progress_callback:
        progress_callback(2, 4, f"Treinando LDA com {num_topics} tópicos...")

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    # Coerência
    try:
        coherence = CoherenceModel(
            model=lda_model,
            texts=processed,
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence()
    except Exception:
        coherence = 0.0

    if progress_callback:
        progress_callback(3, 4, "Extraindo tópicos e distribuições...")

    # Extrai tópicos
    topics = []
    total_docs = len(corpus)
    for topic_id in range(num_topics):
        keywords = lda_model.show_topic(topic_id, topn=15)
        label = keywords[0][0] if keywords else f"Tópico {topic_id + 1}"

        # Prevalência: % de documentos onde esse tópico é dominante
        dominant_count = sum(
            1 for doc_bow in corpus
            if _dominant_topic(lda_model.get_document_topics(doc_bow)) == topic_id
        )
        prevalence = round(dominant_count / max(total_docs, 1), 3)

        topics.append(Topic(
            id=topic_id,
            label=label,
            keywords=[(word, round(weight, 4)) for word, weight in keywords],
            prevalence=prevalence,
        ))

    # Distribuição por capítulo
    doc_topic_matrix = []
    chapter_distributions = []

    for i, doc_bow in enumerate(corpus):
        topic_dist = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        dist_vector = [round(topic_dist.get(t, 0.0), 4) for t in range(num_topics)]
        doc_topic_matrix.append(dist_vector)

        dominant = max(range(num_topics), key=lambda t: dist_vector[t])
        chapter_distributions.append({
            "chapter_index": i,
            "dominant_topic": dominant,
            "dominant_topic_label": topics[dominant].label if topics else "",
            "distribution": dist_vector,
        })

    if progress_callback:
        progress_callback(4, 4, "Modelagem concluída!")

    return TopicModelResult(
        num_topics=num_topics,
        coherence_score=round(coherence, 4),
        topics=topics,
        chapter_topic_distribution=chapter_distributions,
        document_topic_matrix=doc_topic_matrix,
    )


def _dominant_topic(topic_dist) -> int:
    """Retorna o índice do tópico dominante em uma distribuição."""
    if not topic_dist:
        return 0
    return max(topic_dist, key=lambda x: x[1])[0]


# ─── Visualização ─────────────────────────────────────────────────────────────

def prepare_pyldavis(
    lda_model,
    corpus,
    dictionary,
) -> str | None:
    """
    Prepara HTML interativo do pyLDAvis.
    Retorna string HTML ou None se não disponível.
    """
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis

        pyLDAvis.enable_notebook()
        vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
        return pyLDAvis.prepared_data_to_html(vis)
    except Exception:
        return None
