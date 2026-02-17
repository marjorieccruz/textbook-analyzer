"""
core/quantitative.py
─────────────────────
Análise quantitativa de texto: legibilidade, vocabulário, frequência de termos.
Suporta português e inglês.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import nltk
import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

# Garante downloads necessários do NLTK
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords

# ─── Modelos spaCy (lazy loading) ────────────────────────────────────────────

_NLP_MODELS: dict[str, spacy.Language] = {}

def _get_nlp(lang: str = "pt") -> spacy.Language:
    if lang not in _NLP_MODELS:
        model_name = "pt_core_news_sm" if lang == "pt" else "en_core_web_sm"
        try:
            _NLP_MODELS[lang] = spacy.load(model_name, disable=["parser", "ner"])
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            _NLP_MODELS[lang] = spacy.load(model_name, disable=["parser", "ner"])
    return _NLP_MODELS[lang]


def _get_stopwords(lang: str = "pt") -> set[str]:
    lang_map = {"pt": "portuguese", "en": "english"}
    return set(stopwords.words(lang_map.get(lang, "portuguese")))


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ReadabilityMetrics:
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    avg_sentence_length: float
    avg_word_length: float
    interpretation: str = ""

    def __post_init__(self):
        score = self.flesch_reading_ease
        if score >= 90:
            self.interpretation = "Muito fácil (Ensino Fundamental I)"
        elif score >= 70:
            self.interpretation = "Fácil (Ensino Fundamental II)"
        elif score >= 60:
            self.interpretation = "Médio (Ensino Médio)"
        elif score >= 50:
            self.interpretation = "Razoavelmente difícil (Ensino Superior)"
        elif score >= 30:
            self.interpretation = "Difícil (Graduação/Pós)"
        else:
            self.interpretation = "Muito difícil (Especialistas)"


@dataclass
class VocabularyMetrics:
    total_tokens: int
    unique_tokens: int
    ttr: float                          # Type-Token Ratio
    hapax_legomena: int                 # Palavras que aparecem só 1x
    hapax_ratio: float
    lexical_density: float              # % palavras de conteúdo
    top_words: list[tuple[str, int]] = field(default_factory=list)
    top_bigrams: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class ChapterMetrics:
    chapter_title: str
    page_start: int
    page_end: int
    word_count: int
    char_count: int
    sentence_count: int
    readability: ReadabilityMetrics | None = None
    vocabulary: VocabularyMetrics | None = None
    tfidf_keywords: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class DocumentMetrics:
    total_words: int
    total_sentences: int
    total_paragraphs: int
    total_images: int
    total_tables: int
    content_distribution: dict = field(default_factory=dict)  # % texto vs imagem etc.
    readability: ReadabilityMetrics | None = None
    vocabulary: VocabularyMetrics | None = None
    chapters: list[ChapterMetrics] = field(default_factory=list)
    complexity_progression: list[float] = field(default_factory=list)  # Flesch por capítulo


# ─── Funções de análise ───────────────────────────────────────────────────────

def compute_readability(text: str, lang: str = "pt") -> ReadabilityMetrics:
    """Calcula métricas de legibilidade."""
    textstat.set_lang(lang)
    clean = _clean_text(text)

    if len(clean.split()) < 50:
        # Texto muito curto para análise confiável
        return ReadabilityMetrics(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            gunning_fog=0.0,
            avg_sentence_length=0.0,
            avg_word_length=0.0,
            interpretation="Texto insuficiente para análise",
        )

    sentences = _split_sentences(clean)
    words = clean.split()

    avg_sentence_len = len(words) / max(len(sentences), 1)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

    return ReadabilityMetrics(
        flesch_reading_ease=textstat.flesch_reading_ease(clean),
        flesch_kincaid_grade=textstat.flesch_kincaid_grade(clean),
        gunning_fog=textstat.gunning_fog(clean),
        avg_sentence_length=round(avg_sentence_len, 2),
        avg_word_length=round(avg_word_len, 2),
    )


def compute_vocabulary(text: str, lang: str = "pt") -> VocabularyMetrics:
    """Calcula métricas de vocabulário e riqueza lexical."""
    nlp = _get_nlp(lang)
    stops = _get_stopwords(lang)
    clean = _clean_text(text)

    # Tokeniza com spaCy (lemmatização)
    doc = nlp(clean[:1_000_000])  # Limite de segurança para textos muito grandes

    all_tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and len(token.text) > 2
    ]

    content_tokens = [t for t in all_tokens if t not in stops]
    total = len(all_tokens)
    unique = len(set(all_tokens))

    freq = Counter(content_tokens)
    hapax = sum(1 for c in freq.values() if c == 1)

    # Bigramas (ignorando stopwords)
    bigrams = Counter(
        f"{content_tokens[i]} {content_tokens[i+1]}"
        for i in range(len(content_tokens) - 1)
    )

    return VocabularyMetrics(
        total_tokens=total,
        unique_tokens=unique,
        ttr=round(unique / max(total, 1), 4),
        hapax_legomena=hapax,
        hapax_ratio=round(hapax / max(unique, 1), 4),
        lexical_density=round(len(content_tokens) / max(total, 1), 4),
        top_words=freq.most_common(30),
        top_bigrams=bigrams.most_common(20),
    )


def compute_tfidf_keywords(
    chapter_texts: list[str],
    top_n: int = 10,
    lang: str = "pt",
) -> list[list[tuple[str, float]]]:
    """
    Calcula palavras-chave por capítulo via TF-IDF.

    Returns:
        Lista de listas [(palavra, score)] para cada capítulo.
    """
    stops = list(_get_stopwords(lang))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=stops,
        min_df=1,
        ngram_range=(1, 2),
        token_pattern=r"\b[a-záàâãéêíóôõúüçñ]{3,}\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(chapter_texts)
        feature_names = vectorizer.get_feature_names_out()

        result = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray()[0]
            top_indices = row.argsort()[-top_n:][::-1]
            keywords = [(feature_names[j], round(float(row[j]), 4)) for j in top_indices if row[j] > 0]
            result.append(keywords)
        return result
    except Exception:
        return [[] for _ in chapter_texts]


def analyze_document(
    structure,  # DocumentStructure de ingestion.py
    lang: str = "pt",
    progress_callback=None,
) -> DocumentMetrics:
    """
    Análise quantitativa completa de um documento.

    Args:
        structure: DocumentStructure retornada por ingestion.ingest_pdf()
        lang: Idioma do documento ('pt' ou 'en')
        progress_callback: Função (current, total, msg)

    Returns:
        DocumentMetrics com todas as métricas calculadas.
    """
    from core.ingestion import get_full_text, iter_chapters

    full_text = get_full_text(structure)
    total_images = sum(len(p.images) for p in structure.pages)
    total_tables = sum(len(p.tables) for p in structure.pages)
    total_words = len(full_text.split())
    total_sentences = len(_split_sentences(full_text))
    total_paragraphs = full_text.count("\n\n") + 1

    if progress_callback:
        progress_callback(1, 4, "Calculando legibilidade geral...")

    readability = compute_readability(full_text, lang)

    if progress_callback:
        progress_callback(2, 4, "Analisando vocabulário...")

    vocabulary = compute_vocabulary(full_text, lang)

    if progress_callback:
        progress_callback(3, 4, "Analisando capítulos...")

    chapters_list = list(iter_chapters(structure))
    chapter_texts = [text for _, text in chapters_list]
    tfidf_keywords = compute_tfidf_keywords(chapter_texts, lang=lang)

    chapter_metrics = []
    complexity_progression = []

    for i, (chapter_info, chapter_text) in enumerate(chapters_list):
        ch_words = chapter_text.split()
        ch_sentences = _split_sentences(chapter_text)

        ch_readability = compute_readability(chapter_text, lang) if len(ch_words) > 50 else None
        ch_vocabulary = compute_vocabulary(chapter_text, lang) if len(ch_words) > 50 else None

        chapter_metrics.append(ChapterMetrics(
            chapter_title=chapter_info.get("title", f"Capítulo {i+1}"),
            page_start=chapter_info.get("start_page", 0),
            page_end=chapter_info.get("end_page", 0),
            word_count=len(ch_words),
            char_count=len(chapter_text),
            sentence_count=len(ch_sentences),
            readability=ch_readability,
            vocabulary=ch_vocabulary,
            tfidf_keywords=tfidf_keywords[i] if i < len(tfidf_keywords) else [],
        ))

        if ch_readability:
            complexity_progression.append(ch_readability.flesch_reading_ease)

    if progress_callback:
        progress_callback(4, 4, "Finalizando...")

    return DocumentMetrics(
        total_words=total_words,
        total_sentences=total_sentences,
        total_paragraphs=total_paragraphs,
        total_images=total_images,
        total_tables=total_tables,
        content_distribution={
            "text_pages": sum(1 for p in structure.pages if p.text.strip()),
            "image_count": total_images,
            "table_count": total_tables,
            "avg_words_per_page": round(total_words / max(structure.total_pages, 1)),
        },
        readability=readability,
        vocabulary=vocabulary,
        chapters=chapter_metrics,
        complexity_progression=complexity_progression,
    )


# ─── Utilitários ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove ruído de extração PDF."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", " ", text)
    text = re.sub(r"\b\d+\b", "", text)  # Remove números isolados
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Divide texto em sentenças."""
    try:
        from nltk.tokenize import sent_tokenize
        return [s for s in sent_tokenize(text) if len(s.split()) > 3]
    except Exception:
        return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.split()) > 3]
