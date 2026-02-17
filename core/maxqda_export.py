"""
core/maxqda_export.py
──────────────────────
Exportação para formato MAXQDA (.mx20).
O arquivo .mx20 é um ZIP contendo XMLs estruturados.
Compatível com MAXQDA 2020, 2022 e 2024.
"""

from __future__ import annotations

import zipfile
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass
class MAXQDASegment:
    """Representa um segmento de texto a ser codificado."""
    text: str
    code: str
    color: str = "#FFFF00"
    memo: str = ""
    page_start: int = 0
    page_end: int = 0


# ─── Paleta de cores para tópicos ────────────────────────────────────────────

TOPIC_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#85C1E9", "#A9CCE3",
    "#FAD7A0", "#A9DFBF", "#D7BDE2", "#AED6F1", "#A2D9CE",
]


# ─── XML Builders ─────────────────────────────────────────────────────────────

def _pretty_xml(element: ET.Element) -> str:
    """Retorna XML formatado com indentação."""
    rough = ET.tostring(element, encoding="unicode", xml_declaration=False)
    reparsed = minidom.parseString(f'<?xml version="1.0" encoding="UTF-8"?>{rough}')
    return reparsed.toprettyxml(indent="  ", encoding=None)


def _build_project_xml(
    project_name: str,
    created_at: str,
) -> str:
    """Constrói o XML de configuração do projeto MAXQDA."""
    root = ET.Element("MaxQDAProject")
    root.set("Version", "20.0")
    root.set("CreationDateTime", created_at)
    root.set("ProjectName", project_name)

    settings = ET.SubElement(root, "Settings")
    ET.SubElement(settings, "Language").text = "pt-BR"
    ET.SubElement(settings, "CreatedBy").text = "Textbook Analyzer"
    ET.SubElement(settings, "Description").text = (
        "Análise gerada automaticamente pela ferramenta Textbook Analyzer. "
        "Os códigos automáticos são sugestões baseadas em LDA — revise e ajuste conforme necessário."
    )

    return _pretty_xml(root)


def _build_documents_xml(
    document_title: str,
    chapters: list[dict],
    created_at: str,
) -> str:
    """
    Constrói o XML de documentos.
    Cada capítulo vira um documento separado no MAXQDA.
    """
    root = ET.Element("Documents")
    root.set("Version", "20.0")

    doc_group = ET.SubElement(root, "DocumentGroup")
    doc_group.set("Name", document_title)
    doc_group.set("Color", "#1f6feb")
    doc_group.set("CreationDateTime", created_at)

    for i, chapter in enumerate(chapters):
        doc = ET.SubElement(doc_group, "TextDocument")
        doc.set("Name", chapter.get("title", f"Capítulo {i+1}"))
        doc.set("CreationDateTime", created_at)
        doc.set("DocumentNumber", str(i + 1))
        doc.set("Color", "#FFFFFF")

        # Texto do capítulo
        text_elem = ET.SubElement(doc, "TextContent")
        text_elem.text = chapter.get("text", "")[:500_000]  # Limite de segurança

        # Variáveis de documento (métricas quantitativas)
        vars_elem = ET.SubElement(doc, "DocumentVariables")
        variables = {
            "word_count": str(chapter.get("word_count", 0)),
            "flesch_score": str(chapter.get("flesch", "")),
            "flesch_interpretation": chapter.get("flesch_interpretation", ""),
            "dominant_topic": chapter.get("dominant_topic", ""),
            "ttr": str(chapter.get("ttr", "")),
            "page_start": str(chapter.get("page_start", "")),
            "page_end": str(chapter.get("page_end", "")),
        }
        for var_name, var_value in variables.items():
            var_elem = ET.SubElement(vars_elem, "Variable")
            var_elem.set("Name", var_name)
            var_elem.set("Value", var_value)

    return _pretty_xml(root)


def _build_codes_xml(
    topics: list[dict],
    auto_codes: list[str] | None = None,
    created_at: str = "",
) -> str:
    """
    Constrói a árvore de códigos (codebook) do MAXQDA.

    Estrutura:
    📁 ANÁLISE AUTOMÁTICA (Textbook Analyzer)
      📁 Tópicos LDA
        🏷️ Tópico 1: [palavra-chave]
        🏷️ Tópico 2: [palavra-chave]
      📁 Análise Qualitativa
        🏷️ [códigos sugeridos pelo LLM]
      📁 Legibilidade
        🏷️ Muito Fácil
        🏷️ Médio
        🏷️ Difícil
    """
    root = ET.Element("Codes")
    root.set("Version", "20.0")

    # Grupo raiz
    root_group = ET.SubElement(root, "Code")
    root_group.set("Name", "ANÁLISE AUTOMÁTICA (Textbook Analyzer)")
    root_group.set("Color", "#1f6feb")
    root_group.set("CreationDateTime", created_at)
    root_group.set("IsCoded", "false")
    root_group.set("IsGroupCode", "true")
    root_group.set("Memo", "Códigos gerados automaticamente. Revise e ajuste conforme sua análise.")

    # Sub-grupo: Tópicos LDA
    topics_group = ET.SubElement(root_group, "Code")
    topics_group.set("Name", "Tópicos LDA")
    topics_group.set("Color", "#4ECDC4")
    topics_group.set("IsGroupCode", "true")
    topics_group.set("Memo", "Tópicos identificados automaticamente por modelagem LDA (Latent Dirichlet Allocation)")

    for i, topic in enumerate(topics):
        color = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        keywords = topic.get("keywords", [])
        top_words = ", ".join(kw[0] if isinstance(kw, (list, tuple)) else kw for kw in keywords[:5])

        code = ET.SubElement(topics_group, "Code")
        code.set("Name", f"T{i+1}: {topic.get('label', top_words[:30])}")
        code.set("Color", color)
        code.set("IsGroupCode", "false")
        code.set("Memo", f"Palavras-chave: {top_words}\nPrevalência: {topic.get('prevalence', 0):.1%}")

    # Sub-grupo: Análise Qualitativa
    qual_group = ET.SubElement(root_group, "Code")
    qual_group.set("Name", "Análise Qualitativa")
    qual_group.set("Color", "#96CEB4")
    qual_group.set("IsGroupCode", "true")
    qual_group.set("Memo", "Códigos para análise manual — adicione aqui suas próprias categorias")

    default_qual_codes = auto_codes or [
        "Definição de conceito",
        "Exemplo prático",
        "Exercício/atividade",
        "Referência bibliográfica",
        "Destaque do autor",
        "Passagem problemática",
        "Para revisão",
    ]
    qual_colors = ["#F7DC6F", "#DDA0DD", "#98D8C8", "#F0B27A", "#85C1E9", "#F1948A", "#A9CCE3"]

    for i, code_name in enumerate(default_qual_codes):
        code = ET.SubElement(qual_group, "Code")
        code.set("Name", code_name)
        code.set("Color", qual_colors[i % len(qual_colors)])
        code.set("IsGroupCode", "false")

    # Sub-grupo: Legibilidade
    leg_group = ET.SubElement(root_group, "Code")
    leg_group.set("Name", "Legibilidade")
    leg_group.set("Color", "#BB8FCE")
    leg_group.set("IsGroupCode", "true")

    for level, color in [
        ("Muito Fácil (Flesch 90-100)", "#2ECC71"),
        ("Fácil (Flesch 70-89)", "#82E0AA"),
        ("Médio (Flesch 50-69)", "#F7DC6F"),
        ("Difícil (Flesch 30-49)", "#F0B27A"),
        ("Muito Difícil (Flesch 0-29)", "#F1948A"),
    ]:
        code = ET.SubElement(leg_group, "Code")
        code.set("Name", level)
        code.set("Color", color)
        code.set("IsGroupCode", "false")

    return _pretty_xml(root)


def _build_variables_xml(created_at: str) -> str:
    """Define as variáveis de documento disponíveis no MAXQDA."""
    root = ET.Element("Variables")
    root.set("Version", "20.0")

    variable_defs = [
        ("word_count", "Integer", "Número total de palavras"),
        ("flesch_score", "Float", "Índice Flesch de legibilidade"),
        ("flesch_interpretation", "Text", "Interpretação do índice Flesch"),
        ("dominant_topic", "Text", "Tópico LDA dominante"),
        ("ttr", "Float", "Type-Token Ratio (riqueza vocabular)"),
        ("page_start", "Integer", "Página inicial"),
        ("page_end", "Integer", "Página final"),
    ]

    for name, var_type, description in variable_defs:
        var = ET.SubElement(root, "Variable")
        var.set("Name", name)
        var.set("Type", var_type)
        var.set("Description", description)
        var.set("CreationDateTime", created_at)

    return _pretty_xml(root)


# ─── Exportação principal ─────────────────────────────────────────────────────

def export_to_maxqda(
    document_title: str,
    chapters: list[dict],
    topics: list[dict],
    output_path: str | Path,
    document_analysis=None,
    progress_callback=None,
) -> Path:
    """
    Gera arquivo .mx20 compatível com MAXQDA.

    Args:
        document_title: Título do livro.
        chapters: Lista de dicts com {title, text, word_count, flesch, ttr, dominant_topic, page_start, page_end}.
        topics: Lista de dicts com {id, label, keywords, prevalence} (saída do módulo topics.py).
        output_path: Caminho de saída do arquivo .mx20.
        document_analysis: Análise qualitativa opcional (saída do módulo llm.py).
        progress_callback: Função (current, total, msg).

    Returns:
        Path do arquivo gerado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    project_name = document_title[:60]

    if progress_callback:
        progress_callback(1, 4, "Construindo estrutura do projeto...")

    project_xml = _build_project_xml(project_name, created_at)
    docs_xml = _build_documents_xml(document_title, chapters, created_at)

    if progress_callback:
        progress_callback(2, 4, "Gerando árvore de códigos...")

    # Extrai sugestões de códigos da análise LLM se disponível
    auto_codes = None
    if document_analysis and hasattr(document_analysis, "main_themes"):
        auto_codes = document_analysis.main_themes[:10]

    codes_xml = _build_codes_xml(topics, auto_codes=auto_codes, created_at=created_at)
    variables_xml = _build_variables_xml(created_at)

    if progress_callback:
        progress_callback(3, 4, "Empacotando arquivo .mx20...")

    # Gera sumário em JSON para referência
    summary = {
        "generated_by": "Textbook Analyzer",
        "generated_at": created_at,
        "document": document_title,
        "chapters_count": len(chapters),
        "topics_count": len(topics),
        "topics": [
            {
                "id": t.get("id", i),
                "label": t.get("label", ""),
                "keywords": t.get("keywords", [])[:5],
            }
            for i, t in enumerate(topics)
        ],
        "instructions": {
            "pt": "Abra este arquivo no MAXQDA 2020 ou superior. Os códigos automáticos são sugestões — ajuste conforme sua análise.",
            "en": "Open this file in MAXQDA 2020 or later. Automatic codes are suggestions — adjust according to your analysis.",
        },
    }

    # Empacota tudo em ZIP com extensão .mx20
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.xml", project_xml)
        zf.writestr("documents.xml", docs_xml)
        zf.writestr("codes.xml", codes_xml)
        zf.writestr("variables.xml", variables_xml)
        zf.writestr("textbook_analyzer_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    if progress_callback:
        progress_callback(4, 4, "Exportação concluída!")

    return output_path


# ─── Helpers de conversão ─────────────────────────────────────────────────────

def build_chapters_for_export(
    structure,        # DocumentStructure
    doc_metrics,      # DocumentMetrics
    topic_result,     # TopicModelResult
) -> list[dict]:
    """
    Converte as estruturas internas para o formato esperado por export_to_maxqda().
    """
    from core.ingestion import iter_chapters

    chapters_export = []
    chapters_iter = list(iter_chapters(structure))

    for i, (ch_info, ch_text) in enumerate(chapters_iter):
        # Métricas do capítulo
        ch_metrics = None
        if doc_metrics and i < len(doc_metrics.chapters):
            ch_metrics = doc_metrics.chapters[i]

        # Tópico dominante
        dominant_topic = ""
        if topic_result and i < len(topic_result.chapter_topic_distribution):
            dist = topic_result.chapter_topic_distribution[i]
            dominant_id = dist.get("dominant_topic", 0)
            if dominant_id < len(topic_result.topics):
                dominant_topic = topic_result.topics[dominant_id].label

        chapters_export.append({
            "title": ch_info.get("title", f"Capítulo {i+1}"),
            "text": ch_text,
            "page_start": ch_info.get("start_page", 0),
            "page_end": ch_info.get("end_page", 0),
            "word_count": ch_metrics.word_count if ch_metrics else len(ch_text.split()),
            "flesch": ch_metrics.readability.flesch_reading_ease if (ch_metrics and ch_metrics.readability) else "",
            "flesch_interpretation": ch_metrics.readability.interpretation if (ch_metrics and ch_metrics.readability) else "",
            "ttr": round(ch_metrics.vocabulary.ttr, 4) if (ch_metrics and ch_metrics.vocabulary) else "",
            "dominant_topic": dominant_topic,
        })

    return chapters_export


def build_topics_for_export(topic_result) -> list[dict]:
    """Converte TopicModelResult para o formato de export."""
    if not topic_result:
        return []
    return [
        {
            "id": t.id,
            "label": t.label,
            "keywords": t.keywords,
            "prevalence": t.prevalence,
        }
        for t in topic_result.topics
    ]
