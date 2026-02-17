"""
core/ingestion.py
─────────────────
Módulo de ingestão de PDFs.
Extrai texto estruturado, imagens e metadados usando PyMuPDF (fitz) + pdfplumber.
Suporta PDFs nativos e escaneados (fallback OCR via Tesseract).
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    page_number: int
    text: str
    images: list[dict] = field(default_factory=list)   # {index, path, caption}
    tables: list[list] = field(default_factory=list)
    is_chapter_start: bool = False
    chapter_title: str | None = None


@dataclass
class DocumentStructure:
    title: str
    author: str
    total_pages: int
    chapters: list[dict] = field(default_factory=list)  # {title, start_page, end_page}
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Salva estrutura em JSON para reuso posterior."""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "DocumentStructure":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        doc = cls(**{k: v for k, v in data.items() if k != "pages"})
        doc.pages = [PageContent(**p) for p in data.get("pages", [])]
        return doc


# ─── Chapter Detection ────────────────────────────────────────────────────────

CHAPTER_PATTERNS = [
    r"^cap[íi]tulo\s+\d+",
    r"^chapter\s+\d+",
    r"^\d+\s*[\.–—]\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]",
    r"^parte\s+\d+",
    r"^unit\s+\d+",
    r"^se[çc][aã]o\s+\d+",
]

def _detect_chapter(text: str) -> tuple[bool, str | None]:
    """Verifica se o início da página marca um novo capítulo."""
    first_lines = text.strip().split("\n")[:5]
    for line in first_lines:
        line = line.strip()
        for pattern in CHAPTER_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True, line
    return False, None


# ─── Image Extraction ─────────────────────────────────────────────────────────

def _extract_images_from_page(
    page: fitz.Page,
    doc: fitz.Document,
    output_dir: Path,
    page_num: int,
) -> list[dict]:
    """Extrai imagens de uma página e salva em disco."""
    images = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Ignora imagens muito pequenas (ícones, marcadores)
            img = Image.open(io.BytesIO(image_bytes))
            if img.width < 80 or img.height < 80:
                continue

            filename = f"page{page_num:04d}_img{img_index:02d}.{image_ext}"
            save_path = output_dir / filename
            save_path.write_bytes(image_bytes)

            images.append({
                "index": img_index,
                "path": str(save_path),
                "width": img.width,
                "height": img.height,
                "ext": image_ext,
                "caption": None,  # preenchido depois pelo módulo LLM
            })
        except Exception:
            continue

    return images


# ─── Table Extraction ─────────────────────────────────────────────────────────

def _extract_tables_from_page(pdf_path: str | Path, page_num: int) -> list[list]:
    """Extrai tabelas de uma página usando pdfplumber (mais preciso que PyMuPDF)."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 < len(pdf.pages):
                page = pdf.pages[page_num - 1]
                raw_tables = page.extract_tables()
                for t in raw_tables:
                    if t and len(t) > 1:
                        tables.append(t)
    except Exception:
        pass
    return tables


# ─── Main Ingestion Function ──────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    extract_images: bool = True,
    extract_tables: bool = True,
    progress_callback=None,
) -> DocumentStructure:
    """
    Extrai conteúdo completo de um PDF.

    Args:
        pdf_path: Caminho para o arquivo PDF.
        output_dir: Diretório para salvar imagens extraídas.
        extract_images: Se True, extrai e salva imagens.
        extract_tables: Se True, extrai tabelas com pdfplumber.
        progress_callback: Função (current, total, msg) para atualizar progresso.

    Returns:
        DocumentStructure com todo o conteúdo estruturado.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    # Diretório para imagens
    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Metadados
    meta = doc.metadata or {}
    structure = DocumentStructure(
        title=meta.get("title") or pdf_path.stem,
        author=meta.get("author") or "Desconhecido",
        total_pages=total_pages,
        metadata={
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "creation_date": meta.get("creationDate", ""),
            "file_size_mb": round(pdf_path.stat().st_size / 1_048_576, 2),
            "filename": pdf_path.name,
        },
    )

    # Índice/TOC do PDF (se disponível)
    toc = doc.get_toc()
    if toc:
        for level, title, page in toc:
            if level == 1:
                structure.chapters.append({
                    "title": title,
                    "start_page": page,
                    "end_page": None,
                })
        # Preenche end_page
        for i, ch in enumerate(structure.chapters[:-1]):
            ch["end_page"] = structure.chapters[i + 1]["start_page"] - 1
        if structure.chapters:
            structure.chapters[-1]["end_page"] = total_pages

    current_chapter = None

    # Processa cada página
    for page_num in range(1, total_pages + 1):
        if progress_callback:
            progress_callback(page_num, total_pages, f"Processando página {page_num}/{total_pages}...")

        page = doc[page_num - 1]

        # Extrai texto
        text = page.get_text("text")

        # Fallback: se página sem texto, tenta OCR
        if not text.strip() and len(text) < 50:
            text = _ocr_page(page)

        # Detecta capítulo se não temos TOC
        is_chapter, chapter_title = _detect_chapter(text)
        if is_chapter and not toc:
            current_chapter = chapter_title
            structure.chapters.append({
                "title": chapter_title,
                "start_page": page_num,
                "end_page": None,
            })
            if len(structure.chapters) > 1:
                structure.chapters[-2]["end_page"] = page_num - 1

        page_content = PageContent(
            page_number=page_num,
            text=text,
            is_chapter_start=is_chapter,
            chapter_title=current_chapter,
        )

        if extract_images:
            page_content.images = _extract_images_from_page(
                page, doc, output_dir, page_num
            )

        if extract_tables:
            page_content.tables = _extract_tables_from_page(pdf_path, page_num)

        structure.pages.append(page_content)

    doc.close()

    # Fecha capítulos sem end_page
    if structure.chapters and structure.chapters[-1]["end_page"] is None:
        structure.chapters[-1]["end_page"] = total_pages

    return structure


def _ocr_page(page: fitz.Page) -> str:
    """OCR de uma página usando Tesseract como fallback."""
    try:
        import pytesseract
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang="por+eng")
    except Exception:
        return ""


# ─── Iterador por capítulo ────────────────────────────────────────────────────

def iter_chapters(structure: DocumentStructure) -> Iterator[tuple[dict, str]]:
    """
    Itera sobre capítulos, retornando (chapter_info, full_text).
    Útil para processar livros grandes em chunks.
    """
    if not structure.chapters:
        # Sem capítulos detectados: trata o livro todo como um único bloco
        full_text = "\n\n".join(p.text for p in structure.pages)
        yield {"title": structure.title, "start_page": 1, "end_page": structure.total_pages}, full_text
        return

    for chapter in structure.chapters:
        pages_in_chapter = [
            p for p in structure.pages
            if chapter["start_page"] <= p.page_number <= (chapter["end_page"] or structure.total_pages)
        ]
        chapter_text = "\n\n".join(p.text for p in pages_in_chapter)
        yield chapter, chapter_text


def get_full_text(structure: DocumentStructure) -> str:
    """Retorna texto completo do documento."""
    return "\n\n".join(p.text for p in structure.pages)
