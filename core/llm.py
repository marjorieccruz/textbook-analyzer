"""
core/llm.py
───────────
Integração com LLMs para análise qualitativa.
Suporta: Ollama (local), Anthropic Claude, OpenAI GPT.
Padrão: Ollama (dados nunca saem do computador local).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ChapterAnalysis:
    chapter_title: str
    summary: str
    learning_objectives: list[str]
    key_concepts: list[str]
    pedagogical_notes: str
    difficulty_level: str   # Iniciante / Intermediário / Avançado


@dataclass
class DocumentAnalysis:
    overall_summary: str
    target_audience: str
    main_themes: list[str]
    pedagogical_approach: str
    strengths: list[str]
    suggestions: list[str]
    chapters: list[ChapterAnalysis] = field(default_factory=list)


@dataclass
class ImageAnalysis:
    page_number: int
    image_path: str
    description: str
    image_type: str    # Gráfico / Diagrama / Fotografia / Tabela / Outro
    pedagogical_role: str


# ─── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Cliente unificado para diferentes provedores LLM.
    Troca de provedor apenas com mudança de variável de ambiente.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.provider = LLMProvider(
            provider or os.getenv("LLM_PROVIDER", "ollama")
        )
        self.model = model or self._default_model()

    def _default_model(self) -> str:
        defaults = {
            LLMProvider.OLLAMA: os.getenv("OLLAMA_MODEL", "llama3.2"),
            LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
            LLMProvider.OPENAI: os.getenv("OPENAI_MODEL", "gpt-4o"),
        }
        return defaults[self.provider]

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        """Envia prompt e retorna resposta como string."""
        if self.provider == LLMProvider.OLLAMA:
            return self._ollama(prompt, system, max_tokens)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._anthropic(prompt, system, max_tokens)
        elif self.provider == LLMProvider.OPENAI:
            return self._openai(prompt, system, max_tokens)
        raise ValueError(f"Provedor desconhecido: {self.provider}")

    def is_available(self) -> tuple[bool, str]:
        """Verifica se o provedor está configurado e acessível."""
        try:
            if self.provider == LLMProvider.OLLAMA:
                import requests
                base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                r = requests.get(f"{base}/api/tags", timeout=3)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if not any(self.model.split(":")[0] in m for m in models):
                        return False, f"Modelo '{self.model}' não encontrado. Execute: ollama pull {self.model}"
                    return True, "OK"
                return False, "Ollama não está rodando. Execute: ollama serve"

            elif self.provider == LLMProvider.ANTHROPIC:
                if not os.getenv("ANTHROPIC_API_KEY"):
                    return False, "ANTHROPIC_API_KEY não configurada no .env"
                return True, "OK"

            elif self.provider == LLMProvider.OPENAI:
                if not os.getenv("OPENAI_API_KEY"):
                    return False, "OPENAI_API_KEY não configurada no .env"
                return True, "OK"

        except Exception as e:
            return False, str(e)

        return False, "Provedor desconhecido"

    # ── Implementações por provedor ──────────────────────────────────────────

    def _ollama(self, prompt: str, system: str, max_tokens: int) -> str:
        import requests
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{base}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = client.messages.create(**kwargs)
        return msg.content[0].text.strip()

    def _openai(self, prompt: str, system: str, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


# ─── Análises específicas ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um especialista em análise pedagógica e linguística de materiais didáticos.
Analise o texto fornecido de forma objetiva e estruturada.
Responda sempre em português do Brasil.
Seja conciso e acadêmico."""


def analyze_chapter(
    client: LLMClient,
    chapter_title: str,
    chapter_text: str,
    max_chars: int = 6000,
) -> ChapterAnalysis:
    """Análise qualitativa de um capítulo via LLM."""

    # Trunca texto para não exceder contexto
    truncated = chapter_text[:max_chars]
    if len(chapter_text) > max_chars:
        truncated += "\n[... texto truncado para análise ...]"

    prompt = f"""Analise o seguinte capítulo de livro didático:

**Título:** {chapter_title}

**Texto:**
{truncated}

Forneça:
1. RESUMO (2-3 parágrafos)
2. OBJETIVOS DE APRENDIZADO (lista de 3-6 itens)
3. CONCEITOS-CHAVE (lista de 5-10 termos)
4. NOTAS PEDAGÓGICAS (1 parágrafo sobre abordagem metodológica)
5. NÍVEL DE DIFICULDADE: Iniciante / Intermediário / Avançado

Formato da resposta:
RESUMO: [texto]
OBJETIVOS: [item1 | item2 | item3]
CONCEITOS: [conceito1 | conceito2 | conceito3]
PEDAGOGIA: [texto]
NIVEL: [Iniciante/Intermediário/Avançado]"""

    response = client.complete(prompt, system=SYSTEM_PROMPT)
    return _parse_chapter_analysis(chapter_title, response)


def analyze_document_overview(
    client: LLMClient,
    title: str,
    author: str,
    chapter_summaries: list[str],
    overall_metrics: dict,
) -> DocumentAnalysis:
    """Análise geral do documento via LLM."""

    summaries_text = "\n".join(
        f"- Capítulo {i+1}: {s}" for i, s in enumerate(chapter_summaries[:15])
    )

    prompt = f"""Analise este livro didático com base nas informações fornecidas:

**Título:** {title}
**Autor:** {author}
**Total de palavras:** {overall_metrics.get('total_words', 'N/A')}
**Índice Flesch:** {overall_metrics.get('flesch', 'N/A')} ({overall_metrics.get('flesch_interp', '')})
**Capítulos:**
{summaries_text}

Forneça uma análise completa:
1. RESUMO GERAL (2-3 parágrafos)
2. PÚBLICO-ALVO
3. TEMAS PRINCIPAIS (lista de 5-8 temas)
4. ABORDAGEM PEDAGÓGICA (1 parágrafo)
5. PONTOS FORTES (lista de 3-5)
6. SUGESTÕES DE MELHORIA (lista de 3-5)

Formato:
RESUMO: [texto]
PUBLICO: [texto]
TEMAS: [tema1 | tema2 | tema3]
ABORDAGEM: [texto]
PONTOS_FORTES: [item1 | item2 | item3]
SUGESTOES: [item1 | item2 | item3]"""

    response = client.complete(prompt, system=SYSTEM_PROMPT, max_tokens=3000)
    return _parse_document_analysis(response)


def describe_image(
    client: LLMClient,
    image_path: str,
    page_number: int,
    context_text: str = "",
) -> ImageAnalysis:
    """Descreve uma imagem extraída do PDF."""

    # Apenas para provedores com suporte multimodal
    if client.provider in (LLMProvider.ANTHROPIC,):
        return _describe_image_multimodal(client, image_path, page_number, context_text)

    # Fallback: análise baseada apenas no contexto textual
    prompt = f"""Com base no contexto do texto ao redor (página {page_number}), 
descreva o que provavelmente é a imagem referenciada nesta seção do livro didático.

Contexto:
{context_text[:1000]}

Responda:
DESCRICAO: [descrição provável da imagem]
TIPO: [Gráfico / Diagrama / Fotografia / Tabela / Outro]
PAPEL_PEDAGOGICO: [como essa imagem provavelmente apoia o aprendizado]"""

    response = client.complete(prompt, system=SYSTEM_PROMPT)
    return _parse_image_analysis(response, image_path, page_number)


def _describe_image_multimodal(
    client: LLMClient,
    image_path: str,
    page_number: int,
    context_text: str,
) -> ImageAnalysis:
    """Análise multimodal de imagem via Claude."""
    import base64
    from pathlib import Path

    try:
        image_data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
        ext = Path(image_path).suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        import anthropic
        api_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = api_client.messages.create(
            model=client.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": f"""Esta imagem foi extraída da página {page_number} de um livro didático.
Contexto: {context_text[:500]}

Responda:
DESCRICAO: [descrição detalhada]
TIPO: [Gráfico / Diagrama / Fotografia / Tabela / Ilustração / Outro]
PAPEL_PEDAGOGICO: [função pedagógica da imagem]"""},
                ],
            }],
        )
        return _parse_image_analysis(msg.content[0].text, image_path, page_number)
    except Exception as e:
        return ImageAnalysis(
            page_number=page_number,
            image_path=image_path,
            description=f"Erro na análise: {str(e)}",
            image_type="Desconhecido",
            pedagogical_role="",
        )


# ─── Parsers de resposta ─────────────────────────────────────────────────────

def _parse_chapter_analysis(title: str, response: str) -> ChapterAnalysis:
    def extract(key: str) -> str:
        import re
        match = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_list(key: str) -> list[str]:
        raw = extract(key)
        return [item.strip() for item in raw.split("|") if item.strip()] if raw else []

    return ChapterAnalysis(
        chapter_title=title,
        summary=extract("RESUMO"),
        learning_objectives=extract_list("OBJETIVOS"),
        key_concepts=extract_list("CONCEITOS"),
        pedagogical_notes=extract("PEDAGOGIA"),
        difficulty_level=extract("NIVEL") or "Intermediário",
    )


def _parse_document_analysis(response: str) -> DocumentAnalysis:
    def extract(key: str) -> str:
        import re
        match = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_list(key: str) -> list[str]:
        raw = extract(key)
        return [item.strip() for item in raw.split("|") if item.strip()] if raw else []

    return DocumentAnalysis(
        overall_summary=extract("RESUMO"),
        target_audience=extract("PUBLICO"),
        main_themes=extract_list("TEMAS"),
        pedagogical_approach=extract("ABORDAGEM"),
        strengths=extract_list("PONTOS_FORTES"),
        suggestions=extract_list("SUGESTOES"),
    )


def _parse_image_analysis(response: str, image_path: str, page_number: int) -> ImageAnalysis:
    def extract(key: str) -> str:
        import re
        match = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return ImageAnalysis(
        page_number=page_number,
        image_path=image_path,
        description=extract("DESCRICAO") or response[:300],
        image_type=extract("TIPO") or "Desconhecido",
        pedagogical_role=extract("PAPEL_PEDAGOGICO") or "",
    )
