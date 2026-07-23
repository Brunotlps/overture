"""User-facing language support for agent answers.

Internal prompts, logs, and API error details stay in English; only text the
end user reads (the LLM answer and canned fallback answers) is localized.
"""

from typing import Literal

Language = Literal["pt-BR", "en"]

DEFAULT_LANGUAGE: Language = "pt-BR"

# Appended to the system prompt so the answer language is explicit instead of
# relying on the LLM mirroring the language of the question.
ANSWER_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "pt-BR": (
        "Write the final answer in Brazilian Portuguese (pt-BR), regardless of "
        "the language of the question or of the repository content. Keep code "
        "identifiers, file paths, and quoted code in their original form."
    ),
    "en": (
        "Write the final answer in English, regardless of the language of the "
        "question or of the repository content. Keep code identifiers, file "
        "paths, and quoted code in their original form."
    ),
}

EMPTY_FINAL_ANSWER_MESSAGES: dict[str, str] = {
    "pt-BR": (
        "Não consegui produzir uma resposta final a partir da resposta do "
        "modelo. Tente reformular a pergunta."
    ),
    "en": (
        "I could not produce a final answer from the model response. "
        "Please try rephrasing the question."
    ),
}

BUDGET_EXCEEDED_MESSAGES: dict[str, str] = {
    "pt-BR": (
        "Atingi o número máximo de chamadas de ferramenta permitidas para esta "
        "requisição. Restrinja a pergunta ou pergunte sobre uma parte mais "
        "específica do repositório."
    ),
    "en": (
        "I reached the maximum number of tool calls allowed for this request. "
        "Please narrow the question or ask about a more specific part of the "
        "repository."
    ),
}


def get_message(messages: dict[str, str], language: str) -> str:
    """Return the message for language, falling back to the default language."""
    return messages.get(language, messages[DEFAULT_LANGUAGE])
