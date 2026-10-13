from pydantic import BaseModel, Field

from agents.tools.base import BaseTool


class TextAnalyzerInput(BaseModel):
    text: str = Field(..., description="The text content to analyze")


class TextAnalyzerTool(BaseTool):

    @property
    def name(self) -> str:
        return "text_analyzer"

    @property
    def description(self) -> str:
        return (
            "Analyzes text and returns word count, character count, and sentence count. "
            "Use this tool ONLY when the user asks to analyze, count, or measure properties of text."
        )

    @property
    def input_schema(self):
        return TextAnalyzerInput

    def execute(self, input_data: TextAnalyzerInput) -> dict:
        text = input_data.text
        words = text.split()
        sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return {
            "word_count": len(words),
            "character_count": len(text),
            "sentence_count": len(sentences),
        }
