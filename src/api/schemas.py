"""
Structured output schemas for Medical QA via Pydantic.
Ensures validated, typed LLM responses.
"""

from pydantic import BaseModel, Field


class MedicalAnswer(BaseModel):
    """Structured schema for medical QA response."""

    answer: str = Field(description="Текстовый ответ на медицинский вопрос")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность модели в ответе (0-1)",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Список источников/документов, использованных для ответа",
    )
    requires_doctor_visit: bool = Field(
        description="Требуется ли рекомендовать визит к врачу",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Предупреждения (например: 'Не является медицинской консультацией')",
    )

class Query(BaseModel):
    question: str