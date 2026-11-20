"""Task types (qa, classification, extraction, safety, rag, generic)."""

from llmbench.tasks.base import (
    TASKS,
    ClassificationTask,
    ExtractionTask,
    QATask,
    RAGTask,
    SafetyTask,
    Task,
    get_task,
    is_abstention,
)

__all__ = [
    "TASKS",
    "ClassificationTask",
    "ExtractionTask",
    "QATask",
    "RAGTask",
    "SafetyTask",
    "Task",
    "get_task",
    "is_abstention",
]
