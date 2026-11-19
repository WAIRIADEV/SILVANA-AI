
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import settings


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for the isolated RAG evaluation environment."""

    root: Path

    database_path: Path
    storage_root: Path

    dataset_path: Path
    corpus_root: Path
    results_root: Path

    max_file_size_bytes: int = 20 * 1024 * 1024

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_threads: int = 2

    chunk_max_tokens: int = 450
    chunk_token_overlap: int = 60

    retrieval_top_k: int = 5

    @classmethod
    def create(cls) -> "EvaluationConfig":
        """Build paths relative to this evaluation project."""

        root = Path(__file__).resolve().parent
        data_root = root / "data"

        return cls(
            root=root,
            database_path=data_root / "evaluation.sqlite3",
            storage_root=data_root / "documents",
            dataset_path=root / "datasets" / "baseline.json",
            corpus_root=root / "corpus" / "documents",
            results_root=root / "results",
        )

    def validate_isolation(self) -> None:
        """Reject evaluation paths that overlap production resources."""

        evaluation_database = self.database_path.resolve()
        evaluation_storage = self.storage_root.resolve()

        production_storage = (
            settings.SILVANA_DOCUMENT_STORAGE_ROOT.resolve()
        )

        production_database = (
            production_storage.parent / "documents.sqlite3"
        ).resolve()

        protected_paths = {
            "production document storage": production_storage,
            "production document database": production_database,
        }

        evaluation_paths = {
            "evaluation database": evaluation_database,
            "evaluation storage": evaluation_storage,
        }

        for evaluation_name, evaluation_path in evaluation_paths.items():
            for protected_name, protected_path in protected_paths.items():
                if (
                    evaluation_path == protected_path
                    or evaluation_path.is_relative_to(protected_path)
                    or protected_path.is_relative_to(evaluation_path)
                ):
                    raise RuntimeError(
                        f"Evaluation isolation failure: "
                        f"{evaluation_name} overlaps "
                        f"{protected_name}."
                    )

        if evaluation_database.is_relative_to(evaluation_storage):
            raise RuntimeError(
                "Evaluation database must not be inside "
                "evaluation document storage."
            )

        if evaluation_storage.is_relative_to(evaluation_database):
            raise RuntimeError(
                "Evaluation storage must not be inside "
                "the evaluation database path."
            )

    def prepare_directories(self) -> None:
        """Create evaluation directories after isolation validation."""

        self.validate_isolation()

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.corpus_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results_root.mkdir(
            parents=True,
            exist_ok=True,
        )


evaluation_config = EvaluationConfig.create()
