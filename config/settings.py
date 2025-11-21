"""
Configuration settings for the ArXiv Research Publishing System.

Uses Pydantic Settings for configuration management with environment variable support.
"""

from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Email Configuration
    # smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    # smtp_port: int = Field(default=587, description="SMTP server port")
    # smtp_username: str = Field(default="", description="SMTP username")
    # smtp_password: str = Field(default="", description="SMTP password")
    # smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")
    # from_email: str = Field(default="", description="From email address")

    # ArXiv Configuration
    min_papers_threshold: int = Field(
        default=100,
        description="Minimum number of papers to fetch before analysis",
        ge=10,
    )
    papers_per_topic: int = Field(
        default=5,
        description="Number of representative papers to select per topic",
        ge=1,
        le=10,
    )
    default_days_back: int = Field(
        default=7,
        description="Default number of days to look back for papers",
        ge=1,
    )
    max_papers_per_agent: int = Field(
        default=1000,
        description="Maximum papers an agent can fetch",
        ge=10,
    )

    # Topic Modeling Configuration
    min_topic_size: int = Field(
        default=5,
        description="Minimum number of papers for a topic",
        ge=2,
    )
    max_topics: int = Field(
        default=10,
        description="Maximum number of topics to extract",
        ge=1,
        le=20,
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    # TODO: check Arxiv tags
    # Agent ArXiv Categories Configuration
    agent_categories: Dict[str, List[str]] = Field(
        default={
            "michel": ["math.HO", "math.GM"],  # History and Overview, General Mathematics
            "chris": ["math.PR", "stat.TH"],  # Probability, Statistics Theory
            "alain": ["math.AG", "math.RA", "math.GR"],  # Algebraic Geometry, Rings and Algebras, Group Theory
            "bruno": ["math.DG", "math.SP"],  # Differential Geometry, Spectral Theory
            "elisa": ["cs.CR", "math.OC"],  # Cryptography, Optimization and Control
            "felix": ["math.DS", "math.SG"],  # Dynamical Systems, Symplectic Geometry
            "abdoulaye": ["cs.LG", "stat.ML"],  # Machine Learning, Machine Learning (stats)
        },
        description="ArXiv categories for each specialized agent",
    )

    # Directory Configuration
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory for output files",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for data storage",
    )
    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Directory for cached data",
    )
    pdf_dir: Path = Field(
        default=Path("data/pdfs"),
        description="Directory for downloaded PDFs",
    )

    # Processing Configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
        ge=1,
    )
    max_workers: int = Field(
        default=4,
        description="Number of parallel workers for downloads",
        ge=1,
        le=10,
    )
    request_delay: float = Field(
        default=3.0,
        description="Delay between ArXiv API requests (seconds)",
        ge=0.5,
    )

    # One-Pager Generation Configuration
    target_word_count: int = Field(
        default=800,
        description="Target word count for one-pager",
        ge=300,
        le=2000,
    )
    max_papers_in_summary: int = Field(
        default=10,
        description="Maximum papers to include in one-pager",
        ge=3,
        le=20,
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Path = Field(
        default=Path("arxiv_editor.log"),
        description="Log file path",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )

    # LLM Configuration (optional, for enhanced generation)
    use_llm: bool = Field(
        default=False,
        description="Use LLM for enhanced text generation",
    )
    llm_provider: str = Field(
        default="anthropic",
        description="LLM provider (anthropic, openai)",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM provider",
    )
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="LLM model to use",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("output_dir", "data_dir", "cache_dir", "pdf_dir")
    @classmethod
    def create_directory(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("from_email")
    @classmethod
    def set_default_from_email(cls, v: str, info) -> str:
        """Set from_email to smtp_username if not provided."""
        if not v and "smtp_username" in info.data:
            return info.data["smtp_username"]
        return v

    def get_agent_categories(self, agent_name: str) -> List[str]:
        """
        Get ArXiv categories for a specific agent.

        Args:
            agent_name: Name of the agent (lowercase)

        Returns:
            List of ArXiv category codes
        """
        return self.agent_categories.get(agent_name.lower(), [])

    def get_all_categories(self) -> List[str]:
        """
        Get all unique ArXiv categories across all agents.

        Returns:
            List of unique ArXiv category codes
        """
        all_cats = []
        for categories in self.agent_categories.values():
            all_cats.extend(categories)
        return list(set(all_cats))

    def validate_email_config(self) -> bool:
        """
        Check if email configuration is complete.

        Returns:
            True if all required email settings are provided
        """
        return bool(
            self.smtp_host
            and self.smtp_username
            and self.smtp_password
        )


# Create a singleton instance
_settings = None


def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
