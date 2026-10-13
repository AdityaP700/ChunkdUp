from setuptools import setup, find_packages

setup(
    name="chunkdup",
    version="0.2.0",
    description="Persistent memory runtime for AI agents",
    packages=find_packages(where="ai-system"),
    package_dir={"": "ai-system"},
    python_requires=">=3.8",
    # Core lightweight runtime requires zero external dependencies
    install_requires=[],
    extras_require={
        "postgres": [
            "psycopg2-binary>=2.9.9",
            "sqlalchemy>=2.0.0",
            "pgvector>=0.2.0",
            "python-dotenv>=1.0.0",
        ],
        "retrieval": [
            "numpy>=1.26.0",
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
            "rank-bm25>=0.2.2",
            "transformers>=4.40.0",
        ],
        "ml": [
            "numpy>=1.26.0",
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
            "rank-bm25>=0.2.2",
            "transformers>=4.40.0",
        ],
        "all": [
            "psycopg2-binary>=2.9.9",
            "sqlalchemy>=2.0.0",
            "pgvector>=0.2.0",
            "python-dotenv>=1.0.0",
            "numpy>=1.26.0",
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
            "rank-bm25>=0.2.2",
            "transformers>=4.40.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
