# 1. Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="chunkdup",
    version="0.2.0",
    packages=find_packages(where="ai-system"),
    package_dir={"": "ai-system"},
    install_requires=[
        "psycopg2-binary>=2.9.9",
        "sqlalchemy>=2.0.0",
        "pgvector>=0.2.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)
