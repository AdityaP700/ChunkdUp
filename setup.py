# setup.py (in the ROOT)
from setuptools import setup, find_packages

setup(
    name="chunkdup",
    version="0.1.0",
    package_dir={'': 'ai-system'}, 
    packages=find_packages(where='ai-system'),
    install_requires=[
        "sentence-transformers",
        "google-generativeai",
        "anthropic",
    ],
    python_requires=">=3.8",
)