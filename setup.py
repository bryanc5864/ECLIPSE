"""
ECLIPSE: Setup configuration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eclipse-ecdna",
    version="0.1.0",
    author="Bryan Cheng and Jasper Zhang",
    description="Extrachromosomal Circular DNA Learning for Integrated Prediction of Synthetic-lethality and Expression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bryanc5864/ECLIPSE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "full": [
            "torch-geometric>=2.3.0",
            "torchsde>=0.2.5",
            "transformers>=4.30.0",
            "cooler>=0.9.0",
            "pybedtools>=0.9.0",
            "dowhy>=0.9.0",
            "wandb>=0.15.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eclipse=main:main",
        ],
    },
)
