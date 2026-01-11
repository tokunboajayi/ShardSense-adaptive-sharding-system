from setuptools import find_packages, setup

setup(
    name="shardsense",
    version="0.1.0",
    description="Adaptive Data Sharding with Online Load Imbalance Prediction",
    author="Tokunbo Ajayi",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "xgboost>=1.7.0",
        "streamlit>=1.20.0",
        "altair>=5.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "twine>=4.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "shardsense=shardsense.cli:main",
        ],
    },
)
