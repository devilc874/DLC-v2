"""Setup script for the DLC Python package."""

from setuptools import setup, find_packages

setup(
    name="dlc",
    version="0.1.0",
    description="Delta-Linear Compression engine for time-series sensor data",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "hypothesis>=6.0",
            "pytest>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dlc=dlc.cli:main",
        ],
    },
)
