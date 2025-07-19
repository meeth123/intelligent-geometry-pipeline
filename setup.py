#!/usr/bin/env python3
"""
Setup script for Intelligent Geometry Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intelligent-geometry-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Complete 6-Agent AI System powered by Google Gemini 2.5 Pro for Geometric Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/intelligent-geometry-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "geometry-pipeline=frontend:main",
        ],
    },
    keywords="ai gemini geometry mathematics constraint-solving computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/intelligent-geometry-pipeline/issues",
        "Source": "https://github.com/yourusername/intelligent-geometry-pipeline",
        "Documentation": "https://github.com/yourusername/intelligent-geometry-pipeline/wiki",
    },
) 