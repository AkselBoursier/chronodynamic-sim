#!/usr/bin/env python3
"""
Setup script for Chronodynamic Cosmology Simulation Suite
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chronodynamic-sim",
    version="1.0.0",
    author="Aksel Boursier",
    author_email="akselboursier@example.com",
    description="Numerical simulation suite for Chronodynamic Cosmological Divergence (CCD) model",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/akselboursier/chronodynamic-sim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
        "class": [
            "classy>=2.9.0",
        ],
        "camb": [
            "camb>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chronodynamic-sim=chronodynamic_sim.cli:main",
            "ccd-mcmc=chronodynamic_sim.scripts.run_mcmc:main",
            "ccd-dashboard=chronodynamic_sim.visualization.interactive_dashboard:main",
        ],
    },
    package_data={
        "chronodynamic_sim": [
            "data/*.json",
            "data/*.yaml",
            "configs/*.yml",
            "configs/*.toml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)