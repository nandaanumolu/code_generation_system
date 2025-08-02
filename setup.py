"""
ADK Multi-Agent System Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adk-multi-agent-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready multi-agent system built with ADK framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adk-multi-agent-system",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.5.0",
            "mkdocstrings>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adk-agent=adk_multi_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "adk_multi_agent": [
            "config/*.yaml",
            "templates/*.jinja2",
            "static/*",
        ],
    },
)