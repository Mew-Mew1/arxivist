"""setup.py for react_agent -- reproduction of ReAct (arXiv:2210.03629)."""

from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="react_agent",
    version="0.1.0",
    description="Reproduction of ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=install_requires,
)
