# setup.py

from setuptools import setup, find_packages

setup(
    name="credit_risk_model_altdata",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
