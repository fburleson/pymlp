from setuptools import setup, find_packages

setup(
    name="pymlp",
    description="A small library for creating and training fully connected neural networks",
    long_description=open("README.md").read(),
    packages=find_packages(),
    version="0.1.0",
    install_requires=["numpy", "seaborn", "pandas", "pytest"],
    author="Joel Burleson",
    url="https://github.com/fburleson/pymlp",
    python_requires=">=3.10",
)
