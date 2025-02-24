from setuptools import setup, find_packages

def read_requirements(file):
    with open(file, 'r') as f:
        return f.read().splitlines()

setup(
    name="glm4voice-finetune",
    version="0.1.0",
    packages=find_packages(where='src'),
    python_requires=">=3.8"
)
