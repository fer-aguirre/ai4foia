from setuptools import setup, find_packages

setup(
    name='AI4FOIA',
    version='0.1.0',
    author='Fer Aguirre',
    description='Proof-of-concept to recommend recipients for FOIA requests.',
    python_requires='>=3',
    license='MIT License',
    packages=find_packages(),
)