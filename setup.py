from setuptools import setup, find_packages

setup(
    name='moo',
    version='0.0.1',
    author='Mivinci',
    description='ML model deployment made simple',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'numpy',
        'grpcio',
    ],
    python_requires='>=3.7',
)