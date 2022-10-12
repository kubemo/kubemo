from setuptools import setup, find_packages

# basics
name = 'kubemo'
version = '0.0.dev2'
author = 'mivinci'
description = 'ML model deployment made simple'
repo_url = 'https://github.com/kubemo/kubemo'

# Python
python_requires = '>=3.7'

# introduction
with open('README.md') as f:
    long_description = f.read()


# dependencies
install_requires = [
    'Pillow',
    'numpy',
    'requests',
]


# setup
setup(
    name=name,
    version=version,
    author=author,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=repo_url,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=python_requires,
)