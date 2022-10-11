from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='kubemo',
    version='0.0.1',
    author='Mivinci',
    description='ML model deployment made simple',
    long_description=long_description,
    url='https://github.com/kubemo/kubemo',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'numpy',
    ],
    python_requires='>=3.7',
)