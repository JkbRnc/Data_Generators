from xml.etree.ElementInclude import include
from setuptools import setup, find_packages

setup(
    name="TabularDataGenerator",
    version='1.0.0',
    python_requires='>=3.10.4',
    packages=find_packages(
        include=['Data_Generators', 'Data_Generators.*']
        ),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'torch>=1.8.0,<2'
    ]
)