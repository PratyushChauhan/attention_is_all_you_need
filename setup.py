from setuptools import setup, find_packages
import torch

setup(
    name='my_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here
        torch,
    ],
    author='Your Name',
    description='A brief description of your project',
)
