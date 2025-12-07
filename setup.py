from setuptools import setup
import setuptools

setup(
    name='normalizingflows',
    version='0.2',
    author="Antoine Wehenkel",
    author_email="antoine.wehenkel@gmail.com",
    description="Implementation of affine and monotonic normalizing flows - autoregressive/coupling/graphical conditioners",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/awehenkel/Normalizing-Flows",
    install_requires=['umnn', 'torch', 'networkx'],
    packages=setuptools.find_packages(),
    classifiers=[
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: MIT License",
     "Operating System :: OS Independent",
     ],
)