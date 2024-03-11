import os
from setuptools import setup, find_packages
import scattermoe

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "scattermoe",
    version = "0.0.0",
    author = "Shawn Tan",
    author_email = "shawn@wtf.sg",
    description = "Triton-based implementation of Sparse Mixture of Experts.",
    license = "Apache License",
    keywords = "triton pytorch llm",
    url = "",
    packages=find_packages(),
    long_description=read('README.md'),
    install_requires=['torch', 'triton'],
    tests_require=['pytest'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
    ],
)

