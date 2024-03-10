import os
from setuptools import setup

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
    packages=['scattermoe'],
    long_description=read('README.md'),
    requires=['torch', 'triton'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
    ],
)

