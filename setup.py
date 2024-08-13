#!/usr/bin/env python

# Author: Faisal Jayousi

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        install_requires=[
            "numpy>=1.25.0",
            "skan==0.11.0",
            "skl-graph==2023.15",
        ],
    )
