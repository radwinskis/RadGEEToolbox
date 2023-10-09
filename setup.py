from setuptools import setup, find_packages

setup(
    name="RadGEEToolbox",
    version="1.3",
    author="Mark Radwin",
    author_email="markradwin@gmail.com",
    description="Python package simplifying large-scale operations using Google Earth Engine (GEE) for users who utilize Landsat and Sentinel",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/radwinskis/RadGEEToolbox",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "ee",
        "geemap"
    ],
    python_requires=">=3.6",
)