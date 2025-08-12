from setuptools import setup, find_packages

setup(
    name="RadGEEToolbox",
    version="1.6.5",
    author="Mark Radwin",
    author_email="markradwin@gmail.com",
    description="Streamlined Multispectral & SAR Analysis for Google Earth Engine Python API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/radwinskis/RadGEEToolbox",
    packages=find_packages(),
    include_package_data=True,  # This line includes non-code files
    package_data={
        '': ['notebooks/*.ipynb'],},  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "earthengine-api",
        "numpy",
        "pandas"
    ],
    python_requires=">=3.8",
)