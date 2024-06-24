from setuptools import setup, find_packages

setup(
    name="giskardpy",
    version="1.0.0",
    author="Simon Stelter",
    author_email="stelter@uni-bremen.de",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SemRoCo/giskardpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
    ],
)