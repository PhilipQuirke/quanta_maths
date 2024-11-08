from setuptools import setup, find_packages

setup(
    name="MathsMechInterp",
    version="1.0",
    description="Maths Mechanistic Interpretability tools",
    author="Philip Quirke et al",
    author_email="philipquirkenz@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.1"
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
