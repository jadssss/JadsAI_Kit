# setup.py
from setuptools import setup, find_packages

setup(
    name="jadsai_kit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.10.0',
        'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'jadsai-kit = jadsai_kit.core:main',
        ],
    },
    author="jads___",
    author_email="jads___@oaklog.online",
    description="Open-source neural network builder with GUI & CLI modes",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jads___/jadsai_kit",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
