from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Unsupervised Structural Embedding Framework'
LONG_DESCRIPTION = 'An unsupervised framework for exploring and evaluating node structural embedding of graphs'

setup(
    name="usef",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Ash Dehghan",
    author_email="ash.dehghan@gmail.com",
    license='BSD-3',
    packages=find_packages(),
    install_requires=['pandas==2.0.3', 'networkx==3.1', 'numpy==1.25.2', 'scipy==1.11.1', 'scikit-learn==1.3.0'],
    keywords='embedding',
    classifiers= [
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        'License :: BSD 3',
        "Programming Language :: Python :: 3.x",
    ]
)
