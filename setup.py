from setuptools import setup, find_packages

VERSION = '0.0.1'
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
    install_requires=['pandas', 'networkx', 'numpy', 'scipy', 'sklearn'],
    keywords='embedding',
    classifiers= [
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        'License :: BSD 3',
        "Programming Language :: Python :: 3.x",
    ]
)