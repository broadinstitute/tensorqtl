import os
import re
from setuptools import setup, find_packages
with open("tensorqtl/__init__.py") as reader:
    __version__ = re.search(
        r'__version__ ?= ?[\'\"]([\w.]+)[\'\"]',
        reader.read()
    ).group(1)
_README           = os.path.join(os.path.dirname(__file__), 'README.md')
_LONG_DESCRIPTION = open(_README).read()

# Setup information
setup(
    name = 'tensorqtl',
    version = __version__,
    packages = find_packages(),
    description = 'GPU-based QTL mapper',
    author = 'Francois Aguet (Broad Institute)',
    author_email = 'francois@broadinstitute.org',
    long_description = _LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    entry_points = {
        'console_scripts': [
            'tensorqtl = tensorqtl:__main__'
        ]
    },
    install_requires = [
    'numpy',
    'pandas',
    'pyarrow',
    'scipy',
    'pandas-plink',
    'torch',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
