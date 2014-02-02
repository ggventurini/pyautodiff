import sys
from setuptools import setup

if sys.version < '3':
    raise ImportError(
        'This version of autodiff only support Python 3+. Please check out an '
        'earlier branch for use with Python 2.')

setup(
    name='autodiff',
    version='0.5',
    maintainer='Lowin Data Company',
    maintainer_email='info@lowindata.com',
    description=('Automatic differentiation for NumPy.'),
    license='BSD-3',
    url='https://github.com/LowinData/pyautodiff',
    long_description = open('README.md').read(),
    install_requires=['numpy', 'theano', 'meta'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ]
)
