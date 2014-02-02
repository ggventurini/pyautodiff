import sys
from setuptools import setup

if not sys.version < '3':
    raise ImportError(
        'This version of autodiff does not support Python 3+. There may be '
        'another branch available that does.')

setup(
    name='autodiff',
    version='0.4',
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
        'Programming Language :: Python :: 2',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ]
)
