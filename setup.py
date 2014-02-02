from setuptools import setup

setup(
    name='autodiff',
    version='0.4',
    maintainer='Lowin Data Company',
    maintainer_email='info@lowindata.com',
    description=('Automatic differentiation for NumPy.'),
    license='new BSD',
    url='https://github.com/LowinData/pyautodiff',
    long_description = open('README.md').read(),
    install_requires=['numpy', 'theano', 'meta'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)
