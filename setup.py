from setuptools import find_packages, setup

setup(
    name='redax',
    version='0.0.1',
    description='Control Synthesizer in Python with (Dynamic|Declarative|Robust) Abstractions',
    author='Eric S. Kim',
    author_email='eskim@eecs.berkeley.edu',
    license='BSD-3',
    entry_points={

    },
    install_requires=[
        'bidict',
        'dd',
        'numpy',
        'toposort',
        'pyqtgraph',
        'pyopengl',
        'funcy',
        'dataclasses'
    ],
    packages=find_packages(),
)
