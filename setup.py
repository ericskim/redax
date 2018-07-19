from setuptools import find_packages, setup

setup(
    name='vpax',
    version='0.0.1',
    description='Variable Precision Abstract Control Synthesis (VPAX)',
    author='Eric S. Kim',
    author_email='eskim@eecs.berkeley.edu',
    license='MIT',
    entry_points={
        
    },
    install_requires=[
        'bidict',
        'toposort',
        'dd'
    ],
    packages=find_packages(),
)
