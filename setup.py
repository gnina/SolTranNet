from setuptools import setup, find_packages
import pathlib

here=pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding='utf-8')

setup(

    name='soltrannet',

    version='1.0.0',

    description='Molecule attention transformer for predicting aqueous solubility (log S)',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/gnina/SolTranNet',

    author='Paul Francoeur',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='ML, solubility, tool',

    packages=find_packages(),

    python_requires='>=3.6, <4',

    install_requires=[
        'torch >= 1.7.0',
        'pathlib >= 1.0',
    ],

    package_data={
        'soltrannet':['soltrannet_aqsol_trained.weights'],
    },

    entry_points={
        'console_scripts':['soltrannet=soltrannet.command_line:main'],
    },

    test_suite='nose.collector',
    tests_require=['nose','nose-cover3'],

    project_urls={
        'Bug Reports': 'https://github.com/gnina/SolTranNet/issues',
        'Source': 'https://github.com/gnina/SolTranNet',
    },
)
