from setuptools import setup, find_packages
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name='module0-charge-light-association',
        version='0.0',
        description='',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/peter-madigan/module0_charge_light_association',
        author='Peter Madigan',
        author_email='pmadigan@lbl.gov',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3'
        ],
        packages=find_packages(),
        install_requires=[
            'h5py ~=3.2.1',
            'numpy ~=1.19.1',
            'tqdm'
            ],
        scripts=[
            'charge_light_association.py'
            ],
)
