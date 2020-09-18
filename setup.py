#!/usr/bin/env python
import io
from setuptools import setup, find_packages


readme = io.open('README.md', encoding='utf-8').read()

requirements = ['numpy',
                "sklearn",
                "joblib==0.14.1",
                "sentencepiece",
                "six>=1.12.0"
                ]
setup(
    # Metadata
    name='easytransfer',
    version='0.1.1',
    python_requires='>=2.7,>=3.6',
    author='PAI NLP',
    author_email='minghui.qmh@alibaba-inc.com,'
                 'jerry.lp@alibaba-inc.com,'
                 'haojie.phj@alibaba-inc.com',
    url='http://gitlab.alibaba-inc.com/groups/PAI-TL',
    description='PAI EasyTransfer Toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    extras_require={
        "tf": ["tensorflow==1.12"],
        "tf_gpu": ["tensorflow-gpu==1.12"]
    },
    entry_points = {
        'console_scripts': [
            'easy_transfer_app=easytransfer.app_zoo_cli:main',
            'ez_bert_feat=easytransfer.feat_ext_cli:main']
    },
    packages=find_packages(),
    license='Apache-2.0',

    # Package info
    install_requires=requirements
)
