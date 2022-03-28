# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

print(find_packages(where=".", exclude=('tests', 'docs')))

setup(
    name='ai',
    version='0.1.0',
    description='AI related source-code for the chatbot project',
    long_description='Lo',
    author='Upen Dhakal',
    # url='https://github.com/kennethreitz/samplemod',
    # license=license,
    packages=find_packages(where=".", exclude=('tests', 'docs'))
)