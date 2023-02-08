from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='ani_ext',
    description='Python interface to new ANI models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/plin1112/ani_ext',
    author='Ping Lin',
    license='Apache License 2.0',
    author_email='plin1112@ufl.edu',
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)
