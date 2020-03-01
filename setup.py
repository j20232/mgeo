from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mgeo",
    version="0.1",
    author="mocobt",
    author_email="mocobt@gmail.com",
    description="image-based geometry processing library",
    long_description=long_description,
    license="BSD 3-Clause",
    packages=find_packages(),
    test_suite="tests.context.suite"
)
