from setuptools import setup, find_packages
import versioneer

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="torch_gauge",
    description="A light-weight PyTorch extension for gauge-equivariant geometric learning",
    author="Zhuoran Qiao",
    author_email="zqiao@caltech.edu",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=['torch>=1.7.0'],
)
