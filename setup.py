from setuptools import setup, find_packages
import versioneer

setup(
    name="torch_gauge",
    description="A light-weight PyTorch extension for gauge-equivariant geometric learning",
    author="Zhuoran Qiao",
    author_email="zqiao@caltech.edu",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
)
