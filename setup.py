from setuptools import setup, find_packages
from distutils.util import convert_path


version_ns = {}
vpath = convert_path("qe/version.py")
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

setup(
    name="QuantumEnvelope",
    version=version_ns["__version__"],
    packages=find_packages(),
    description="An open source python package for processing and analysis of 4D STEM data.",
    url="https://github.com/lerandc/QuantumEnvelope/",
    author="Luis Rangel DaCosta",
    author_email="luisrd@berkeley.edu",
    python_requires=">=3.9",
    install_requires=[
       "numpy",
       "mpi4py"
    ],
)