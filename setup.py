import os
import sys
import subprocess

from distutils.util import convert_path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

version_ns = {}
vpath = convert_path("arches/version.py")
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

# Parse command line flags
# TODO: add parser to run specific tests
options = {k: "OFF" for k in ["--test"]}
for flag in options.keys():
    if flag in sys.argv:
        options[flag] = "ON"
        sys.argv.remove(flag)

# Command line flags forwarded to CMake
cmake_cmd_args = []
for f in sys.argv:
    if f.startswith("-D"):
        cmake_cmd_args.append(f)

for f in cmake_cmd_args:
    sys.argv.remove(f)


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=".", **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            cmake_args = [
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            ]

            cmake_args += cmake_cmd_args

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp)

            # Build
            subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


c_module_name = "arches.kernels"

setup(
    name="ARCHES",
    version=version_ns["__version__"],
    packages=find_packages(),
    description="Argonne Configuration Interaction for High-Performance Exascale Systems ",
    url="https://github.com/lerandc/arches/",
    author="Luis Rangel DaCosta",
    author_email="luisrd@berkeley.edu",
    python_requires=">=3.10",
    install_requires=["numpy", "mpi4py"],
    # ext_modules=[CMakeExtension(c_module_name)],
    cmdclass={"build_ext": cmake_build_ext},
)
