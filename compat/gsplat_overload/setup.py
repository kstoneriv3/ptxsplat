from setuptools import find_packages, setup


VERSION = "0.1.0"

setup(
    name="ptxsplat-gsplat-overload",
    version=VERSION,
    description="Optional gsplat import compatibility layer for ptxsplat",
    url="https://github.com/kstoneriv3/ptxsplat",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[f"ptxsplat=={VERSION}"],
)
