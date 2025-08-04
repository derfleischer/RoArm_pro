"""Setup script for RoArm Pro"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="roarm-pro",
    version="1.0.0",
    author="RoArm Pro Team",
    description="Professional RoArm M3 Controller - Focused macOS Edition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyserial>=3.5",
        "numpy>=1.24.3",
        "pyyaml>=6.0.1",
        "dataclasses-json>=0.6.1",
    ],
    entry_points={
        "console_scripts": [
            "roarm=roarm_pro.main:main",
        ],
    },
    package_data={
        "roarm_pro": ["config/*.yaml"],
    },
)
