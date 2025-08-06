#!/usr/bin/env python3
"""
RoArm M3 Professional Control System
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "RoArm M3 Professional Control System"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().split('\n') 
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        'pyserial>=3.5',
        'pyyaml>=6.0',
        'numpy>=1.20.0',
        'colorama>=0.4.4'
    ]

setup(
    name='roarm-m3-professional',
    version='2.0.0',
    author='RoArm Professional Team',
    description='Professional control system for Waveshare RoArm M3 robot',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/derfleischer/RoArm_pro',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt', '*.md'],
    },
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'pandas>=1.3.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'roarm=main:main',
            'roarm-debug=utils.debug_tool:main',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Robotics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='robotics robot-arm waveshare roarm control-system 3d-scanning',
)
