"""Setup script for PyTorch Operation Tracer"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-op-tracer",
    version="1.0.0",
    author="UniAD PyTorch Tracer Team",
    author_email="",
    description="A comprehensive tool for tracing PyTorch operations with UniAD support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenDriveLab/UniAD",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "uniad": [
            "mmcv-full>=1.4.0",
            "mmdet>=2.14.0",
            "mmdet3d>=0.17.1",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "pandas>=1.1.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "pytorch-trace=pytorch_op_tracer.trace_ops:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)