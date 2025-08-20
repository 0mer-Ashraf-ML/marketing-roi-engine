from setuptools import setup, find_packages

setup(
    name="advertising-roi-engine",
    version="1.0.0",
    description="AI-Driven Advertising ROI Optimization & Attribution Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ROI Analytics Team",
    author_email="analytics@company.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "cvxpy>=1.2.0",
        "statsmodels>=0.13.0",
        "python-dateutil>=2.8.0",
        "pytz>=2022.1",
        "tqdm>=4.64.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0"
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "roi-engine=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False
)
