"""Setup configuration for lukit."""

from setuptools import find_packages, setup

setup(
    name="lukit",
    version="0.1.0",
    description="LLM Uncertainty Kit - A toolkit for evaluating uncertainty in LLM responses",
    author="LUKIT Contributors",
    author_email="lukit@baidu.com",
    url="https://github.com/baidu/lukit",
    packages=find_packages(),
    package_data={
        "": ["configs/*.json"],
    },
    include_package_data=True,
    install_requires=[
        "datasets>=4.0.0",
        "nltk>=3.8.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "lukit=lukit.cli.main:main",
            "lukit-eval=lukit.bin.evaluate:main",
            "lukit-leaderboard=lukit.bin.leaderboard:main",
            "lukit-visualize=lukit.bin.visualize:main",
        ],
    },
    python_requires=">=3.8",
)
