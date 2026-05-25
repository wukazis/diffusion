from setuptools import find_packages, setup


setup(
    name="flow-matching-facades-i2i",
    packages=find_packages(),
    install_requires=["blobfile>=1.0.5", "numpy", "pillow", "torch", "tqdm"],
)
