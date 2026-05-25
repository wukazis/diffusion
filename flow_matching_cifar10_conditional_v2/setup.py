from setuptools import find_packages, setup


setup(
    name="flow-matching-cifar10-conditional",
    packages=find_packages(),
    install_requires=["blobfile>=1.0.5", "numpy", "pillow", "torch", "torchvision", "tqdm"],
)
