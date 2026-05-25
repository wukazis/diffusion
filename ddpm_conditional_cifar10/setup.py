from setuptools import find_packages, setup


setup(
    name="ddpm-conditional-cifar10",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["blobfile", "numpy", "pillow", "torch", "torchvision", "tqdm"],
)
