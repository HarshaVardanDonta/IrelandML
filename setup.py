from setuptools import find_packages, setup

setup(
    name="IrelandML",
    packages=find_packages(exclude=["IrelandML_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
