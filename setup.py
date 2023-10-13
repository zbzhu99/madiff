from setuptools import setup

setup(
    name="madiff",
    description="Multi-agent Diffusion Model.",
    packages=["diffuser"],
    package_dir={
        "diffuser": "./diffuser",
    },
)
