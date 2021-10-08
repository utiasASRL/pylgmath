import setuptools

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="asrl-pylgmath",
    version="1.0.1",
    author="Yuchen Wu",
    author_email="cheney.wu@mail.utoronto.ca",
    description="Python implementation of lgmath",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utiasASRL/pylgmath",
    packages=setuptools.find_packages(),
    license="BSD",
    python_requires='>=3.8',
    install_requires=["numpy>=1.21.0", "scipy>=1.7.0"],
)
