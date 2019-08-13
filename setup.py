import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bad-drbenvincent",
    version="0.0.1",
    author="Benjamin T. Vincent",
    author_email="b.t.vincent@dundee.ac.uk",
    description="Bayesian adaptive design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drbenvincent/bad/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
