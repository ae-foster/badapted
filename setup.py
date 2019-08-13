import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="badapted",
    version="0.0.1",
    author="Benjamin T. Vincent",
    author_email="b.t.vincent@dundee.ac.uk",
    description="Bayesian ADAPTive Experimental Design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drbenvincent/badapted",
    keywords = ["bayesian", "adaptive design", "inference", "optimisation"],
    packages=['badapted'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
