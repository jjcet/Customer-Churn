import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


__version__ = "0.0.0"

REPO_NAME = "Customer-Churn"
AUTHOR_USER_NAME = "jjcet"
SRC_REPO = "CustomerChurn"
AUTHOR_EMAIL = "dycjh@example.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Customer Churn Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjcet/{}".format(REPO_NAME),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
    