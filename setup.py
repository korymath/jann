from setuptools import setup
from setuptools import find_packages


setup(
    name="Jann",
    verson="0.0.1",
    description="Jann is a Nearest Neighbour retrieval-based chatbot.",
    author="Kory Mathewson",
    author_email="korymath@gmail.com",
    license="MIT",
    url="https://github.com/korymath/jann",
    packages=find_packages(),
    setup_requires=[
        "pytest-runner"
    ],
    tests_require=[
        "pytest"
    ],
)
