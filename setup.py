"""Setup for fanpy.

See:
https://packaging.python.org/en/latest/distributing.html
Templated from:
https://github.com/pypa/sampleproject
"""

from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fanpy",
    version="1.0.0",
    description="A module for developing and testing multideterminant wavefunctions and related ab initio methods in electronic structure theory.",
    long_description=long_description,
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type="text/markdown",
    url="https://github.com/mqcomplab/Fanpy",
    # This should be your name or the name of the organization which owns the
    # project.
    author="MQCompLab",
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="ramirandaq@gmail.com",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        # Specify the Python versions you support here.
        # These classifiers are *not* checked by 'pip install'. See instead
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="wavefunction hamiltonian optimization schrodinger equation quantum chemistry",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.25", 
                      "scipy>=1.9", 
                      "cython", 
                      "pandas", 
                      "cma", 
                      "psutil"],
    extras_require={
        "dev": [
            "tox",
            "pytest",
            "pytest-cov",
            "flake8",
            "flake8-pydocstyle",
            "flake8-import-order",
            "pep8-naming",
            "pylint",
            "bandit",
            "black",
        ],
        "test": ["pytest", "pytest-cov", "numdifftools"],
        "horton": ["horton"],
        "pyscf": ["pyscf"],
        "tensorflow": ["tensorflow"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={},
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    entry_points={
        "console_scripts": [
            "fanpy_make_script=fanpy.scripts.gaussian.make_script:main",
            "fanpy_run_calc=fanpy.scripts.gaussian.run_calc:main",
            "fanpy_make_fanci_script=fanpy.scripts.gaussian.make_fanci_script:main",
            "fanpy_make_pyscf_script=fanpy.scripts.pyscf.make_script:main",
            "fanpy_make_fanci_pyscf_script=fanpy.scripts.pyscf.make_fanci_script:main",
        ]
    },
    # List additional URLs that are relevant to your project as a dict.
    project_urls={
        "Bug Reports": "https://github.com/mqcomplab/Fanpy/issues",
        "Organization": "https://github.com/mqcomplab",
        "Source": "https://github.com/mqcomplab/Fanpy/",
    },
    zip_safe=False,
)
