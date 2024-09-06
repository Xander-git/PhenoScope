from setuptools import setup, find_packages

VERSION = "0.3.0"
DESCRIPTION = "A python package for high-throughput analysis of microorganism colonies on solid media plates"

setup(
        name="phenoscope",
        version=VERSION,
        description=DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Alexander Nguyen",
        author_email="anguy344@ucr.edu",
        url="https://github.com/Xander-git/PhenoScope",
        packages=find_packages(),  # Automatically finds all packages
        install_requires=[
            "setuptools<60.0",
            "numpy",
            "pandas",
            "scikit-image",
            "scikit-learn",
            "joblib",
            "boto3",
            "docutils",
            "h5py",
            "imageio",
            "inflect",
            "mahotas",
            "mysqlclient",
            "psutil",
            "pytest",
            "pyzmq",
            "scipy",
            "scyjava",
            "seaborn",
            "centrosome",
            "cellprofiler",
            "cellprofiler-core"
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.10',
)
