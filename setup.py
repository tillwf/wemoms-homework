from setuptools import find_packages, setup
from pathlib import Path

def load_requirements():
    req = Path('__file__').parent / f"requirements.txt"
    return [
        line.strip() for line in req.open('r')
        if line.strip() and not line.strip().startswith("--")
    ]

setup(
    name='wemoms_homework',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='==3.10.*',
    install_requires=load_requirements(),
    version='0.1.0',
    description='Standalone version of wemoms homework'
)
