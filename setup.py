import io
import re
import setuptools


with open('active_subsampling/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

setuptools.setup(
    name='active_subsampling',
    version=__version__,
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'rdkit',
        'deepchem==2.5.0',
        'tensorflow',
        'matplotlib==3.1.3',
    ],
    description='Using active learning for data curation',
    long_description=long_description,
    url='https://github.com/RekerLab/active-subsampling',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)
