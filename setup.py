from setuptools import setup
from sde_mc.version import __version__

setup(
    name='sde_mc',
    url='https://github.com/Piers14/sde_mc',
    author='Piers Hinds',
    author_email='pmxph7@nottingham.ac.uk',
    packages=['sde_mc'],
    install_requires=['numpy', 'torch', 'scipy'],
    tests_require=['pytest'],
    version=__version__,
    license='MIT',
    description='Monte Carlo simulation for SDEs with variance reduction methods'
)
