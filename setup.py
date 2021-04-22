from setuptools import setup

setup(
    name='sde_mc',
    url='https://github.com/Piers14/sde_mc',
    author='Piers Hinds',
    author_email='pmxph7@nottingham.ac.uk',
    packages=['sde_mc'],
    install_requires=['numpy', 'torch', 'time', 'scipy', 'abc'],
    version='0.1',
    license='MIT',
    description='Monte Carlo simulation for SDEs with variance reduction methods'
)
