from setuptools import setup

setup(
        name='dnmfx',
        version='0.1',
        description='Distributed Stochastic Non-Negative Matrix Factorization',
        url='https://github.com/funkelab/dnmfx',
        author='Alicia Lu, Jan Funke',
        author_email='lualicia88@gmail.com, funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'dnmfx'
        ],
        install_requires=[
            'numpy',
            'jax'
        ]
)
