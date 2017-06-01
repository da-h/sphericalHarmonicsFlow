from setuptools import setup, find_packages

setup(
    name='Spherical Harmonics for Tensorflow',
    version='0.1.0',
    py_modules=['SH'],
    packages=find_packages(),
    package_data={"SH":["sh_basis.pkl"]},
    entry_points={
        'console_scripts': ['SH=SH:SH', ],
    },
    long_description=open('README.md').read(),
)
