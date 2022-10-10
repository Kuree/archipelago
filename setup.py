from setuptools import setup


setup(
    name='archipelago',
    packages=[
        "archipelago"
    ],
    version='0.0.10',
    author='Keyi Zhang',
    author_email='keyi@stanford.edu',
    description='Fast CGRA PnR based on thunder and cyclone',
    url="https://github.com/Kuree/archipelago",
    install_requires=[
        "pythunder",
        "pycyclone"
    ],
    data_files=[
        ('sta_delays', ['archipelago/sta_delays.json']),
    ],
    include_package_data = True
)
