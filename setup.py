from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='onion_lib',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['onion_lib'],
    install_requires=required_packages,
    python_requires='>3.6.0',
    package_data={
        '': ['dataset/*/*.csv']
    },

    entry_points={
        'console_scripts': [
            'onion_lib = onion_lib.run_cli:entry_point'
        ]
    },
)
