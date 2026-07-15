from setuptools import setup, find_packages

setup(
    name='sraz',
    version='0.1.0',
    description='AlphaZero on grammar games: symbolic regression',
    author='',
    author_email='',
    url='',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires='>=3.10',
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
