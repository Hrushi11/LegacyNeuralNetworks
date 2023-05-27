from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'Legacy Neural Networks'
 
# Setting up
setup(
    name="LegacyNeuralNetworks",
    version=VERSION,
    author="Hrushikesh Kachgunde",
    author_email="<hrushiskachgunde@gmail.com>",
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scikit-learn', 'tensorflow'],
    keywords=['python', 'neural networks', 'ann', 'mcculloh-pitt', 'ART neural network'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)