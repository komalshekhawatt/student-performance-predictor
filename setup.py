from setuptools import find_packages, setup

setup(
    name='student_performance',
    version='0.0.1',
    author='Nakul',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn', 'matplotlib', 'scikit-learn']
)