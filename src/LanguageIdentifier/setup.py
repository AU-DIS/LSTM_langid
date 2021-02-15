from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='LanguageIdentifier',
    version='1.1.1',
    description='Language identifier based on an LSTM model',
    long_description_content_type="text/markdown",
    packages = find_packages(),
    long_description=README,
    license='MIT',
    author='Mads Toftrup, Søren Asger Sørensen, Manuel R.Ciosici, Ira Assent',
    author_email='mabeto5p@live.dk',
    keywords=['Lid', 'LSTM', 'Language identifier',  'Language'],
    package_data = {'':['LID_mixed_model.pkl']},
    url="https://dummyUrl.com",
    download_url='https://pypi.org/project/LanguageIdentifier/',
    include_package_data = True
)

install_requires = [
    'torch'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)