from setuptools import setup

__version__ = '0.0.1'
url = 'https://github.com/ntt123/wavernn-16bit'

install_requires = ['optax', 'jax', 'jaxlib', 'einops', 'librosa', 'dm-haiku @ git+https://github.com/deepmind/dm-haiku', 'tqdm']
setup_requires = []
tests_require = []

setup(
    name='wavernn-16bit',
    version=__version__,
    description='an unofficial wavernn implementation',
    author='ntt123',
    url=url,
    keywords=['text-to-speech', 'tts', 'deep-learning', 'dm-haiku', 'jax', 'vocoder', 'speech-synthesis'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=['wavernn'],
    python_requires='>=3.6',
)
