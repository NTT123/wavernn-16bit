from setuptools import setup

__version__ = "0.0.3"
url = "https://github.com/ntt123/wavernn-16bit"

install_requires = [
    "dm-haiku",
    "einops",
    "jax",
    "jaxlib",
    "librosa",
    "optax",
    "tqdm",
]
setup_requires = []
tests_require = []

setup(
    name="wavernn-16bit",
    version=__version__,
    description="An unofficial wavernn implementation",
    author="ntt123",
    url=url,
    keywords=[
        "text-to-speech",
        "tts",
        "deep-learning",
        "dm-haiku",
        "jax",
        "vocoder",
        "speech-synthesis",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=["wavernn"],
    python_requires=">=3.6",
)
