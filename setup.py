from setuptools import setup

APP = ['PriceBot.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': ['tkinter', 'numpy'],
    # disable codesigning (no identity on the runner)
    'codesign_identity': None,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
