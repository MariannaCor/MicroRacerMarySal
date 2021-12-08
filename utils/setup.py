from setuptools import setup, Extension

module = Extension ('lidar', sources=['C:\\Users\\Salvo\\GitHub\\MicroRacer_Corinaldesi_Fiorilla\\lidar.pyx'])

setup(
    name='cython_config',
    version='1.0',
    author='jetbrains',
    ext_modules=[module]
)