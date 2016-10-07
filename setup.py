from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpyPath = numpy.get_include()
setup(
    ext_modules = cythonize([Extension("cpy_DNS_2D_Visco",
                                       ["cpy_DNS_2D_Visco.pyx",
                                        "src/DNS_2D_Visco.c", 
                                        "src/time_steppers.c",
                                        "src/fields_IO.c",
                                        "src/fields_2D.c"],
                                       include_dirs =["include",
                                                      numpyPath, "/opt/local/include"],
                                       library_dirs = ["/opt/local/lib"],
                                       libraries = ["hdf5", "hdf5_hl", "fftw3",
                                                    "m"],
                                      extra_compile_args=["-w", "-fpic"])])
)


