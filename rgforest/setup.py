import os
import subprocess
import sys
import contextlib

import numpy
from distutils.core import Extension
from distutils.core import setup


PACKAGES = [
    'rgforest',
    'rgforest.tests'
]


RGF_DIR = './rgf1.2/'


RGF_MODS = [
    '_rgf',
    'dataset'
    ]


OTHER_MODS = ['_memory']


def get_rgf_includes():
    non_absolutes = [
        'src/com',
        'src/tet',
        'src/tet_tools']
    return [RGF_DIR + include for include in non_absolutes]


def get_rgf_sources():
    non_absolutes = [
	'src/tet/driv_rgf.cpp',
	'src/com/AzDmat.cpp',
	'src/tet/AzFindSplit.cpp',
	'src/com/AzIntPool.cpp',
	'src/com/AzLoss.cpp',
	'src/tet/AzOptOnTree_TreeReg.cpp',
	'src/tet/AzOptOnTree.cpp',
	'src/com/AzParam.cpp',
	'src/tet/AzReg_Tsrbase.cpp',
	'src/tet/AzReg_TsrOpt.cpp',
	'src/tet/AzReg_TsrSib.cpp',
	'src/tet/AzRgf_FindSplit_Dflt.cpp',
	'src/tet/AzRgf_FindSplit_TreeReg.cpp',
	'src/tet/AzRgf_Optimizer_Dflt.cpp',
	'src/tet/AzRgforest.cpp',
	'src/tet/AzRgfTree.cpp',
	'src/com/AzSmat.cpp',
	'src/tet/AzSortedFeat.cpp',
	'src/com/AzStrPool.cpp',
	'src/com/AzSvDataS.cpp',
	'src/com/AzTaskTools.cpp',
	'src/tet/AzTETmain.cpp',
	'src/tet/AzTETproc.cpp',
	'src/com/AzTools.cpp',
	'src/tet/AzTree.cpp',
	'src/tet/AzTreeEnsemble.cpp',
	'src/tet/AzTrTree.cpp',
	'src/tet/AzTrTreeFeat.cpp',
	'src/com/AzUtil.cpp',]

    return [RGF_DIR + include for include in non_absolutes]


def clean(path):
    for name in RGF_MODS + OTHER_MODS:
        for ext in ['.cpp', '.so']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def generate_sources(root):
    for base, _, files in os.walk(root):
        for filename in files:
            if filename.endswith('pyx'):
                yield filename


def generate_cython(root):
    print("Cythonizing sources")
    for source in generate_sources(root):
        cythonize_source(source)


def cythonize_source(source):
    print("Processing %s" % source)

    flags = ['--fast-fail', '--cplus']

    try:
        p = subprocess.call(['cython'] + flags + [source])
        if p != 0:
            raise Exception('Cython failed')
    except OSError:
        raise OSError('Cython needs to be installed')


def configuration(parent_package="", top_path=None):
    config = Configuration("rgforest", parent_package, top_path)
    config.add_extension("_rgf",
                         sources=["_rgf.pyx"] + get_rgf_sources(),
                         include_dirs=get_rgf_includes() + [numpy.get_include()],
                         extra_compile_args=['-O3', '-fPIC'],
                         language='c++')
    config.add_extension("dataset",
                         sources=["dataset.pyx"] + get_rgf_sources(),
                         include_dirs=get_rgf_includes() + [numpy.get_include()],
                         extra_compile_args=['-O3', '-fPIC'],
                         language='c++')
    config.add_extension("_memory",
                         sources=["_memory.pyx"],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-O3'],
                         language='c++')
    return config


def generate_extensions():
    ext_modules = []
    for mod_name in RGF_MODS:
        mod_path = mod_name + '.cpp'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path] + get_rgf_sources(),
                      include_dirs=get_rgf_includes() + [numpy.get_include()],
                      extra_compile_args=['-O3', '-fPIC'],
                      language='c++'))

    for mod_name in  OTHER_MODS:
        mod_path = mod_name + '.cpp'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=['-O3', '-fPIC'],
                      language='c++'))

    return ext_modules


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    with chdir(root):
        generate_cython(root)
        ext_modules = generate_extensions()
        setup(
            name="_rgf",
            packages=PACKAGES,
            package_data={'': ['*.pyx', '*.pxd']},
            ext_modules=ext_modules,
        )

if __name__ == '__main__':
    setup_package()
