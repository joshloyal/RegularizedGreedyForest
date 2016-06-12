import os
import subprocess
import sys
import contextlib

import numpy

from Cython.Compiler.Options import directive_defaults
from setuptools import Extension, setup


PACKAGES = [
    'rgforest',
    'rgforest.tests']


RGF_MODS = [
    'rgforest._tree',
    'rgforest._ensemble',
    'rgforest._builder',
    'rgforest.dataset']


OTHER_MODS = ['rgforest._memory']


RGF_DIR = 'rgfpackage'


def get_python_package(root):
    return os.path.join(root, 'rgforest')

def get_rgf_directory(root):
    return os.path.join(root, RGF_DIR)


def get_rgf_includes(root):
    non_absolutes = [
        'src/com',
        'src/tet',
        'src/tet_tools']

    rgf_dir = get_rgf_directory(root)
    return [os.path.join(rgf_dir, include) for include in non_absolutes]


def get_rgf_sources(root):
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

    rgf_dir = get_rgf_directory(root)
    return [os.path.join(rgf_dir, include) for include in non_absolutes]


def clean(path):
    for name in RGF_MODS + OTHER_MODS:
        name = name.replace('.', '/')
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
                yield os.path.join(base, filename)


def generate_cython(root, cython_cov=False):
    print("Cythonizing sources")
    for source in generate_sources(get_python_package(root)):
        cythonize_source(source, cython_cov)


def cythonize_source(source, cython_cov=False):
    print("Processing %s" % source)

    flags = ['--fast-fail', '--cplus']
    if cython_cov:
        flags.extend(['--directive', 'linetrace=True'])

    try:
        p = subprocess.call(['cython'] + flags + [source])
        if p != 0:
            raise Exception('Cython failed')
    except OSError:
        raise OSError('Cython needs to be installed')


def generate_extensions(root, macros=[]):
    ext_modules = []
    for mod_name in RGF_MODS:
        mod_path = mod_name.replace('.', '/') + '.cpp'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path] + get_rgf_sources(root),
                      include_dirs=[os.path.join(root, 'rgforest')] + get_rgf_includes(root) + [numpy.get_include()],
                      extra_compile_args=['-O3', '-fPIC'],
                      define_macros=macros,
                      language='c++'))

    for mod_name in  OTHER_MODS:
        mod_path = mod_name.replace('.', '/') + '.cpp'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=['-O3', '-fPIC'],
                      define_macros=macros,
                      language='c++'))

    return ext_modules


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    cython_cov = 'CYTHON_COV' in os.environ

    macros = []
    if cython_cov:
        print("Adding coverage information to cythonized files.")
        macros =  [('CYTHON_TRACE_NOGIL', 1)]

    with chdir(root):
        generate_cython(root, cython_cov)
        ext_modules = generate_extensions(root, macros=macros)
        setup(
            name="rgforest",
            version='0.1.0',
            description='Cython Wrapper around the Regularized Greedy Forest Algorithm',
            author='Joshua D. Loyal',
            url='https://github.com/joshloyal/RegularizedGreedyForest',
            license='MIT',
            install_requires=['numpy', 'scipy', 'scikit-learn'],
            packages=PACKAGES,
            package_data={'': ['rgforest/*.pyx', 'rgforest/*.pxd']},
            ext_modules=ext_modules,
        )


if __name__ == '__main__':
    setup_package()
