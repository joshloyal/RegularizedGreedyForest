from distutils.core import setup, Extension
from Cython.Build import cythonize

RGF_DIR = './rgf1.2/'

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

ext = Extension("rgf", ["rgf.pyx"] + get_rgf_sources(),
                include_dirs=get_rgf_includes(),
                extra_compile_args=['-O2', '-fPIC'],
                language='c++')

setup(name="rgf",
      ext_modules=cythonize(ext))
