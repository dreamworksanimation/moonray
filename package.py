# Copyright 2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
import os, sys

unittestflags = (['@run_all', '--unittest-xml']
                 if os.environ.get('BROKEN_CUSTOM_ARGS_UNITTESTS') else [])

name = 'moonray'

if 'early' not in locals() or not callable(early):
    def early(): return lambda x: x

@early()
def version():
    """
    Increment the build in the version.
    """
    _version = '16.36'
    from rezbuild import earlybind
    return earlybind.version(this, _version)

description = 'Moonray package'

authors = [
    'PSW Rendering and Shading',
    'moonbase-dev@dreamworks.com'
]

help = ('For assistance, '
        "please contact the folio's owner at: moonbase-dev@dreamworks.com")

variants = [
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2023.1', 'gcc-11.x', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2023.1', 'gcc-11.x', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2023.1', 'clang-17.0.6.x', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
#    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2023.1', 'clang-17.0.6.x', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
    
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.11.x'],
    
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    # ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2021.0', 'clang-13', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.8.x.2'],
]

conf_rats_variants = variants[0:2]
conf_CI_variants = list(filter(lambda v: 'os-CentOS-7' in v, variants))

requires = [
    'amorphous',
    'boost',
    'cuda-12.1.0.x',
    'embree-4.2.0.x',
    'mcrt_denoise-5.22',
    'mkl',
    'openexr',
    'openimageio-2.3.20.0.x',
    'opensubdiv-3.5.0.x.0',
    'openvdb',
    'optix-7.6.0.x',
    'random123-1.08.3',
    'scene_rdl2-14.22',
]

private_build_requires = [
    'cmake_modules-1.0',
    'cppunit',
    'ispc-1.20.0.x',
    'python-2.7|3.7|3.9|3.10'
]

commandstr = lambda i: "cd build/"+os.path.join(*variants[i])+"; ctest -j $(nproc)"
testentry = lambda i: ("variant%d" % i,
                       { "command": commandstr(i),
                         "requires": ["cmake-3.23"] + variants[i] } )
testlist = [testentry(i) for i in range(len(variants))]
tests = dict(testlist)

def commands():
    prependenv('CMAKE_MODULE_PATH', '{root}/lib64/cmake')
    prependenv('CMAKE_PREFIX_PATH', '{root}')
    prependenv('SOFTMAP_PATH', '{root}')
    prependenv('RDL2_DSO_PATH', '{root}/rdl2dso')
    prependenv('LD_LIBRARY_PATH', '{root}/lib64')
    prependenv('PATH', '{root}/bin')
    prependenv('MOONRAY_CLASS_PATH', '{root}/coredata')

uuid = '355edd2d-293f-4725-afc4-73182082debd'

config_version = 0
