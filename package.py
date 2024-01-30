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
    _version = '15.12'
    from rezbuild import earlybind
    return earlybind.version(this, _version)

description = 'Moonray package'

authors = [
    'PSW Rendering and Shading',
    'moonbase-dev@dreamworks.com'
]

help = ('For assistance, '
        "please contact the folio's owner at: moonbase-dev@dreamworks.com")

if 'scons' in sys.argv:
    build_system = 'scons'
    build_system_pbr = 'bart_scons-10'
else:
    build_system = 'cmake'
    build_system_pbr = 'cmake_modules'

variants = [
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2021.0', 'clang-13', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2020.3', 'icc-19.0.5.281.x.2', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2020.3', 'icc-19.0.5.281.x.2', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-optdebug', 'refplat-vfx2020.3', 'gcc-6.3.x.2', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-CentOS-7', 'opt_level-debug', 'refplat-vfx2020.3', 'gcc-6.3.x.2', 'amorphous-8', 'openvdb-8', 'zlib-1.2.8.x.2'],
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2021.0', 'gcc-9.3.x.1', 'amorphous-8', 'openvdb-8', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2022.0', 'gcc-9.3.x.1', 'amorphous-9', 'openvdb-9', 'imath-3', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-optdebug', 'refplat-vfx2023.0', 'gcc-11.x', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
    ['os-rocky-9', 'opt_level-debug', 'refplat-vfx2023.0', 'amorphous-9', 'openvdb-10', 'imath-3', 'zlib-1.2.11.x'],
]

conf_rats_variants = variants[0:2]
conf_CI_variants = list(filter(lambda v: 'os-CentOS-7' in v, variants))

scons_targets = ['@install_include', '@install', '--no-cache'] + unittestflags
no_unit_targets = ['@install_include', '@install', '--no-cache']
proxy_targets = ['@rdl2proxy', '@install_SConscript']

sconsTargets = {
    'refplat-vfx2020.3': scons_targets,
    'refplat-vfx2021.0': scons_targets,
    'refplat-vfx2022.0': scons_targets,
    'refplat-vfx2023.0': scons_targets,
    }

requires = [
    'amorphous',
    'boost',
    'cuda-12.1.0.x',
    'embree-4.2.0.x',
    'mcrt_denoise-4.8',
    'mkl',
    'openexr',
    'openimageio-2.3.20.0.x',
    'opensubdiv-3.5.0.x.0',
    'openvdb',
    'optix-7.6.0.x',
    'random123-1.08.3',
    'scene_rdl2-13.7',
]

private_build_requires = [
    build_system_pbr,
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

if build_system is 'cmake':
    def commands():
        prependenv('CMAKE_MODULE_PATH', '{root}/lib64/cmake')
        prependenv('CMAKE_PREFIX_PATH', '{root}')
        prependenv('SOFTMAP_PATH', '{root}')
        prependenv('RDL2_DSO_PATH', '{root}/rdl2dso')
        prependenv('LD_LIBRARY_PATH', '{root}/lib64')
        prependenv('PATH', '{root}/bin')
        prependenv('MOONRAY_CLASS_PATH', '{root}/coredata')
else:
    def commands():
        prependenv('SOFTMAP_PATH', '{root}')
        prependenv('RDL2_DSO_PATH', '{root}/rdl2dso')
        prependenv('LD_LIBRARY_PATH', '{root}/lib')
        prependenv('PATH', '{root}/bin')
        prependenv('MOONRAY_CLASS_PATH', '{root}/coredata')
    def pre_build_commands():
        env.REZ_BUILD_SCONS_ARGS.append('@install_include @install')

uuid = '355edd2d-293f-4725-afc4-73182082debd'

config_version = 0
