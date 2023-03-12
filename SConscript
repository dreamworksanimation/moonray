Import('env')

import os

def defineFromENV(name):
    return eval("""('{name}', '\\\\"'+os.getenv('{name}')+'\\\\"')""".format(
        name=name))

env.AppendUnique(CPPDEFINES=[
    defineFromENV('REZ_BUILD_PROJECT_NAME'),
    defineFromENV('REZ_BUILD_PROJECT_VERSION'),
])

AddOption('--arrascodecov',
          dest='arrascodecov',
          type='string',
          action='store',
          help='Build with codecov instrumentation')

if GetOption('arrascodecov') is not None:
    env['CXXFLAGS'].append('-prof_gen:srcpos,threadsafe')
    env['CXXFLAGS'].append('-prof_dir%s' % GetOption('arrascodecov'))
    env.CacheDir(None)

env.Tool('component')
env.Tool('dwa_install')
env.Tool('dwa_run_test')
env.Tool('dwa_utils')
env['CPPCHECK_LOC'] = '/rel/third_party/cppcheck/1.71/cppcheck'
env.Tool('cppcheck')
env.Tool('python_sdk')

from dwa_sdk import DWALoadSDKs
DWALoadSDKs(env)

env.Tool('ispc_dso', toolpath = ['bart_tools'])

env.AppendUnique(ISPCFLAGS = ['--dwarf-version=2', '--wno-perf'])
# When building opt-debug, ignore ISPC performance warnings
if env['TYPE_LABEL'] == 'opt-debug':
    env.AppendUnique(ISPCFLAGS = ['--wno-perf', '--werror'])

# Suppress depication warning from tbb-2020.
env.AppendUnique(CPPDEFINES='TBB_SUPPRESS_DEPRECATED_MESSAGES')

# For SBB integration - SBB uses the keyword "restrict", so the compiler needs to enable it.
if "icc" in env['CC']:
    env['CXXFLAGS'].append('-restrict')
# Disable OpenMP use.
env.Replace(USE_OPENMP = [])

#Set optimization level for debug -O0
#icc defaults to -O2 for debug and -O3 for opt/opt-debug
if env['TYPE_LABEL'] == 'debug':
    env.AppendUnique(CCFLAGS = ['-O0'])
    if 'icc' in env['CC'] and '-inline-level=0' not in env['CXXFLAGS']:
        env['CXXFLAGS'].append('-inline-level=0')

# don't error on writes to static vars
if 'icc' in env['CC'] and '-we1711' in env['CXXFLAGS']:
    env['CXXFLAGS'].remove('-we1711')

# For Arras, we've made the decision to part with the studio standards and use #pragma once
# instead of include guards. We disable warning 1782 to avoid the related (incorrect) compiler spew.
if 'icc' in env['CC'] and '-wd1782' not in env['CXXFLAGS']:
    env['CXXFLAGS'].append('-wd1782')       # #pragma once is obsolete. Use #ifndef guard instead.

if 'icc' in env['CC']:
    env['CXXFLAGS'].append('-wd1')         # Openvdb-7 has some missing newlines
    env['CXXFLAGS'].append('-wd304')       # Openvdb-7 has some missing access control
    env['CXXFLAGS'].append('-wd1572')      # Openvdb-8 is using floating point equality
    env['CXXFLAGS'].append('-wd1684')      # conversion from pointer to same-sized integral type
    env['CXXFLAGS'].append('-wd1875')      # offsetof applied to non-POD (Plain Old Data) types is nonstandard

elif 'gcc' in env['CC']:
    # GCC - Options
    env['CXXFLAGS'].append('-fpermissive') #TODO: Is this necessary?
    env['CXXFLAGS'].append('-D__AVX__')    # O: Trying to force OIIO to let us use AVX
    env['CXXFLAGS'].append('-mavx')        # O: Covering out bases

    # GCC - Warning Suppression
    env['CXXFLAGS'].append('-Wno-sign-compare')    # W: Unsigned vs. Signed comparisons
    env['CXXFLAGS'].remove('-Wswitch')
    env['CXXFLAGS'].append('-Wno-switch')          # W: Not every case is handled explicitly
    env['CXXFLAGS'].append('-Wno-conversion')      # W: Conversion may alter value
    env['CXXFLAGS'].append('-w')

elif 'clang' in env['CC']:
    env['CXXFLAGS'].append('-Wno-switch')          # W: Not every case is handled explicitly
    env['CXXFLAGS'].append('-Wno-return-stack-address')
                                                   # W: address of stack memory associated with local variable
    env['CXXFLAGS'].append('-Wno-uninitialized')   # W: field 'mWeights' is uninitialized when used here
    env['CXXFLAGS'].append('-Wno-static-float-init') # E: In-class initializer for static data member

env.TreatAllWarningsAsErrors()

# Create include/moonray link to ../lib in the build directory.
Execute("mkdir -p include")
Execute("rm -f include/moonray")
Execute("ln -sfv ../moonray include/moonray")
#Execute("ln -sfv lib moonray")
env.Append(CPPPATH=[env.Dir('#')])
env.Append(CPPPATH=[env.Dir('#include')])
env.Append(CPPPATH=[env.Dir('$INSTALL_DIR/include')])
env.Append(CPPPATH=[env.Dir('include')])
env.Append(ISPC_CPPPATH=[env.Dir('include')])

# Scan our source for SConscripts and parse them.
#env.DWASConscriptWalk(topdir='#lib', ignore=['lib/common/mcrt_comp', 'lib/common/rec_load'])
env.DWASConscriptWalk(topdir='#moonray', ignore=['moonray/common/mcrt_comp', 'moonray/common/rec_load'])
env.DWASConscriptWalk(topdir='#cmd', ignore=[])
env.DWASConscriptWalk(topdir='#dso', ignore=['dso/computation'])
env.DWASConscriptWalk(topdir='#doc', ignore=[])
env.DWASConscriptWalk(topdir='#tests', ignore=[])

# Install third party stubs that we need, so our users can use them too.
env.DWAInstallSConscripts('#SConscripts')

# Install our SDK for our users to use.
SDKScript = env.DWAInstallSDKScript('#SDKScript')
env.Alias('@rdl2proxy', SDKScript)

# Install the bart tools that we provide so our users can use them too.
bart_tools = env.DWAInstallFiles(
    'bart', ['#bart_tools/%s.py' % name for name in ('ispc_dso',)])
env.Alias('@rdl2proxy', bart_tools)
env.NoCache(bart_tools)

# Pass in the location of our stubs, so that those third party components will be resolved.
env.DWAResolveUndefinedComponents(['#SConscripts'])

env.DWAFillInMissingInitPy()
env.DWAFreezeComponents()

# Set default target
env.Default([env.Alias('@install_include'), env.Alias('@install')])

# @cppcheck target
env.DWACppcheck(env.Dir('.'), [
        ".git",
        "build",
        "build_env",
        "install",
        "include",
        "third_party",
        "bamboo_tools",
        "bart_tools",
        "boot",
        "doc",
        "jsoncpp",
        "lib/rendering/geom/opensubdiv",
        "lib/rendering/geom/unittest",
        "lib/rendering/rt/kernels",
        "site_tools",
        "lib/deepfile",
        "lib/statistics"
        ])

# Add the @cppcheck aliases targets to the @run_all alias so that they are build along with the unittests.
env.Alias('@run_all', env.Alias('@cppcheck'))
