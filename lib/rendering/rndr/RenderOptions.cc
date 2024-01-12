// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "RenderOptions.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/render/util/Args.h>
#include <scene_rdl2/render/util/Files.h>
#include <scene_rdl2/render/util/Strings.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <tbb/task_scheduler_init.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdint.h>

#ifndef MOONRAY_EXEC_MODE_DEFAULT
#define MOONRAY_EXEC_MODE_DEFAULT AUTO
#endif

// ---------------------------------------------------------------
// These macros are used in the getUsageMessage() function below
// to print "(default)" next to the default exec_mode's description
#define AUTO        0
#define SCALAR      1
#define VECTORIZED  2
#define XPU         3

#define DEFAULT_EXEC_MODE_STR_AUTO            ""
#define DEFAULT_EXEC_MODE_STR_SCALAR          ""
#define DEFAULT_EXEC_MODE_STR_VECTORIZED      ""
#define DEFAULT_EXEC_MODE_STR_XPU             ""

#if     MOONRAY_EXEC_MODE_DEFAULT == AUTO
#undef  DEFAULT_EXEC_MODE_STR_AUTO
#define DEFAULT_EXEC_MODE_STR_AUTO          "(default) "
#elif   MOONRAY_EXEC_MODE_DEFAULT == SCALAR
#undef  DEFAULT_EXEC_MODE_STR_SCALAR
#define DEFAULT_EXEC_MODE_STR_SCALAR        "(default) "
#elif   MOONRAY_EXEC_MODE_DEFAULT == VECTORIZED
#undef  DEFAULT_EXEC_MODE_STR_VECTORIZED
#define DEFAULT_EXEC_MODE_STR_VECTORIZED    "(default) "
#elif   MOONRAY_EXEC_MODE_DEFAULT == XPU
#undef  DEFAULT_EXEC_MODE_STR_XPU
#define DEFAULT_EXEC_MODE_STR_XPU           "(default) "
#endif

#undef AUTO
#undef SCALAR
#undef VECTORIZED
#undef XPU
// ---------------------------------------------------------------

namespace moonray {
namespace rndr {

using namespace scene_rdl2::math;
using namespace scene_rdl2;
using scene_rdl2::util::Args;
using scene_rdl2::util::stringToUnsignedLong;
using scene_rdl2::util::stringToInt;
using scene_rdl2::util::stringToLong;
using scene_rdl2::util::stringToFloat;
using scene_rdl2::util::stringToBool;

namespace {

void
addOverride(std::vector<RenderOptions::AttributeOverride>& overrides,
            const std::string& object, const std::string& attribute,
            const std::string& value, const std::string& binding,
            bool isSceneVar = false)
{
    // Look for an existing override of the same object/attribute.
    auto iter = std::find_if(overrides.begin(), overrides.end(),
    [&object, &attribute](const RenderOptions::AttributeOverride& attrOverride) {
        return attrOverride.mObject == object &&
               attrOverride.mAttribute == attribute;
    });

    // If we didn't find one, create it.
    if (iter == overrides.end()) {
        overrides.push_back(RenderOptions::AttributeOverride(object, attribute, isSceneVar));
        iter = --overrides.end();
    }

    // Set the value, if provided.
    if (!value.empty()) {
        iter->mValue = value;
    }

    // Set the binding, if provided.
    if (!binding.empty()) {
        iter->mBinding = binding;
    }
}

} // namespace

RenderOptions::RenderOptions() :
    // These are not defaults. They are "unset" values. If neither the command
    // line nor the SceneContext provides a suitable value for these, we use a
    // last-ditch default value that is set in the respective getter function.
    mSceneContext(nullptr),
    mThreads(0),
    mRenderMode(RenderMode::PROGRESSIVE),
    mFastMode(FastRenderMode::NORMALS),
    mGeneratePixelInfo(false),
    mRes(0.0f),
    mSceneFiles(),
    mDsoPath(""),
    mTextureCacheSizeMb(0),
    mAttributeOverrides(),
    mRdlaGlobals(),
    mCommandLine(""),
    // These are defaults.
    mExecutionMode(mcrt_common::ExecutionMode::MOONRAY_EXEC_MODE_DEFAULT),
    mTileProgress(true),
    mApplicationMode(ApplicationMode::UNDEFINED),
    mApplyColorRenderTransform(false),
    mAthenaTagsString(""),
    mGUID(scene_rdl2::util::GUID::nil()),
    mDebugConsolePort(-1), // negative value means debug console = off
    // For developer profiling purposes only.
    mHasVectorizedOverrides(false),
    mPerThreadRayStatePoolSize(0),
    mRayQueueSize(0),
    mOcclusionQueueSize(0),
    mPresenceShadowsQueueSize(0),
    mShadeQueueSize(0),
    mRadianceQueueSize(0),
    mShadingWorkloadChunkSize(32),
    mFps(0.0f)                  // 0.0 means not defined.
{
    parserConfigure();
}

RenderOptions::~RenderOptions()
{
}

void
RenderOptions::parseFromCommandLine(int argc, char* argv[])
{
    Args args(argc, argv);
    Args::StringArray values;
    std::vector<std::string> validFlags;

    //  instead of changing Args class to access its data we will just
    // consume the command line arguments and rebuild the string here
    for( int i = 0; i < argc; i++) {
        mCommandLine += argv[i];
        mCommandLine += " ";
    }

    {
        std::vector<std::string> sceneFiles;

        int foundAtIndex = args.getFlagValues("-in", -1 /*get all filenames*/, values);
        validFlags.push_back("-in");
        while (foundAtIndex >= 0) {
            for (const std::string& v : values) {
                sceneFiles.push_back(v);
            }
            foundAtIndex = args.getFlagValues("-in", -1 /*get all filenames*/, values, foundAtIndex + 1);
        }

        if (!sceneFiles.empty()) {
            setSceneFiles(std::move(sceneFiles));
        }
    }

    {
        std::vector<std::string> deltasFiles;
        int foundAtIndex = args.getFlagValues("-deltas", -1 /*get all filenames*/, values);
        validFlags.push_back("-deltas");
        while (foundAtIndex >= 0) {
            for (const std::string& v : values) {
                deltasFiles.push_back(v);
            }
            foundAtIndex =
                args.getFlagValues("-deltas", -1 /*get all filenames*/, values, foundAtIndex + 1);
        }
        if (!deltasFiles.empty()) setDeltasFiles(std::move(deltasFiles));
    }

    validFlags.push_back("-threads");
    if (args.getFlagValues("-threads", 1, values) >= 0) {
        setThreads(stringToUnsignedLong(values[0]));
    }

    validFlags.push_back("-dso_path");
    if (args.getFlagValues("-dso_path", 1, values) >= 0) {
        setDsoPath(values[0]);
    }

    validFlags.push_back("-exec_mode");
    if (args.getFlagValues("-exec_mode", 1, values) >= 0) {
        setDesiredExecutionMode(values[0]);
    }

    {
        validFlags.push_back("-no_tile_progress");
        const bool noTileProgress = (args.getFlagValues("-no_tile_progress", 0, values) >= 0);
        if (noTileProgress) {
            setTileProgress(false);
        }
    }

    {
        validFlags.push_back("-apply_crt");
        const bool applyColorRenderTransform = (args.getFlagValues("-apply_crt", 0, values) >= 0);
        if (applyColorRenderTransform) {
            setApplyColorRenderTransform(true);
        }
    }

    {
        validFlags.push_back("-snap_path");
        const bool snapshotPathExists = (args.getFlagValues("-snap_path", 1, values) >= 0);
        if (snapshotPathExists) {
            setSnapshotPath(values[0]);
        }
    }

    validFlags.push_back("-override_lut");
    if (args.getFlagValues("-override_lut", 1, values) >= 0) {
        setColorRenderTransformOverrideLut(values[0]);
    }

    {
        std::vector<RdlaGlobal> rdlaGlobals;

        validFlags.push_back("-rdla_set");
        int foundAtIndex = args.getFlagValues("-rdla_set", 2, values);
        while (foundAtIndex >= 0) {
            rdlaGlobals.emplace_back(values[0], values[1]);
            foundAtIndex = args.getFlagValues("-rdla_set", 2, values, foundAtIndex + 1);
        }

        if (!rdlaGlobals.empty()) {
            setRdlaGlobals(std::move(rdlaGlobals));
        }
    }

    {
        std::vector<AttributeOverride> overrides;


        int foundAtIndex = args.getFlagValues("-scene_var", 2, values);
        validFlags.push_back("-scene_var");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", values[0], values[1], "", true);
            foundAtIndex = args.getFlagValues("-scene_var", 2, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-attr_set", 3, values);
        validFlags.push_back("-attr_set");
        while (foundAtIndex >= 0) {
            addOverride(overrides, values[0], values[1], values[2], "", true);
            foundAtIndex = args.getFlagValues("-attr_set", 3, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-attr_bind", 3, values);
        validFlags.push_back("-attr_bind");
        while (foundAtIndex >= 0) {
            addOverride(overrides, values[0], values[1], "", values[2], true);
            foundAtIndex = args.getFlagValues("-attr_bind", 3, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-out", 1, values);
        validFlags.push_back("-out");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "output file", values[0], "", true);
            foundAtIndex = args.getFlagValues("-out", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-info", 0, values);
        validFlags.push_back("-info");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "info", "true", "", true);
            foundAtIndex = args.getFlagValues("-info", 0, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-debug", 0, values);
        validFlags.push_back("-debug");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "debug", "true", "", true);
            foundAtIndex = args.getFlagValues("-debug", 0, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-stats", 1, values);
        validFlags.push_back("-stats");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "stats file", values[0], "", true);
            foundAtIndex = args.getFlagValues("-stats", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-size", 2, values);
        validFlags.push_back("-size");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "frame width", values[0], "", true);
            addOverride(overrides, "__SceneVariables__", "frame height", values[1], "", true);
            foundAtIndex = args.getFlagValues("-size", 2, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-res", 1, values);
        validFlags.push_back("-res");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "res", values[0], "", true);
            foundAtIndex = args.getFlagValues("-res", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-camera", 1, values);
        validFlags.push_back("-camera");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "camera name", values[0], "", true);
            foundAtIndex = args.getFlagValues("-camera", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-layer", 1, values);
        validFlags.push_back("-layer");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "layer name", values[0], "", true);
            foundAtIndex = args.getFlagValues("-layer", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-sub_viewport", 4, values);
        validFlags.push_back("-sub_viewport");
        while (foundAtIndex >= 0) {
            std::string viewportString = values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3];
            addOverride(overrides, "__SceneVariables__", "sub viewport", viewportString, "", true);
            foundAtIndex = args.getFlagValues("-sub_viewport", 4, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-sub_vp", 4, values);
        validFlags.push_back("-sub_vp");
        while (foundAtIndex >= 0) {
            std::string viewportString = values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3];
            addOverride(overrides, "__SceneVariables__", "sub viewport", viewportString, "", true);
            foundAtIndex = args.getFlagValues("-sub_vp", 4, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-record_rays", 1, values);
        validFlags.push_back("-record_rays");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "debug rays file", values[0], "", true);
            foundAtIndex = args.getFlagValues("-record_rays", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-primary_range", -1, values);
        validFlags.push_back("-primary_range");
        while (foundAtIndex >= 0) {
            if (values.size() == 1) {
                addOverride(overrides, "__SceneVariables__", "debug rays primary range", values[0], "", true);
            } else if (values.size() == 2) {
                addOverride(overrides, "__SceneVariables__", "debug rays primary range", values[0] + ", " + values[1], "", true);
            }
            foundAtIndex = args.getFlagValues("-primary_range", -1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-depth_range", -1, values);
        validFlags.push_back("-depth_range");
        while (foundAtIndex >= 0) {
            if (values.size() == 1) {
                addOverride(overrides, "__SceneVariables__", "debug rays depth range", values[0], "", true);
            } else if (values.size() == 2) {
                addOverride(overrides, "__SceneVariables__", "debug rays depth range", values[0] + ", " + values[1], "", true);
            }
            foundAtIndex = args.getFlagValues("-depth_range", -1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-debug_pixel", 2, values);
        validFlags.push_back("-debug_pixel");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "debug pixel", values[0] + ", " + values[1], "", true);
            foundAtIndex = args.getFlagValues("-debug_pixel", 2, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-fast_geometry_update", 1, values);
        validFlags.push_back("-fast_geometry_update");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "fast geometry update", values[0], "", true);
            foundAtIndex = args.getFlagValues("-fast_geometry_update", 1, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-checkpoint", 0, values);
        validFlags.push_back("-checkpoint");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "checkpoint_active", "true", "",  true);
            foundAtIndex = args.getFlagValues("-checkpoint", 0, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-resumable_output", 0, values);
        validFlags.push_back("-resumable_output");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "resumable_output", "true", "", true);
            foundAtIndex = args.getFlagValues("-resumable_output", 0, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-resume_render", 0, values);
        validFlags.push_back("-resume_render");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "resume_render", "true", "", true);
            foundAtIndex = args.getFlagValues("-resume_render", 0, values, foundAtIndex + 1);
        }

        foundAtIndex = args.getFlagValues("-debug_console", 1, values);
        validFlags.push_back("-debug_console");
        while (foundAtIndex >= 0) {
            addOverride(overrides, "__SceneVariables__", "debug_console", values[0], "", true);
            foundAtIndex = args.getFlagValues("-debug_console", 1, values, foundAtIndex + 1);
        }

        if (!overrides.empty()) {
            setAttributeOverrides(overrides);
        }
    }

    validFlags.push_back("-athena_tags");
    if (args.getFlagValues("-athena_tags", 1, values) >= 0) {
        setAthenaTags(values[0]);
    }

    validFlags.push_back("-guid");
    if (args.getFlagValues("-guid", 1, values) >= 0) {
        setGUID(scene_rdl2::util::GUID(values[0]));
    }

    //
    // For developer profiling only, not documented or exposed to user.
    // Use like so:
    //
    //  -set_vectorized_overrides n1 n2 n3 n4 n5 n6 n7 n8 n9
    //
    // where:
    //  n1 = a float containing the perThreadRayStatePoolSize / 1024
    //  n2 = a float containing the primaryRayQueueSize / 1024
    //  n3 = unused
    //  n4 = a float containing the occlusionQueueSize / 1024
    //  n5 = a float containing the shadeQueueSize / 1024
    //  n6 = a float containing the radianceQueueSize / 1024
    //  n7 = a float containing the shadingWorkloadChunkSize / 1024
    //  n8 = a float containing the presenceShadowsQueueSize / 1024
    //
    // For example, to mimic the defaults you'd use:
    //
    //  -set_vectorized_overrides 64 0.5 1 1 0.125 0.5 0.03125 1
    //
    mHasVectorizedOverrides = false;
    validFlags.push_back("-set_vectorized_overrides");
    if (args.getFlagValues("-set_vectorized_overrides", 8, values) >= 0) {
        mHasVectorizedOverrides = true;
        float perThreadRayStatePoolSize = std::stof(values[0]);
        float rayQueueSize              = std::stof(values[1]);
        float occlusionQueueSize        = std::stof(values[3]);
        float shadeQueueSize            = std::stof(values[4]);
        float radianceQueueSize         = std::stof(values[5]);
        float shadingWorkloadChunkSize  = std::stof(values[6]);
        float presenceShadowsQueueSize  = std::stof(values[7]);

        mPerThreadRayStatePoolSize = unsigned(perThreadRayStatePoolSize * 1024.f);
        mRayQueueSize              = unsigned(rayQueueSize              * 1024.f);
        mOcclusionQueueSize        = unsigned(occlusionQueueSize        * 1024.f);
        mShadeQueueSize            = unsigned(shadeQueueSize            * 1024.f);
        mRadianceQueueSize         = unsigned(radianceQueueSize         * 1024.f);
        mShadingWorkloadChunkSize  = unsigned(shadingWorkloadChunkSize  * 1024.f);
        mPresenceShadowsQueueSize  = unsigned(presenceShadowsQueueSize  * 1024.f);
    }

    if ( ! args.allFlagsValid( validFlags ) ) {
        fprintf(stderr, "Invalid Input Flag Found!  Exiting %s...\n", argv[0]);
        exit(-1);
    }
}

void
RenderOptions::setSceneContext(const scene_rdl2::rdl2::SceneContext* sceneContext)
{
    mSceneContext = sceneContext;
}

std::string
RenderOptions::getUsageMessage(const std::string& programName, bool guiMode)
{
    return scene_rdl2::util::buildString(
"Usage: ", programName, " [options]\n"
"Options:\n"
"    -help\n"
"        Print this message.\n"
"\n"
"    -in scene.rdl{a|b}\n"
"        Input RDL scene data. May appear more than once. Processes multiple\n"
"        files in order.\n",

(guiMode ?
"        Reapplies all files whenever any one changes.\n"
:
"\n"),

"    -deltas file.rdl{a|b}\n"
"        Updates to apply to RDL scene data. May appear more than once.\n",

(guiMode ?
"        Applies deltas from a particular file whenever it changes.\n"
:
"        First renders without deltas and outputs the image. Then applies each\n"
"        delta file in order, outputting an image between each one.\n"
),

"\n"
"    -out scene.exr\n"
"        Output image name and type.\n"
"\n"
"    -threads n\n"
"        Number of threads to use (all by default).\n"
"\n"
"    -size 1920 1080\n"
"        Canonical frame width and height (in pixels).\n"
"\n"
"    -res 1.0\n"
"        Resolution divisor for frame dimensions.\n"
"\n"
"    -exec_mode mode\n"
"        Choose a specific mode of execution. Valid options are:\n"
"        scalar     - " DEFAULT_EXEC_MODE_STR_SCALAR        "run in scalar mode.\n"
"        vectorized - " DEFAULT_EXEC_MODE_STR_VECTORIZED    "always run vectorized even if some features are unsupported.\n"
"        xpu        - " DEFAULT_EXEC_MODE_STR_XPU           "run in xpu mode.\n"
"        auto       - " DEFAULT_EXEC_MODE_STR_AUTO          "attempt to run xpu mode, the fallback to vectorized mode, then fallback to scalar if some features are unsupported.\n"
"\n"
"    -sub_viewport l b r t\n"
"    -sub_vp       l b r t\n"
"        Clamp viewport render region.\n"
"\n"
"    -debug_pixel x y\n"
"        Only render this one pixel for debugging. Overrides viewport.\n"
"\n"
"    -dso_path dso/path\n"
"        Prepend to search path for RDL DSOs.\n"
"\n"
"    -camera camera\n"
"        Camera to render from.\n"
"\n"
"    -layer layer\n"
"        Layer to render from.\n"
"\n"
"    -fast_geometry_update\n"
"        Turn on supporting fast geometry update for animation.\n"
"\n"
"    -record_rays .raydb/.mm\n"
"        Save ray database or mm for later debugging.\n"
"\n"
"    -primary_range 0 [0]\n"
"        Start and end range of primary ray(s) to debug. Only active with\n"
"        -record_rays.\n"
"\n"
"    -depth_range 0 [0]\n"
"        Start and end range of ray depths to debug. Only active with\n"
"        -record_rays.\n"
"\n"
"    -rdla_set \"var name\" \"expression\"\n"
"        Sets a global variable in the Lua interpreter before any RDLA is\n"
"        executed.\n"
"\n"
"    -scene_var \"name\" \"value\"\n"
"        Override a specific scene variable.\n"
"\n"
"    -attr_set \"object\" \"attribute name\" \"value\"\n"
"        Override the value of an attribute on a specific SceneObject.\n"
"\n"
"    -attr_bind \"object\" \"attribute name\" \"bound object name\"\n"
"        Override the binding on an attribute of a specific SceneObject.\n"
"\n"
"    -info\n"
"        Enable verbose progress and statistics logging on stdout.\n"
"\n"
"    -debug\n"
"        Enable debug level logging on stdout.\n"
"\n"
"    -stats filename.csv\n"
"        Enable logging of statistics to a formatted file.\n"
"\n"
"    -athena_tags \"TAG1=VALUE1 TAG2=VALUE2 ... TAGN=VALUEN\" \n"
"        Provided string will be sent to Athena Log Server and can be used to access stats on this render\n"
"        Intended to be used for storing user specific data of interest such as RATS tests, testmaps, etc\n"
"        TAG and VALUES are entirely up to the user\n"
"\n"
"    -resume_render\n"
"        activate both of resume render and checkpoint render\n"
"\n"
"    -resumable_output\n" 
"        Make aov output as resumable for resume render\n",

(!guiMode ?
"\n" 
"    -checkpoint\n"
"        Enable progress checkpoint rendering mode\n"
:
 ""),

(guiMode ?
"\n"
"    -free_cam\n"
"        Use a WASD FPS style camera when in interactive mode (defaults to\n"
"        orbit camera).\n"
"\n"
"    -no_tile_progress\n"
"        Turn off the diagnostic tile outlines rendered on top of the image when in gui mode.\n"
"\n"
"    -apply_crt\n"
"        Apply color render transform. The default LUT is used if no override is specified.\n"
"\n"
"    -snap_path <path>\n"
"        Specify a file path for render snapshots.\n"
"\n"
"    -override_lut\n"
"        Path to a binary file containing a 64*64*64*RGBfloat OpenGL compatible volume texture.\n"
"\n"
"    -debug_console <port>\n" 
"        Activate debug console port for telnet connection.\n"
"        (port=0 : auto search available port by kernel. result port shows as stderr message of moonray_gui)\n"
:
"")
    );
}

uint32_t
RenderOptions::getThreads() const
{
    // Command line override takes priority.
    if (mThreads != 0) {
        return mThreads;
    }

    // Last-ditch default.
    return tbb::task_scheduler_init::default_num_threads();
}

void
RenderOptions::setThreads(uint32_t threads)
{
    mThreads = threads;
}

const std::vector<std::string>&
RenderOptions::getSceneFiles() const
{
    return mSceneFiles;
}

void
RenderOptions::setSceneFiles(std::vector<std::string> sceneFiles)
{
    mSceneFiles = std::move(sceneFiles);
}

const std::vector<std::string> &
RenderOptions::getDeltasFiles() const
{
    return mDeltasFiles;
}

void
RenderOptions::setDeltasFiles(std::vector<std::string> deltasFiles)
{
    mDeltasFiles = std::move(deltasFiles);
}

std::string
RenderOptions::getDsoPath() const
{
    // Command line override takes priority.
    if (!mDsoPath.empty()) {
        return mDsoPath;
    }

    // Last-ditch default.
    return "";
}

void
RenderOptions::setDsoPath(const std::string& dsoPath)
{
    mDsoPath = dsoPath;
}

std::vector<RenderOptions::AttributeOverride>
RenderOptions::getAttributeOverrides() const
{
    return mAttributeOverrides;
}

void
RenderOptions::setAttributeOverrides(const std::vector<AttributeOverride>& overrides)
{
    mAttributeOverrides = overrides;
}

const std::vector<RenderOptions::RdlaGlobal>&
RenderOptions::getRdlaGlobals() const
{
    return mRdlaGlobals;
}

void
RenderOptions::setRdlaGlobals(std::vector<RdlaGlobal> rdlaGlobals)
{
    mRdlaGlobals = std::move(rdlaGlobals);
}

void
RenderOptions::setDesiredExecutionMode(const std::string &execMode)
{
    mcrt_common::ExecutionMode mode = mExecutionMode;

    if (execMode == "auto") {
        mode = mcrt_common::ExecutionMode::AUTO;
    } else if (execMode == "vector" || execMode == "vectorized") {
        mode = mcrt_common::ExecutionMode::VECTORIZED;
    } else if (execMode == "scalar") {
        mode = mcrt_common::ExecutionMode::SCALAR;
    } else if (execMode == "xpu") {
        mode = mcrt_common::ExecutionMode::XPU;
    } else {
        std::stringstream errMsg;
        errMsg << "Unexpected string passed to setDesiredExecutionMode(): '" << execMode << "'!";
        throw scene_rdl2::except::ValueError(errMsg.str());
    }

    // Call out to non-string overload.
    setDesiredExecutionMode(mode);
}

void
RenderOptions::setFastGeometry()
{
    // not the cleanest but lets see if this fixes our segfaul..
    std::vector<AttributeOverride> overrides;
    addOverride(overrides, "__SceneVariables__", "fast geometry update", "true", "", true);
    if (!overrides.empty()) {
        setAttributeOverrides(overrides);
    }
}

void
RenderOptions::setupTLSInitParams(mcrt_common::TLSInitParams *params, bool realtimeRender) const
{
    // Init baseline parameters.
    params->setVectorizedDefaults(realtimeRender);

    if (mHasVectorizedOverrides) {
        params->mPerThreadRayStatePoolSize = mPerThreadRayStatePoolSize;
        params->mRayQueueSize              = mRayQueueSize;
        params->mOcclusionQueueSize        = mOcclusionQueueSize;
        params->mPresenceShadowsQueueSize  = mPresenceShadowsQueueSize;
        params->mShadeQueueSize            = mShadeQueueSize;
        params->mRadianceQueueSize         = mRadianceQueueSize;
    }

    scene_rdl2::logging::Logger::info("Setting mPerThreadRayStatePoolSize to ", params->mPerThreadRayStatePoolSize);
    scene_rdl2::logging::Logger::info("Setting mRayQueueSize to ", params->mRayQueueSize);
    scene_rdl2::logging::Logger::info("Setting mOcclusionQueueSize to ", params->mOcclusionQueueSize);
    scene_rdl2::logging::Logger::info("Setting mShadeQueueSize to ", params->mShadeQueueSize);
    scene_rdl2::logging::Logger::info("Setting mRadianceQueueSize to ", params->mRadianceQueueSize);
    scene_rdl2::logging::Logger::info("Setting mShadingWorkloadChunkSize to ", mShadingWorkloadChunkSize);
    scene_rdl2::logging::Logger::info("Setting mPresenceShadowsQueueSize to ", params->mPresenceShadowsQueueSize);

    // always assign these three
    params->mDesiredNumTBBThreads = getThreads();
}

std::string
RenderOptions::show() const
{
    auto showRenderMode = [](const RenderMode &mode) -> std::string {
        switch (mode) {
        case RenderMode::BATCH : return "BATCH";
        case RenderMode::PROGRESSIVE : return "PROGRESSIVE";
        case RenderMode::PROGRESSIVE_FAST : return "PROGRESSIVE_FAST";
        case RenderMode::REALTIME : return "REALTIME";
        case RenderMode::PROGRESS_CHECKPOINT : return "PROGRESS_CHECKPOINT";
        default : break;
        }
        return "?";
    };
    auto showFastMode = [](const FastRenderMode &mode) -> std::string {
        switch (mode) {
        case FastRenderMode::NORMALS : return "NORMALS";
        case FastRenderMode::NORMALS_SHADING : return "NORMALS_SHADING";
        case FastRenderMode::FACING_RATIO : return "FACING_RATIO";
        case FastRenderMode::FACING_RATIO_INVERSE : return "FACING_RATIO_INVERSE";
        case FastRenderMode::UVS : return "UVS";
        default : break;
        }
        return "?";
    };
    auto showBool = [](const bool flag) -> std::string {
        return (flag) ? "true" : "false";
    };
    auto showVectorString = [](const std::string &msg, std::vector<std::string> vec) -> std::string {
        std::ostringstream ostr;
        ostr << msg << " (size:" << vec.size() << ")";
        if (vec.size() > 0) {
            ostr << " {\n";
            int w = std::to_string(vec.size()).length();
            for (size_t i = 0; i < vec.size(); ++i) {
                ostr << "  i:" << std::setw(w) << ' ' << vec[i] << '\n';
            }
            ostr << "}";
        }
        return ostr.str();
    };
    auto showAttributeOverrides = [](const std::vector<AttributeOverride> &vec) -> std::string {
        auto showAttr = [](const AttributeOverride &attr) -> std::string {
            std::ostringstream ostr;
            ostr
            << "AttributeOverride {\n"
            << "  mObject:" << attr.mObject << '\n'
            << "  mAttribute:" << attr.mAttribute << '\n'
            << "  mValue:" << attr.mValue << '\n'
            << "  mBinding:" << attr.mBinding << '\n'
            << "  mIsSceneVar:" << ((attr.mIsSceneVar) ? "true" : "false") << '\n'
            << "}";
            return ostr.str();
        };
        std::ostringstream ostr;
        ostr << "AttributeOverrides (size:" << vec.size() << ")";
        if (vec.size() > 0) {
            ostr << " {\n";
            for (size_t i = 0; i < vec.size(); ++i) {
                ostr << scene_rdl2::str_util::addIndent("i:" + std::to_string(i) + ' ' + showAttr(vec[i])) << '\n';
            }
            ostr << "}";
        }
        return ostr.str();
    };
    auto showRdlaGlobals = [](const std::vector<RdlaGlobal> &vec) -> std::string {
        auto showRdlaGlobal = [](const RdlaGlobal &rdla) -> std::string {
            std::ostringstream ostr;
            ostr
            << "RdlaGlobal {\n"
            << "  mVar:" << rdla.mVar << '\n'
            << "  mExpression:" << rdla.mExpression << '\n'
            << "}";
            return ostr.str();
        };
        std::ostringstream ostr;
        ostr << "RdlaGlobals (size:" << vec.size() << ")";
        if (vec.size() > 0) {
            ostr << " {\n";
            for (size_t i = 0; i < vec.size(); ++i) {
                ostr << scene_rdl2::str_util::addIndent("i:" + std::to_string(i) + ' ' + showRdlaGlobal(vec[i])) << '\n';
            }
            ostr << "}";
        }
        return ostr.str();
    };
    auto showExecutionMode = [](const mcrt_common::ExecutionMode &mode) -> std::string {
        switch (mode) {
        case mcrt_common::ExecutionMode::AUTO : return "AUTO";
        case mcrt_common::ExecutionMode::VECTORIZED : return "VECTORIZED";
        case mcrt_common::ExecutionMode::SCALAR : return "SCALAR";
        case mcrt_common::ExecutionMode::XPU : return "XPU";
        default : break;
        }
        return "?";
    };
    auto showApplicationMode = [](const ApplicationMode &mode) -> std::string {
        switch (mode) {
        case ApplicationMode::UNDEFINED : return "UNDEFINED";
        case ApplicationMode::MOTIONCAPTURE : return "MOTIONCAPTURE";
        case ApplicationMode::BEARDS : return "BEARDS";
        case ApplicationMode::VR : return "VR";
        default : break;
        }
        return "?";
    };

    std::ostringstream ostr;
    ostr << "RenderOptions {\n"
         << "  mSceneContext:0x" << std::hex << (uintptr_t)mSceneContext << std::dec << '\n'
         << "  mThreads:" << mThreads << '\n'
         << "  mRenderMode:" << showRenderMode(mRenderMode) << '\n'
         << "  mFastMode:" << showFastMode(mFastMode) << '\n'
         << "  mGeneratePixelInfo:" << showBool(mGeneratePixelInfo) << '\n'
         << "  mRes:" << mRes << '\n'
         << scene_rdl2::str_util::addIndent(showVectorString("mSceneFiles", mSceneFiles)) << '\n'
         << scene_rdl2::str_util::addIndent(showVectorString("mDeltasFiles", mDeltasFiles)) << '\n'
         << "  mDsoPath:" << mDsoPath << '\n'
         << "  mTextureCacheSizeMb:" << mTextureCacheSizeMb << '\n'
         << scene_rdl2::str_util::addIndent(showAttributeOverrides(mAttributeOverrides)) << '\n'
         << scene_rdl2::str_util::addIndent(showRdlaGlobals(mRdlaGlobals)) << '\n'
         << "  mCommandLine:" << mCommandLine << '\n'
         << "  mExecutionMode:" << showExecutionMode(mExecutionMode) << '\n'
         << "  mTileProgress:" << ((mTileProgress) ? "true" : "false") << '\n'
         << "  mApplicationMode:" << showApplicationMode(mApplicationMode) << '\n'
         << "  mApplyColorRenderTransform:" << ((mApplyColorRenderTransform) ? "true" : "false") << '\n'
         << "  mSnapshotPath:" << mSnapshotPath << '\n'
         << "  mColorRenderTransformOverrideLut:" << mColorRenderTransformOverrideLut << '\n'
         << "  mAttenaTagsString:" << mAthenaTagsString << '\n'
         << "  mGUID:" << mGUID.asString() << '\n'
         << "  mDebugConsolePort:" << mDebugConsolePort << '\n'
         << "  mHasVectorizedOverrides:" << ((mHasVectorizedOverrides) ? "true" : "false") << '\n'
         << "  mPerThreadRayStatePoolSize:" <<  mPerThreadRayStatePoolSize << '\n'
         << "  mRayQueueSize:" << mRayQueueSize << '\n'
         << "  mOcclusionQueueSize:" << mOcclusionQueueSize << '\n'
         << "  mPresenceShadowsQueueSize:" << mPresenceShadowsQueueSize << '\n'
         << "  mShadeQueueSize:" << mShadeQueueSize << '\n'
         << "  mRadianceQueueSize:" << mRadianceQueueSize << '\n'
         << "  mShadingWorkloadChunkSize:" << mShadingWorkloadChunkSize << '\n'
         << "  mFps:" << mFps << '\n'
         << "}";
    return ostr.str();
}

void
RenderOptions::parserConfigure()
{
    using scene_rdl2::str_util::byteStr;

    mParser.description("renderOptions command");
    mParser.opt("show", "", "show all information", [&](Arg& arg) -> bool { return arg.msg(show() + '\n'); });
}

} // namespace rndr
} // namespace moonray

