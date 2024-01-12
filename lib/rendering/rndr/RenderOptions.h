// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#ifndef RENDEROPTIONS_H
#define RENDEROPTIONS_H

#include "Types.h"
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/common/math/Viewport.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/render/util/GUID.h>
#include <moonray/rendering/shading/Shading.h>
#include <moonray/rendering/mcrt_common/ExecutionMode.h>
// #include <moonray/rendering/mcrt_common/TextureSystem.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scene_rdl2 {
// Forward declarations.
namespace rdl2 { class SceneContext; }
}

namespace moonray {

// Forward declarations.
namespace mcrt_common { struct TLSInitParams; }

namespace rndr {

/**
 * The RenderOptions contain common settings of interest to many parts of the
 * renderer, such as the frame dimensions and output filename.
 *
 * Values are drawn from a variety of sources, including command line arguments,
 * SceneVariables, and direct setting by renderer code.
 *
 * At the moment, options set via parsing the command line or setting them
 * directly will override any equivalent settings in the RDL SceneVariables.
 * In addition, there are last-ditch defaults for each option in case we have
 * no suitable source to set it from.
 *
 * To add a new RenderOption, follow these steps:
 *  1) Add a private member variable to this class, and initialize it to an
 *     "unset" (not default!) value in the constructor.
 *  2) Add a getter and setter for the new variable. The getter should first
 *     see if the value is set and return it, otherwise look it up in the
 *     SceneVariables (if applicable). Finally, it should return a last-ditch
 *     default value. The setter should just be a trivial setter.
 *  3) If the option can be overridden with a command line option, make
 *     sure to add some code for it in parseFromCommandLine(). Don't forget
 *     to add it to the usage message as well in getUsageMessage().
 *  4) If the option must be settable by the MOONRAY infrastructure
 *     ConfigDictionary, you'll need to add the appropriate setter call in
 *     dso/computation/mcrt/McrtComputation.cc. The reason we don't handle that
 *     here is to avoid an infrastructure library dependency.
 *
 * That's it! You can now access the value of the option using the getter
 * (always use the getter!) in any place we use the RenderOptions (right now
 * the RenderContext, RenderDriver, moonray, and MCRT computation).
 */
class RenderOptions
{
public:
    using Arg = scene_rdl2::grid_util::Arg;
    using Parser = scene_rdl2::grid_util::Parser; 

    class AttributeOverride
    {
    public:
        AttributeOverride(const std::string& object, const std::string& attribute, bool isSceneVar = false) :
            mObject(object),
            mAttribute(attribute),
            mIsSceneVar(isSceneVar)
        {
        }

        std::string mObject;
        std::string mAttribute;
        std::string mValue;
        std::string mBinding;
        bool mIsSceneVar; //flags whether this overrides a SceneVariable
    };

    struct RdlaGlobal
    {
        RdlaGlobal(std::string var, std::string expression) :
            mVar(std::move(var)),
            mExpression(std::move(expression))
        {
        }

        std::string mVar;
        std::string mExpression;
    };

    RenderOptions();
    ~RenderOptions();

    /**
     * Parse RenderOptions out of the command line string. Once this call is
     * complete, any options found have been set and their values can be
     * retrieved with the appropriate getter.
     *
     * This does not include fallback values from the SceneVariables for unset
     * options. The RenderContext must be constructed first for the
     * SceneVariable fallbacks to be there.
     *
     * @param   argc    The number of arguments passed to main().
     * @param   argv    The vector of strings passed to main().
     */
    void parseFromCommandLine(int argc, char* argv[]);

    /**
     * Sets the SceneContext from which the SceneVariables will be used as a
     * fallback for unset options. You probably don't need to set this manually.
     * Upon constructing a RenderContext, the context will set this for you
     * to its internal RDL SceneContext.
     *
     * @param   sceneContext    The SceneContext to use for fallback values.
     */
    void setSceneContext(const scene_rdl2::rdl2::SceneContext* sceneContext);

    /// Set number of threads to use
    void setThreads(uint32_t threads);
    uint32_t getThreads() const;

    // Set/get the render mode.
    void setRenderMode(RenderMode mode) { mRenderMode = mode; }
    RenderMode getRenderMode() const    { return mRenderMode; }

    // Set/get the fast render mode.
    void setFastRenderMode(FastRenderMode mode) noexcept { mFastMode = mode; }
    FastRenderMode getFastRenderMode() const noexcept { return mFastMode; }

    // Set/get the application mode.
    void setApplicationMode(ApplicationMode mode) { mApplicationMode = mode; }
    ApplicationMode getApplicationMode() const { return mApplicationMode; }

    // Set/get fps value for realtime renderMode
    void setFps(float fps) { mFps = fps; }
    float getFps() const { return mFps; }

    // Set/get whether to generate pixel info.
    void setGeneratePixelInfo(bool generatePixelInfo) { mGeneratePixelInfo = generatePixelInfo; }
    bool getGeneratePixelInfo() const    { return mGeneratePixelInfo; }

    /// Retrieves the input RDL scene file path.
    const std::vector<std::string>& getSceneFiles() const;

    /// Sets the input RDL scene file path.
    void setSceneFiles(std::vector<std::string> sceneFiles);

    ///  Retrieves the deltas RDL scene file path.
    const std::vector<std::string> & getDeltasFiles() const;

    /// Sets the deltas RDL scene file path.
    void setDeltasFiles(std::vector<std::string> deltasFiles);

    /// Retrieves the search path for RDL DSOs.
    std::string getDsoPath() const;

    /// Sets the search path for RDL DSOs.
    void setDsoPath(const std::string& dsoPath);

    void setTextureCacheSizeMb(int sizeMb) { mTextureCacheSizeMb = sizeMb; }
    int getTextureCacheSizeMb() const { return mTextureCacheSizeMb; }

    /// Retrieves the attribute overrides for SceneObjects in the scene.
    std::vector<AttributeOverride> getAttributeOverrides() const;

    /// Sets the attribute overrides for SceneObjects in the scene. (This
    /// obliterates any existing overrides, so you may want to first get the
    /// overrides, push your changes to the end of the list, then set the new
    /// vector).
    void setAttributeOverrides(const std::vector<AttributeOverride>& overrides);

    /// Get/set Lua global variables that should be set before RDLA is executed.
    const std::vector<RdlaGlobal>& getRdlaGlobals() const;
    void setRdlaGlobals(std::vector<RdlaGlobal> rdlaGlobals);

    // Get/set a flag to determine the *desired* execution mode.
    // The actual render time execution mode is given by FrameState::mVectorized.
    // These may differ since the renderer will revert to scalar execution if
    // volumes are encountered unless the execution mode is VECTORIZED.
    mcrt_common::ExecutionMode getDesiredExecutionMode() const { return mExecutionMode; }
    void setDesiredExecutionMode(mcrt_common::ExecutionMode mode)  { mExecutionMode = mode; }
    void setDesiredExecutionMode(const std::string &execMode);

    // Get/set tile progress.
    void setTileProgress(const bool tileProgress) { mTileProgress = tileProgress; }
    bool getTileProgress() const { return mTileProgress; }

    //
    void setFastGeometry();

    // Get/set apply color render transform
    void setApplyColorRenderTransform(const bool applyCrt) { mApplyColorRenderTransform = applyCrt; }
    bool getApplyColorRenderTransform() const { return mApplyColorRenderTransform; }

    // Get/set the snapshot path
    void setSnapshotPath(const std::string& snapPath) { mSnapshotPath = snapPath; }
    std::string getSnapshotPath() const { return mSnapshotPath; }

    void setColorRenderTransformOverrideLut(const std::string &lut) { mColorRenderTransformOverrideLut = lut; }
    const std::string &getColorRenderTransformOverrideLut() const { return mColorRenderTransformOverrideLut; }

    /// Get / Set for Command Line Athena Log Server specific tag data
    /// Used as keys to track specific User - defined types of Moonray runs (eg. RATS tests)
    const std::string& getAthenaTags() const {
        return mAthenaTagsString;
    }
    void setAthenaTags(const std::string& tagsString) {
        mAthenaTagsString = tagsString;
    }

    /// Get/set for GUID for Athena logging. If not set, Moonray will generate
    /// a uuid4.
    const scene_rdl2::util::GUID& getGUID() const { return mGUID; }
    void setGUID(const scene_rdl2::util::GUID& guid) { mGUID = guid; }

    // Get debugConsole port (port<0 means debug console off, port=0 auto port search by kernel)
    int getDebugConsolePort() const { return mDebugConsolePort; }

    /**
     * Retrieves a usage message for command line programs (named programName)
     * which can be printed for help about which command line options control
     * which render options.
     *
     * @param   programName     The name of the program, used in the usage message.
     * @return  A usage message which can be printed to the console.
     */
    static std::string getUsageMessage(const std::string& programName, bool guiMode);

    std::string getCommandLine()  const { return mCommandLine; }

    void setupTLSInitParams(mcrt_common::TLSInitParams *params, bool realtimeRender) const;
    unsigned getShadingWorkloadChunkSize() const  { return mShadingWorkloadChunkSize; }

    std::string show() const;

    Parser& getParser() { return mParser; }

private:

    void parserConfigure();

    //------------------------------

    // If a SceneContext is get, we'll use its SceneVariables for fallback
    // values when appropriate.
    const scene_rdl2::rdl2::SceneContext* mSceneContext;

    uint32_t mThreads;
    RenderMode mRenderMode;
    FastRenderMode mFastMode;
    bool mGeneratePixelInfo;    // Generally controlled by the application.
    float mRes;
    std::vector<std::string> mSceneFiles;
    std::vector<std::string> mDeltasFiles;
    std::string mDsoPath;
    int mTextureCacheSizeMb;
    std::vector<AttributeOverride> mAttributeOverrides;
    std::vector<RdlaGlobal> mRdlaGlobals;
    std::string mCommandLine;
    mcrt_common::ExecutionMode mExecutionMode;
    bool mTileProgress;        // moonray_gui only
    ApplicationMode mApplicationMode;
    bool mApplyColorRenderTransform; // moonray_gui only
    std::string mSnapshotPath; // moonray_gui only
    std::string mColorRenderTransformOverrideLut;
    std::string mAthenaTagsString;
    scene_rdl2::util::GUID mGUID;
    int mDebugConsolePort; // moonray_gui only

    // Ability to configure various aspects vectorized mode for testing purposes.
    // For developer profiling only, not documented or exposed to user.
    bool mHasVectorizedOverrides;
    unsigned mPerThreadRayStatePoolSize;
    unsigned mRayQueueSize;
    unsigned mOcclusionQueueSize;
    unsigned mPresenceShadowsQueueSize;
    unsigned mShadeQueueSize;
    unsigned mRadianceQueueSize;
    unsigned mShadingWorkloadChunkSize;

    float mFps;                 // only used for realtime rednerMode

    Parser mParser;
};

} // namespace rndr
} // namespace moonray

#endif // RENDEROPTIONS_H
