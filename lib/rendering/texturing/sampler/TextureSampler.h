// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
/// @file TextureSampler.h
///
#pragma once
#include "TextureTLState.h"

#include <scene_rdl2/common/grid_util/Arg.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/common/math/Color.h>

#include <tbb/recursive_mutex.h>

// system
#include <string>
#include <set>

namespace scene_rdl2 {
namespace rdl2 { class Shader; }
}

namespace moonray {

namespace rndr { class RenderContext; }


namespace texture {


class TextureSampler;

// minimize direct dependence on OIIO by using intermediary class type...
typedef OIIO::TextureOpt TextureOptions;
typedef OIIO::ustring OiioUstring;

typedef OIIO::TextureSystem::TextureHandle TextureHandle;

class TextureSampler
{
    using Arg = scene_rdl2::grid_util::Arg;
    using Parser = scene_rdl2::grid_util::Parser;

    friend class TLState;
    friend class rndr::RenderContext;

public:
    TextureSampler();
    ~TextureSampler();

    //------------------------------

    void getStatistics(const std::string& prepend, std::ostream& outs, bool verbose) const;
    void getStatisticsForCsv(std::ostream& outs, bool athenaFormat) const;

    void resetStats() const; // All internal stats are accumulated always if we don't do resetStats().
    float getMainCacheMissFraction() const;
    void getMainCacheInfo(const std::string& prepend, std::ostream& outs) const;
    std::string showStatsFileIOTimeAveragePerThread() const;
    std::string showStats(int level=1, bool icstats=false) const;

    //------------------------------

    // Returns a texture handle which can be subsequently used to sample this
    // image or NULL if we've failed. In the case of failure, the error string
    // contains the reason.
    TextureHandle* getHandle(const std::string& filename,
                             std::string& errorString)
    {
        MNRY_ASSERT(mTextureSystem);
        return getHandle(filename, errorString, mTextureSystem->get_perthread_info());
    }

    // Slightly faster variant of getHandle for cases where we have
    // the OIIO perthread information handy.
    TextureHandle* getHandle(const std::string& filename,
                             std::string& errorString,
                             texture::TLState::Perthread *perThread);

    bool getTextureInfo(TextureHandle* handle,
                        const std::string& data_name,
                        int *data);

    // limits the amount of memory OIIO uses. default is 250MB
    void setMemoryUsage(float megabytes);

    // get current memory usage setting
    float getMemoryUsage() const;

    // limits the number of open files OIIO uses.
    void setOpenFileLimit(int count);

    OIIO::TextureSystem* getTextureSystem() { return mTextureSystem; }

    void registerMapForInvalidation(const std::string &filename,
                                    scene_rdl2::rdl2::Shader *map,
                                    bool multiFileUdimCase);

    void unregisterMapForInvalidation(scene_rdl2::rdl2::Shader *map);

    Parser& getParser() { return mParser; }

protected:
    // Please use moonray::rndr::RenderContext to access these 2 functions...
    void invalidateResources(const std::vector<std::string>& resources) const;
    void invalidateAllResources() const;

    void invalidateResourceInternal(OIIO::ustring file, std::set<scene_rdl2::rdl2::Shader *> *mapsToUpdate) const;

    bool isValid() const;

    bool findShaderAttr(const scene_rdl2::rdl2::Shader *shader,
                        const std::vector<std::string> &attrVec,
                        std::vector<int> &foundId) const;

    std::string showNameToShaderTable() const;
    std::string showShaderToNameTable() const;
    std::string showShaderAttrAll(const std::string &shaderName) const;
    std::string showMaxMemory() const;
    std::string showGetStatistics() const;

    void parserConfigure();

    //------------------------------

    // The oiio system.
    OIIO::TextureSystem*  mTextureSystem;

    //
    // Used to notify ImageMaps of invalidated textures.
    //

    // Different ImageMap objects may reference the same texture file.
    std::multimap<OIIO::ustring, scene_rdl2::rdl2::Shader *> mNameToShader;

    // For the udim case, a single ImageMap may reference multiple texture files.
    std::multimap<scene_rdl2::rdl2::Shader *, OIIO::ustring> mShaderToName;

    tbb::recursive_mutex mMutex;

    Parser mParser;
};

} //  end of texture namespace
} //  end of moonray namespace

