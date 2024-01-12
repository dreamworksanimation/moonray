// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TextureSampler.cpp
///
///

#include "TextureSampler.h"

#include <moonray/statistics/StatsTable.h>
#include <moonray/statistics/StatsTableOutput.h>

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include <algorithm> // min, max
#include <cerrno>
#include <cstring>
#include <sys/stat.h>

namespace {

template <typename F> void
findLine(const std::string& data, const std::string& key, F callBack)
{
    std::stringstream ss(data);
    std::string line;
    while (std::getline(ss, line, '\n')) {
        std::string trimLine = scene_rdl2::str_util::trimBlank(line);
        if (!trimLine.compare(0, key.size(), key)) callBack(trimLine);
    }
}

} // namespace

namespace moonray {
namespace texture {


using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

TextureSampler::TextureSampler()
{
    // Do all image de-coding operations on the render threads, not in an
    // additional OpenExr thread-pool. Search for Imf::setGlobalThreadCount
    // in the OIIO source code to see the impact of this.
    int oiioThreads = 1;
    OIIO::attribute("threads", oiioThreads);
    OIIO::attribute("exr_threads", -1);

    mTextureSystem = OIIO::TextureSystem::create(true);

#if 1
    // Specifically disable auto-mipping and auto-tiling
    mTextureSystem->attribute("automip", int(false));
    mTextureSystem->attribute("autotile", int(0));
    mTextureSystem->attribute("accept_untiled", int(false));
    mTextureSystem->attribute("accept_unmipped", int(false));
#else
    // Specifically enable auto-mipping and auto-tiling
    mTextureSystem->attribute("automip", int(true));
    mTextureSystem->attribute("autotile", int(64));
    mTextureSystem->attribute("accept_untiled", int(true));
    mTextureSystem->attribute("accept_unmipped", int(true));
#endif

    // Convert single channel textures to grayscale (rather than r 0 0)
    mTextureSystem->attribute("gray_to_rgb", 1);

    parserConfigure();
}

TextureSampler::~TextureSampler()
{
    OIIO::TextureSystem::destroy(mTextureSystem, false);
}

TextureHandle*
TextureSampler::getHandle(const std::string& fileName,
                          std::string& errorString,
                          texture::TLState::Perthread *perThread)
{
    MNRY_ASSERT(mTextureSystem);
    MNRY_ASSERT(perThread == mTextureSystem->get_perthread_info());

    OIIO::ustring file = static_cast<OIIO::ustring>(fileName);

    // Note: The OIIO api doesn't seem to provide a way to close a texture
    //       handle. It appears all handles are kept open until the OIIO texture
    //       system is itself destroyed. This may be problematic for interactive
    //       workflows in the future.
    TextureHandle *textureHandle = mTextureSystem->get_texture_handle(file, perThread);

    if (!textureHandle || !mTextureSystem->good(textureHandle)) {
        errorString = mTextureSystem->geterror();
        return nullptr;
    }

    return textureHandle;
}

bool 
TextureSampler::getTextureInfo(TextureHandle* handle,
                               const std::string& data_name,
                               int *data) {
    // Reference of Supported Names.  See OIIO imagecache.cpp get_image_info 
    // for complete list
    // ("exists"), ("broken"), ("channels"), ("subimages"), 
    // ("miplevels"), ("format"), ("cachedformat"), ("cachedpixeltype");

    OIIO::TypeDesc type;
    if (data_name == "channels" || data_name == "format") {
        type = OIIO::TypeDesc::TypeInt;
    } else if (data_name == "datawindow" || data_name == "displaywindow") {
        type = OIIO::TypeDesc(OIIO::TypeDesc::INT, OIIO::TypeDesc::SCALAR, 4);
    } else {
        MNRY_ASSERT(0 && "unknown OIIO::TypeDesc for requested texture info");
    }

    return 
        mTextureSystem->get_texture_info(handle,
                                         mTextureSystem->get_perthread_info(),
                                         0 /* subimage */,
                                         (OIIO::ustring)data_name, 
                                         type,
                                         (void*) data);
}

void
TextureSampler::getStatistics(const std::string& prepend, std::ostream& outs, bool verbose) const
{

    int level = verbose ? 5 : 1;
    std::string stats = mTextureSystem->getstats(level /* logging level 1-5 */,
                                                 true /* output cache stats */);

    // Split apart the stats string into individual lines and prepend each line
    std::string delimiter = "\n";
    size_t pos = 0;
    while ((pos = stats.find(delimiter)) != std::string::npos) {
        std::string statsLine = stats.substr(0, pos);
        outs << prepend << statsLine << std::endl;
        stats.erase(0, pos + delimiter.length());
    }
}

void
TextureSampler::getStatisticsForCsv(std::ostream& outs, bool athenaFormat) const
{
    std::string stats = mTextureSystem->getstats(5 /* logging level 1-5 */, 
                                                 true /* output cache stats */);

    moonray_stats::StatsTable<2> textureStatsTable("OpenImageIO Texture Statistics");
    moonray_stats::StatsTable<2> icStatsTable("OpenImageIO ImageCache Statistics");
    moonray_stats::StatsTable<7> imageFileTable("OpenImageIO Image File Statistics",
                                                "File", "Opens", "Tiles", "MB Read",
                                                "I/O Time (s)", "Res", "Mip Count");
    moonray_stats::StatsTable<2> summaryTable("OpenImageIO Image File Summary");

    // Split apart the stats string into individual lines
    std::string delimiter = "\n";
    size_t pos = 0;
    while ((pos = stats.find(delimiter)) != std::string::npos) {
        std::string statsLine = stats.substr(0, pos);
        stats.erase(0, pos + delimiter.length());

        // Split line into tokens
        std::vector<std::string> tokens;
        std::string tokenDelimiters = " ()";
        size_t tokenPos = 0;
        while ((tokenPos = statsLine.find_first_of(tokenDelimiters)) != std::string::npos) {
            std::string token = statsLine.substr(0, tokenPos);
            if (token.length() > 0) {
                tokens.push_back(statsLine.substr(0, tokenPos));
            }
            statsLine.erase(0, tokenPos + 1);
        }
        if (statsLine.length() > 0) tokens.push_back(statsLine);
        if (tokens.size() == 0) continue;

        // Examine the tokens and populate the stats tables

        // texture : 123 queries in 123 batches
        if (tokens.size() == 7 && tokens[0] == "texture" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Texture queries", tokens[2]);
            textureStatsTable.emplace_back("Texture batches", tokens[5]);
        }
        // texture 3d : 123 queries in 123 batches
        else if (tokens.size() == 8 && tokens[0] == "texture" && tokens[1] == "3d") {
            textureStatsTable.emplace_back("Texture 3d queries", tokens[3]);
            textureStatsTable.emplace_back("Texture 3d batches", tokens[6]);
        }
        // shadow : 123 queries in 123 batches
        else if (tokens.size() == 7 && tokens[0] == "shadow" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Shadow queries", tokens[2]);
            textureStatsTable.emplace_back("Shadow batches", tokens[5]);
        }
        // environment : 123 queries in 123 batches
        else if (tokens.size() == 7 && tokens[0] == "environment" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Environment queries", tokens[2]);
            textureStatsTable.emplace_back("Environment batches", tokens[5]);
        }
        // closest : 123 
        else if (tokens.size() == 3 && tokens[0] == "closest" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Closest interpolations", tokens[2]);
        }
        // bilinear : 123 
        else if (tokens.size() == 3 && tokens[0] == "bilinear" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Bilinear interpolations", tokens[2]);
        }
        // bicubic : 123 
        else if (tokens.size() == 3 && tokens[0] == "bicubic" && tokens[1] == ":") {
            textureStatsTable.emplace_back("Bicubic interpolations", tokens[2]);
        }
        // Average anisotropic probes : 123 
        else if (tokens.size() == 5 && tokens[0] == "Average" && tokens[1] == "anisotropic") {
            textureStatsTable.emplace_back("Average anisotropic probes", tokens[4]);
        }
        // Max anisotropy in the wild : 123 
        else if (tokens.size() == 7 && tokens[0] == "Max" && tokens[1] == "anisotropy") {
            textureStatsTable.emplace_back("Max anisotropy in the wild", tokens[6]);
        }
        // OpenImageIO ImageCache statistics shared ver 2.3.20
        else if (tokens.size() == 6 && tokens[0] == "OpenImageIO" && tokens[1] == "ImageCache") {
            icStatsTable.emplace_back("Image cache type", tokens[3]);
            icStatsTable.emplace_back("OIIO ver", tokens[5]);
        }
        // Images : 5 unique
        else if (tokens.size() == 4 && tokens[0] == "Images" && tokens[1] == ":") {
            icStatsTable.emplace_back("Unique images", tokens[2]);
        }
        // ImageInputs : 5 created, 5 current, 5 peak
        else if (tokens.size() == 8 && tokens[0] == "ImageInputs" && tokens[1] == ":") {
            icStatsTable.emplace_back("ImageInputs created", tokens[2]);
            icStatsTable.emplace_back("ImageInputs current", tokens[4]);
            icStatsTable.emplace_back("ImageInputs peak", tokens[6]);
        }     
        // Total pixel data size of all images referenced : 1.1 GB
        else if (tokens.size() == 11 && tokens[0] == "Total" && tokens[1] == "pixel") {
            icStatsTable.emplace_back("Total size of all images referenced (" + tokens[10] + ")", tokens[9]);
        }
        // Pixel data read : 165.3 MB
        else if (tokens.size() == 6 && tokens[0] == "Pixel" && tokens[1] == "data") {
            icStatsTable.emplace_back("Read from disk (" + tokens[5] + ")", tokens[4]);
        }
        // File I/O time : 5.4s 0.1s average per thread, for 37 threads
        else if (tokens.size() == 12 && tokens[0] == "File" && tokens[1] == "I/O") {
            icStatsTable.emplace_back("File I/O time (s)", tokens[4].substr(0, tokens[4].length() - 1));
            tokens[5].pop_back(); // remove 's'
            icStatsTable.emplace_back("File I/O time per thread average (s)", tokens[5]);
        }       
        // File open time only : 0.0s
        else if (tokens.size() == 6 && tokens[0] == "File" && tokens[1] == "open") {
            tokens[5].pop_back(); // remove 's'
            icStatsTable.emplace_back("File open time only (s)", tokens[5]);
        }
        // Tiles: 14548 created, 12999 current, 12999 peak
        else if (tokens.size() == 7 && tokens[0] == "Tiles:") {
            icStatsTable.emplace_back("Tiles created", tokens[1]);
            icStatsTable.emplace_back("Tiles current", tokens[3]);
            icStatsTable.emplace_back("Tiles peak", tokens[5]);
        }  
        // total tile requests : 73364396 
        else if (tokens.size() == 5 && tokens[0] == "total" && tokens[1] == "tile") {
            icStatsTable.emplace_back("Total tile requests", tokens[4]);
        }   
        // micro-cache misses : 29209283 39.814%    
        else if (tokens.size() == 5 && tokens[0] == "micro-cache" && tokens[1] == "misses") {
            icStatsTable.emplace_back("Micro-cache misses", tokens[3]);
            tokens[4].pop_back(); // remove '%'
            icStatsTable.emplace_back("Micro-cache misses (%)", tokens[4]);
        }   
        // main cache misses : 14655 0.0199756%     
        else if (tokens.size() == 6 && tokens[0] == "main" && tokens[1] == "cache") {
            icStatsTable.emplace_back("Main cache misses", tokens[4]);
            tokens[5].pop_back(); // remove '%'
            icStatsTable.emplace_back("Main cache misses (%)", tokens[5]);
        }   
        // Peak cache memory : 165.5 MB        
        else if (tokens.size() == 6 && tokens[0] == "Peak" && tokens[1] == "cache") {
            icStatsTable.emplace_back("Peak cache memory (" + tokens[5] + ")", tokens[4]);
        }   
        // Tot: 5 12999 165.3 0 0.0 25.9s
        else if (tokens.size() == 7 && tokens[0] == "Tot:") {
            summaryTable.emplace_back("Total opens", tokens[1]);
            summaryTable.emplace_back("Total tiles", tokens[2]);
            summaryTable.emplace_back("Total size read (MB)", tokens[3]);
            tokens[6].pop_back(); // remove 's'
            summaryTable.emplace_back("Total I/O time (s)", tokens[6]);
        }  
        // 1 1 3256 38.2 1.0s 8192x8192x3.u8 /work/rd/raas/maps/toothless/oiio/DIF.tx MIP-COUNT[1871,850,338,125,45,16,4,1,1,1,1,1,1,1]       
        else if (tokens.size() == 8 && std::isdigit(tokens[0][0])) {
            imageFileTable.emplace_back(tokens[6], // file
                                        tokens[1], // opens
                                        tokens[2], // tiles
                                        tokens[3], // MB read
                                        tokens[4], // I/O time
                                        tokens[5], // res
                                        tokens[7]); // mip count
        }
    }

    moonray_stats::writeEqualityCSVTable(outs, textureStatsTable, athenaFormat);
    moonray_stats::writeEqualityCSVTable(outs, icStatsTable, athenaFormat);
    moonray_stats::writeCSVTable(outs, imageFileTable, athenaFormat);
    moonray_stats::writeEqualityCSVTable(outs, summaryTable, athenaFormat);
}

void
TextureSampler::resetStats() const
{
    mTextureSystem->reset_stats();
}

float
TextureSampler::getMainCacheMissFraction() const
//
// Returns main cache misses percentage value as a fraction.
// Returns negative value if error
//
// This stats string parsing is only tested for OIIO version 1.7.7 but there is no runtime version check
// logic. We should check this parsing logic properly working if we use other version of OIIO.
//    
{
    float f = -1.0f;

    std::string statsData = showStats(1, true);

    // The format should be like this.
    // main cache misses : 13805580 (4.33887%)
    unsigned long lu;
    findLine(statsData, "main cache misses", [&](const std::string& line) {
            sscanf(line.c_str(), "main cache misses : %lu (%f%%)", &lu, &f);
        });
    if (f < 0.0f) return f; // return negative value if we can not find field
    return f / 100.0f; // return fraction
}

void
TextureSampler::getMainCacheInfo(const std::string& prepend, std::ostream& outs) const
{
    using scene_rdl2::str_util::byteStr;
    
    float memSizeMB = getMemoryUsage();
    size_t memSizeByte = static_cast<size_t>(memSizeMB * 1024.0f * 1024.0f);
    outs << prepend
         << "texture_cache_size    = " << static_cast<size_t>(memSizeMB)
         << " (" << byteStr(memSizeByte) << ")\n";

    float hitMissFraction = getMainCacheMissFraction();
    if (hitMissFraction < 0.0f) {
        outs << prepend << "main cache miss ratio = ?\n";
    } else {
        outs << prepend << "main cache miss ratio = " << hitMissFraction * 100.0f << "%" << '\n';
    }
}

std::string
TextureSampler::showStatsFileIOTimeAveragePerThread() const
//
// Returns file I/O time (average per thread value) as a string.
// Returns "?" if error.
// 
// This stats string parsing is only tested for OIIO version 1.7.7 but there is no runtime version check
// logic. We should check this parsing logic properly working if we use other version of OIIO.
//
{
    std::string timeStr("?");

    std::string statsData = showStats(1, true);
    
    // The format should be like this
    // File I/O time : 1h 53m 15.2s (3m8.8s average per thread)
    //                               ^^^^^^ return this
    findLine(statsData, "File I/O time", [&](const std::string& line) {
            std::string::size_type start = line.find("(");
            std::string::size_type end = line.find("average");
            if (start != std::string::npos && end != std::string::npos) {
                std::string::size_type subStart = start + 1;
                std::string::size_type subEnd = end - 1;
                if (subStart <= subEnd) {
                    timeStr = line.substr(subStart, subEnd - subStart);
                }
            }
        });
    return timeStr;
}

std::string
TextureSampler::showStats(int level, bool icstats) const
//
// under OIIO 1.7.7, level value range is from 1 to 5.
//
{
    using scene_rdl2::str_util::addIndent;
    using scene_rdl2::str_util::rmLastNL;

    std::ostringstream ostr;
    ostr << "textureSampler stats {\n"
         << addIndent(rmLastNL(mTextureSystem->getstats(std::min(std::max(level, 1), 5), icstats))) << '\n'
         << "}";
    return ostr.str();
}

void
TextureSampler::setMemoryUsage(float megabytes)
{
    mTextureSystem->attribute("max_memory_MB", OIIO::TypeDesc::FLOAT, &megabytes);
}

float
TextureSampler::getMemoryUsage() const
{
    float mb;
    mTextureSystem->getattribute("max_memory_MB", mb);
    return mb;
}
    
void
TextureSampler::setOpenFileLimit(int count)
{
    mTextureSystem->attribute("max_open_files", OIIO::TypeDesc::INT, &count);
}

void
TextureSampler::invalidateResources(const std::vector<std::string>& resources) const
{
    MNRY_ASSERT(isValid());

    std::set<scene_rdl2::rdl2::Shader *> mapsToUpdate;

    // 1st pass, grab list of maps which need updating.
    for (const std::string& resourceName: resources) {
        OIIO::ustring file = static_cast<OIIO::ustring>(resourceName);
        invalidateResourceInternal(file, &mapsToUpdate);
    }

    // 2nd pass, do all the actual updates (could be done in parallel?).
    for (auto it = mapsToUpdate.begin(); it != mapsToUpdate.end(); ++it) {
        (*it)->update();
    }
}

void
TextureSampler::invalidateAllResources() const
{
    MNRY_ASSERT(isValid());
    
    std::set<scene_rdl2::rdl2::Shader *> mapsToUpdate;

    // 1st pass, grab list of maps which need updating.
    for (auto it = mNameToShader.begin(); it != mNameToShader.end(); ++it) {
        invalidateResourceInternal(it->first, &mapsToUpdate);
    }

    // 2nd pass, do all the actual updates (could be done in parallel?).
    for (auto it = mapsToUpdate.begin(); it != mapsToUpdate.end(); ++it) {
        (*it)->update();
    }
}

void
TextureSampler::registerMapForInvalidation(const std::string &fileName, scene_rdl2::rdl2::Shader *map, bool multiFileUdimCase)
{

    // Textures could be loaded in parallel, use mutex to avoid data race.
    tbb::recursive_mutex::scoped_lock lock(mMutex);
    
    OIIO::ustring oiioFileName(fileName);

    auto range = mShaderToName.equal_range(map);
    for (auto nameIt = range.first; nameIt != range.second; ++nameIt) {
        if (fileName == nameIt->second) {
            // This has already been registered, no work to do.
            MNRY_ASSERT(isValid());
            return;
        }
    }

    if (!multiFileUdimCase) {
        // Map is contained but fileName has changed, remove and re-add it.
        unregisterMapForInvalidation(map);
    }

    mNameToShader.insert(std::pair<OIIO::ustring, scene_rdl2::rdl2::Shader *>(oiioFileName, map));
    mShaderToName.insert(std::pair<scene_rdl2::rdl2::Shader *, OIIO::ustring>(map, oiioFileName));
    MNRY_ASSERT(isValid());
}

void
TextureSampler::unregisterMapForInvalidation(scene_rdl2::rdl2::Shader *map)
{
    tbb::recursive_mutex::scoped_lock lock(mMutex);
    MNRY_ASSERT(isValid());

    auto mapRange = mShaderToName.equal_range(map);

    // Do work only if this map was already registered.
    if (mapRange.first != mapRange.second) {

        for (auto mapIt = mapRange.first; mapIt != mapRange.second;) {

            // Delete this key from the mNameToShader container.
            auto name = mapIt->second;
            auto nameRange = mNameToShader.equal_range(name);

            for (auto nameIt = nameRange.first; nameIt != nameRange.second; ++nameIt) {
                if (nameIt->second == map) {
                    mNameToShader.erase(nameIt);
                    break;
                }
            }

            // Now delete it from mShaderToName.
            auto oldIt = mapIt;
            ++mapIt;
            mShaderToName.erase(oldIt);
        }

        MNRY_ASSERT(isValid());
    }
}

void
TextureSampler::invalidateResourceInternal(OIIO::ustring file, std::set<scene_rdl2::rdl2::Shader *> *mapsToUpdate) const
{
    MNRY_ASSERT(mapsToUpdate);

    std::string resourceName = file.c_str();
    mTextureSystem->invalidate(file);

    // Read the texture file again.
    if(!mTextureSystem->imagespec(file)) {
        std::string errorString ("Unable to read texture file " + resourceName);
        Logger::error (errorString);
    }

    // This is a possible attribute name that we might consider to update the texture file.
    // This list needs to update when the shader command changes in the future.
    static const std::vector<std::string> attrVec = {
        "texture",
        "file",
        "positive_x_texture",
        "positive_y_texture",
        "positive_z_texture",
        "negative_x_texture",
        "negative_y_texture",
        "negative_z_texture",
        "fallback glitter texture A",
        "glitter texture A",
        "glitter texture B",
        "image_path",
        "flake_texture_1",
        "flake_texture_2",
        "tangent_space_normal_texture"
    };

    std::string updateFile = std::string(file.c_str());

    auto range = mNameToShader.equal_range(file);
    for (auto itr = range.first; itr != range.second; ++itr) {
        scene_rdl2::rdl2::Shader *map = itr->second;
        std::vector<int> foundId;
        bool updateFlag = false;
        if (findShaderAttr(map, attrVec, foundId)) {
            for (size_t i = 0; i < foundId.size(); ++i) {
                const std::string &attrName = attrVec[foundId[i]];
                if (map->get<std::string>(attrName) == updateFile) {
                    map->beginUpdate();
                    map->set(attrName, std::string());
                    map->set(attrName, updateFile);
                    map->endUpdate();
                    updateFlag = true;
                }
            }
        }

        if (updateFlag) {
            mapsToUpdate->insert(map); // Notify ImageMap objects of texture invalidation.
        }
    }

    /* useful debug code
    {
        std::ostringstream ostr;
        ostr << "mapsToUpdate size:" << mapsToUpdate->size() << " {\n";
        for (auto itr = mapsToUpdate->begin(); itr != mapsToUpdate->end(); ++itr) {
            ostr << "  shaderAddr:0x" << std::hex << reinterpret_cast<uintptr_t>(*itr)
                 << " shaderName:" << (*itr)->getName() << '\n';
        }
        ostr << "}";
        std::cerr << ">> TextureSampler.cc " << ostr.str() << '\n';
    }
    */
}

// Check invariants.
bool
TextureSampler::isValid() const
{
    MNRY_ASSERT(mNameToShader.size() == mShaderToName.size());

    // Check maps are mirrors of each other.
    for (auto it = mNameToShader.begin(); it != mNameToShader.end(); ++it) {
        MNRY_DURING_ASSERTS(auto reverseIt = mShaderToName.find(it->second));
        MNRY_ASSERT(reverseIt != mShaderToName.end());
    }

    for (auto it = mShaderToName.begin(); it != mShaderToName.end(); ++it) {
        MNRY_DURING_ASSERTS(auto reverseIt = mNameToShader.find(it->second));
        MNRY_ASSERT(reverseIt != mNameToShader.end());
    }

    // Check for duplicates.
    MNRY_ASSERT(std::adjacent_find(mNameToShader.begin(), mNameToShader.end()) == mNameToShader.end());
    MNRY_ASSERT(std::adjacent_find(mShaderToName.begin(), mShaderToName.end()) == mShaderToName.end());

    return true;
}

bool
TextureSampler::findShaderAttr(const scene_rdl2::rdl2::Shader *shader,
                               const std::vector<std::string> &attrVec,
                               std::vector<int> &foundId) const
{
    foundId.clear();
    const scene_rdl2::rdl2::SceneClass &sceneClass = shader->getSceneClass();
    for (size_t i = 0; i < attrVec.size(); ++i) {
        for (auto itr = sceneClass.beginAttributes(); itr != sceneClass.endAttributes(); ++itr) {
            if ((*itr)->getFlags() == scene_rdl2::rdl2::FLAGS_FILENAME && (*itr)->getName() == attrVec[i]) {
                foundId.push_back(static_cast<int>(i));
            }
        }
    }
    return (foundId.size() > 0);
}

std::string
TextureSampler::showNameToShaderTable() const
{
    std::ostringstream ostr;

    ostr << "mNameToShader size:" << mNameToShader.size() << " {\n";
    for (auto itr = mNameToShader.begin(); itr != mNameToShader.end(); ++itr) {
        auto count = mNameToShader.count(itr->first);
        auto range = mNameToShader.equal_range(itr->first);

        ostr << "  textureName:" << itr->first << " count:" << count << " {\n";
        for (auto itr2 = range.first; itr2 != range.second; ++itr2) {
            ostr << "    shaderAddr:0x" << std::hex << reinterpret_cast<uintptr_t>(itr2->second)
                 << " shaderName:" << itr2->second->getName() << '\n';
        }
        ostr << "  }\n";
    }
    ostr << "}";

    return ostr.str();
}

std::string
TextureSampler::showShaderToNameTable() const
{
    std::ostringstream ostr;

    ostr << "mShaderToName size:" << mShaderToName.size() << " {\n";
    for (auto itr = mShaderToName.begin(); itr != mShaderToName.end(); ++itr) {
        auto count = mShaderToName.count(itr->first);
        auto range = mShaderToName.equal_range(itr->first);

        ostr << "  shaderAddr:0x" << std::hex << reinterpret_cast<uintptr_t>(itr->first)
             << " shaderName:" << itr->first->getName()
             << " count:" << count << " {\n";
        for (auto itr2 = range.first; itr2 != range.second; ++itr2) {
            ostr << "    name:" << itr2->second << '\n';
        }
        ostr << "  }\n";
    }
    ostr << "}";

    return ostr.str();
}

std::string
TextureSampler::showShaderAttrAll(const std::string &shaderName) const
{
    auto getShader = [&](const std::string &shaderName) -> scene_rdl2::rdl2::Shader * {
        for (auto itr = mShaderToName.begin(); itr != mShaderToName.end(); ++itr) {
            scene_rdl2::rdl2::Shader *shader = itr->first;
            if (shader->getName() == shaderName) return shader;
        }
        return nullptr;
    };
    scene_rdl2::rdl2::Shader *shader = getShader(shaderName);

    std::ostringstream ostr;
    ostr << "shaderName:" << shaderName;
    if (shader) {
        const scene_rdl2::rdl2::SceneClass &sceneClass = shader->getSceneClass();
        ostr << ' ' << sceneClass.showAllAttributes();
    } else {
        ostr << " shader is nullptr";
    }
    return ostr.str();
}

std::string
TextureSampler::showMaxMemory() const
{
    using scene_rdl2::str_util::byteStr;

    std::ostringstream ostr;
    float mb = getMemoryUsage();
    size_t b = static_cast<size_t>(mb * 1024.0f * 1024.f);
    ostr << "getMaxMemory:" << mb << "MB (" << byteStr(b) << ")";
    return ostr.str();
}

std::string
TextureSampler::showGetStatistics() const
{
    std::string prepend = "showGetStatistics : ";

    std::ostringstream ostr;
    getStatistics(prepend, ostr, /* verbose */ false);
    getMainCacheInfo(prepend, ostr);
    return ostr.str();
}

void
TextureSampler::parserConfigure()
{
    mParser.description("textureSampler command");

    mParser.opt("nameToShader", "", "show mNameToShader table. might be pretty long",
                [&](Arg& arg) -> bool { return arg.msg(showNameToShaderTable() + '\n'); });
    mParser.opt("shaderToName", "", "show mShaderToName table",
                [&](Arg& arg) -> bool { return arg.msg(showShaderToNameTable() + '\n'); });
    mParser.opt("shaderAttr", "<shaderName>", "show all shader attribute. might be pretty long",
                [&](Arg& arg) -> bool { return arg.msg(showShaderAttrAll((arg++)()) + '\n'); });
    mParser.opt("getMaxMemory", "", "get texture system cache size",
                [&](Arg& arg) -> bool { return arg.msg(showMaxMemory() + '\n'); });

    mParser.opt("resetStats", "", "reset stats",
                [&](Arg& arg) -> bool { resetStats(); return arg.msg("reset stats\n"); });
    mParser.opt("cacheMiss", "", "get main cache miss fraction",
                [&](Arg& arg) -> bool { return arg.fmtMsg("cacheMiss:%f\n", getMainCacheMissFraction()); });
    mParser.opt("showStats", "<level> <on|off>", "show current stats. level(1~5) with cacheInfo(on|off)",
                [&](Arg& arg) -> bool {
                    std::string msg = showStats(arg.as<int>(0), arg.as<bool>(1));
                    arg += 2;
                    return arg.msg(msg + '\n');
                });
    mParser.opt("showUserFriendlyStats", "", "show user-friendly stats that moonray log shows",
                [&](Arg& arg) -> bool { return arg.msg(showGetStatistics() + '\n'); });
    mParser.opt("showFileIOTimeAveragePerThread", "", "show file I/O time averaged per thread",
                [&](Arg& arg) -> bool {
                    return arg.fmtMsg("fileI/O:%s\n", showStatsFileIOTimeAveragePerThread().c_str());
                });
}

} // namespace texture
} // namespace moonray

