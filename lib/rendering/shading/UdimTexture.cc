// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "UdimTexture.h"

#include <moonray/rendering/shading/Shading.h>
#include <moonray/rendering/shading/Texture.h>

#include <moonray/common/file_resource/file_resource.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/MipSelector.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>

#include <OpenImageIO/texture.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>

#include <dirent.h>
#include <unordered_set>

namespace moonray {
namespace shading {

struct TextureOpt;

static const int sMaxUdim = 10;
static const int sUdimStart = 1001;

namespace {
// accumulators are not run during displacement/render_prep
// but neither is ispc, so the displacement argument is ignored
intptr_t
CPP_startOIIOAccumulator(shading::TLState *tls, uint32_t displacement)
{
    auto *exclAcc = getExclusiveAccumulators(tls);
    if (exclAcc && exclAcc->push(EXCL_ACCUM_OIIO)) {
        return (intptr_t)exclAcc;
    }
    return 0;
}

void
CPP_stopOIIOAccumulator(intptr_t accumulator)
{
    if (accumulator) {
        auto exclAcc = (mcrt_common::ExclusiveAccumulators *)accumulator;
        MNRY_ASSERT(exclAcc->isRunning(EXCL_ACCUM_OIIO));
        exclAcc->pop();
    }
}

template<typename STRTYPE>
inline void udimToStr(int udim, STRTYPE &result, size_t pos)
{
    const char digits[10] = { '0', '1', '2', '3', '4', '5', '6', '7',
                              '8', '9' };
    result[pos]           = digits[(udim / 1000) % 10];
    result[pos + 1]       = digits[(udim / 100)  % 10];
    result[pos + 2]       = digits[(udim / 10)   % 10];
    result[pos + 3]       = digits[ udim         % 10];
}

OIIO::TextureOpt::Wrap
getOIIOWrap(WrapType wrapType) {
    switch (wrapType) {
    case WrapType::Black:
        return OIIO::TextureOpt::WrapBlack;
    case WrapType::Clamp:
        return OIIO::TextureOpt::WrapClamp;
    case WrapType::Periodic:
        return OIIO::TextureOpt::WrapPeriodic;
    case WrapType::Mirror:
        return OIIO::TextureOpt::WrapMirror;
    default:
        return OIIO::TextureOpt::WrapDefault;
    }
}

} // namespace

static ispc::UDIM_TEXTURE_StaticData sUdimTextureStaticData;

class UdimTexture::Impl
{
public:
    Impl(scene_rdl2::rdl2::Shader *shader) :
        mShader(shader),
        mErrorUdimOutOfRangeU(0),
        mErrorUdimOutOfRangeV(0),
        mErrorSampleFail(0),
        mNumTextures(0),
        mIs8bit(false)
    {
        mIspc.mShader = (intptr_t) shader;
        mIspc.mTextureHandles = nullptr;
        mIspc.mIsValid = false;

        // Setup various texture quality options
        mTextureOpt[TrilinearAnisotropic].anisotropic = 4;
        mTextureOpt[TrilinearAnisotropic].mipmode = OIIO::TextureOpt::MipModeAniso;
        mTextureOpt[TrilinearAnisotropic].interpmode = OIIO::TextureOpt::InterpBilinear;

        mTextureOpt[TrilinearIsotropic].anisotropic = 1;
        mTextureOpt[TrilinearIsotropic].mipmode = OIIO::TextureOpt::MipModeTrilinear;
        mTextureOpt[TrilinearIsotropic].interpmode = OIIO::TextureOpt::InterpBilinear;

        mTextureOpt[LinearMipClosestTexel].anisotropic = 1;
        mTextureOpt[LinearMipClosestTexel].mipmode = OIIO::TextureOpt::MipModeTrilinear;
        mTextureOpt[LinearMipClosestTexel].interpmode = OIIO::TextureOpt::InterpClosest;

        mTextureOpt[ClosestMipClosestTexel].anisotropic = 1;
        mTextureOpt[ClosestMipClosestTexel].mipmode = OIIO::TextureOpt::MipModeOneLevel;
        mTextureOpt[ClosestMipClosestTexel].interpmode = OIIO::TextureOpt::InterpClosest;

        // to allow for the possibility that we may someday create image maps
        // on multiple threads, we'll protect the writes of the class statics
        // with a mutex.
        static tbb::mutex errorMutex;
        tbb::mutex::scoped_lock lock(errorMutex);
        MOONRAY_START_THREADSAFE_STATIC_WRITE

        mIspc.mUdimTextureStaticDataPtr = &sUdimTextureStaticData;
        sUdimTextureStaticData.mStartAccumulator = (intptr_t)CPP_startOIIOAccumulator;
        sUdimTextureStaticData.mStopAccumulator = (intptr_t)CPP_stopOIIOAccumulator;

        MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
    }

    Impl(const Impl& other) =delete;
    Impl& operator= (const Impl& other) =delete;

    ~Impl() {}

    bool
    update(scene_rdl2::rdl2::Shader *shader,
           scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry,
           const std::string &filename,
           ispc::TEXTURE_GammaMode gammaMode,
           WrapType wrapS,
           WrapType wrapT,
           bool useDefaultColor,
           const scene_rdl2::math::Color& defaultColor,
           const scene_rdl2::math::Color& fatalColor,
           int maxVdim,
           const std::vector<int>& udimValues,
           const std::vector<std::string>& udimFiles,
           std::string &errorMsg)
    {
        init();

        mIspc.mUseDefaultColor = useDefaultColor;
        mIspc.mDefaultColor.r = defaultColor.r;
        mIspc.mDefaultColor.g = defaultColor.g;
        mIspc.mDefaultColor.b = defaultColor.b;
        mIspc.mFatalColor.r = fatalColor.r;
        mIspc.mFatalColor.g = fatalColor.g;
        mIspc.mFatalColor.b = fatalColor.b;
        mIspc.mIsValid = false;

        mNumTextures = sMaxUdim * maxVdim;
        mIspc.mNumTextures = mNumTextures;
        mWidths.assign(mNumTextures, 0);
        mHeights.assign(mNumTextures, 0);
        mPixelAspectRatios.assign(mNumTextures, 0.0f);
        mTextureHandles.assign(mNumTextures, nullptr);
        mIspc.mTextureHandles = reinterpret_cast<intptr_t *>(&mTextureHandles[0]);
        mTextureOptions.resize(mNumTextures * QualityCount);

        std::size_t udimPos = filename.find("<UDIM>");

        mErrorUdimOutOfRangeU =
            logEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL,
                                         "u for udim must be >= 0 and <= 10");
        mIspc.mErrorUdimOutOfRangeU = mErrorUdimOutOfRangeU;

        mErrorUdimOutOfRangeV =
            logEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL,
                                         "v for udim must be >= 0, and within max V if you don't use default color");
        mIspc.mErrorUdimOutOfRangeV = mErrorUdimOutOfRangeV;

        mErrorSampleFail =
            logEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL,
                                         "unknown oiio sample failure");
        mIspc.mErrorSampleFail = mErrorSampleFail;

        mErrorUdimMissingTexture.clear();
        mErrorUdimMissingTexture.resize(mNumTextures);

        bool applyGamma = true;

        // Populate mTextureHandles with all Udim Multi Files TextureHandle pointers
        if ( udimFiles.size() > 0 ) {
            // Load the *explicit* list of UDim File Names
            if (udimValues.size() != udimFiles.size()) {
                mIspc.mIsValid = false;
                if (mIspc.mUseDefaultColor) {
                    return true;
                } else {
                    errorMsg = "FATAL: Invalid UDim list provided!  Unequal number of UDim Values & UDim File names...";
                    return false;
                }
            }
            if (!prepareUdimTextureHandles(udimFiles,
                                           udimValues,
                                           logEventRegistry,
                                           errorMsg,
                                           gammaMode,
                                           applyGamma)) {
                mIspc.mIsValid = false;
                if (mIspc.mUseDefaultColor) {
                    return true;
                } else {
                    return false;
                }
            }
        } else if (udimPos != std::string::npos) {
            // multi-file token substitution udim case
            if (!prepareUdimTextureHandles(filename,
                                           udimPos,
                                           maxVdim,
                                           logEventRegistry,
                                           errorMsg,
                                           gammaMode,
                                           applyGamma)) {
                mIspc.mIsValid = false;
                if (mIspc.mUseDefaultColor) {
                    return true;
                } else {
                    return false;
                }
            }
        }

        mIspc.mErrorUdimMissingTexture = mErrorUdimMissingTexture.data();
        mIspc.mWidths = mWidths.data();
        mIspc.mHeights = mHeights.data();
        mIspc.mPixelAspectRatios = mPixelAspectRatios.data();

        for (int i=0; i < QualityCount; ++i) {
            mTextureOpt[i].swrap = getOIIOWrap(wrapS);
            mTextureOpt[i].twrap = getOIIOWrap(wrapT);
            mTextureOpt[i].subimagename.clear();
        }
        mIspc.mApplyGamma = applyGamma;
        mIspc.mIs8bit = mIs8bit;

        mIspc.mIsValid = true;
        tbb::mutex errorMutex;

        tbb::blocked_range<int> range(0, mTextureHandleIndices.size());
        tbb::parallel_for(range, [&] (const tbb::blocked_range<int> &r) {
            for (int idx = r.begin(); idx < r.end(); ++idx) {
                int i = mTextureHandleIndices[idx];

                for (int j = 0; j < QualityCount; ++j) {
                    // copy a certain quality setting to ispc structure
                    texture::TextureOptions* textureOption =
                            new texture::TextureOptions(mTextureOpt[j]);
                    mTextureOptions[i * QualityCount + j].reset(textureOption);
                }
            }
        });

        mIspc.mTextureOptions = (intptr_t) (&mTextureOptions);

        if (mIspc.mUseDefaultColor) {
            return true;
        } else {
            return mIspc.mIsValid;
        }
    }

    scene_rdl2::math::Color4
    sample(shading::TLState *tls,
           const shading::State& state,
           int udim,
           const scene_rdl2::math::Vec2f& st,
           float *derivatives) const
    {
        if (!mIspc.mIsValid && mIspc.mUseDefaultColor) {
            return scene_rdl2::math::Color4(scene_rdl2::math::asCpp(mIspc.mDefaultColor).r,
                                scene_rdl2::math::asCpp(mIspc.mDefaultColor).g,
                                scene_rdl2::math::asCpp(mIspc.mDefaultColor).b,
                                1.0f);
        }

        const int index = getTextureOptionIndex(state.isDisplacement(), state);

        OIIO::TextureSystem *texSys = MNRY_VERIFY(tls->mTextureSystem);

        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_OIIO);

        // dwa_texture *must* be given 4 floats for the result.
        ALIGN(16) float tmp[4];

        const texture::TextureHandle *texHandle = getTextureHandle(udim);
        if (texHandle == nullptr) {
            // Can happen if the UDIM file couldn't be loaded for whatever reason,
            // eg. missing UDIM tex
            if (mIspc.mUseDefaultColor) {
                return scene_rdl2::math::Color4(mIspc.mDefaultColor.r,
                                    mIspc.mDefaultColor.g,
                                    mIspc.mDefaultColor.b,
                                    1.f);
            } else {
                if (udim >= mNumTextures) {
                    logEvent(mErrorUdimOutOfRangeV);
                } else {
                    logEvent(mErrorUdimMissingTexture[udim]);
                }

                return scene_rdl2::math::Color4(mIspc.mFatalColor.r,
                                    mIspc.mFatalColor.g,
                                    mIspc.mFatalColor.b,
                                    1.f);
            }
        }

        bool res = texSys->texture(
            const_cast<texture::TextureHandle *>(texHandle),
            tls->mOIIOThreadData,
            const_cast<OIIO::TextureOpt&>(*mTextureOptions[udim * QualityCount + index]),
            st[0], st[1],
            derivatives[0], derivatives[1],
            derivatives[2], derivatives[3],
            4, tmp
        );

        scene_rdl2::math::Color4 result;
        if (res) {
            if (mIspc.mApplyGamma && mIspc.mIs8bit) {
                tmp[0] = tmp[0] > 0.0f ? powf(tmp[0], 2.2f) : 0.0f;
                tmp[1] = tmp[1] > 0.0f ? powf(tmp[1], 2.2f) : 0.0f;
                tmp[2] = tmp[2] > 0.0f ? powf(tmp[2], 2.2f) : 0.0f;
                // don't gamma the alpha channel
            }
            result[0] = tmp[0];
            result[1] = tmp[1];
            result[2] = tmp[2];
            result[3] = tmp[3];
        } else {
            logEvent(mErrorSampleFail);
            result[0] = result[1] = result[2] = result[3] = 0.f;
        }

        return result;
    }

    void
    init()
    {
        mTextureOptions.clear();
        mTextureHandles.clear();
        mTextureHandleIndices.clear();
        
        mWidths.clear();
        mHeights.clear();
        mPixelAspectRatios.clear();

        texture::TextureSampler *textureSampler = texture::getTextureSampler();
        if (textureSampler) {
            textureSampler->unregisterMapForInvalidation(mShader);
        }

        mIspc.mApplyGamma = false;
        mIspc.mIs8bit = false;
        mIspc.mIsValid = false;
    }

    bool
    isValid() const
    {
        return mIspc.mIsValid;
    }

    int
    computeUdim(const shading::TLState *tls, const float u, const float v) const
    {
        if (u < 0.f || u > sMaxUdim) {
            logEvent(mErrorUdimOutOfRangeU);
            return -1;
        }
        if (v < 0.f) {
            logEvent(mErrorUdimOutOfRangeV);
            return -1;
        }
        return int(u) + int(v) * sMaxUdim;
    }

    const ispc::UDIM_TEXTURE_Data& getUdimTextureData() const {
        return mIspc;
    }

    void
    getDimensions(int udim, int &x, int& y) const
    {
        x = mWidths[udim];
        y = mHeights[udim];
    }

    float
    getPixelAspectRatio(int udim) const
    {
        return mPixelAspectRatios[udim];
    }

    static void setUdimMissingTextureWarningSwitch(bool flag) { mUdimMissingTextureWarningSwitch = flag; }
    static bool getUdimMissingTextureWarningSwitch() { return mUdimMissingTextureWarningSwitch; }

private:
    void
        logEvent(int error) const
        {
            scene_rdl2::rdl2::Shader::getLogEventRegistry().log(mShader, error);
        }

    bool
    checkUdimConsistencies(texture::TextureSampler* textureSampler,
                           texture::TextureHandle* handle,
                           int firstUdimChannelCount,
                           int firstUdimFileFormat,
                           const std::string& udimFileName,
                           std::string &errorMsg)
    {
        bool consistent = true;
        int currentUdimFileFormat = 0;
        textureSampler->getTextureInfo(handle, "format", &currentUdimFileFormat);
        if (currentUdimFileFormat != firstUdimFileFormat) {
            consistent = false;
            errorMsg = "FATAL: Inconsistent file format in UDIM files: \"" +
                udimFileName + "\". Has: " + std::to_string(currentUdimFileFormat) +
                " Expecting: " + std::to_string(firstUdimFileFormat);
        }

        return consistent;
    }

    bool
    prepareUdimTextureHandle(const std::string &udimFileName,
                             int idx,
                             int &firstUdimChannelCount,
                             int &firstUdimFileFormat,
                             std::string &errorMsg,
                             ispc::TEXTURE_GammaMode gammaMode,
                             bool& applyGamma)
    {
        std::string errorString;
        texture::TextureSampler *textureSampler = texture::getTextureSampler();

        texture::TextureHandle* handle =
                textureSampler->getHandle(udimFileName, errorString);
        if (!handle) {
            errorMsg = "FATAL: failed to open texture file \"" +
                udimFileName +  "\" (" + errorString + ")";
            return false;
        }

        // verify that channel count and format is consistent across udim textures...
        if (firstUdimChannelCount == 0) { // first time through, initialize...
            textureSampler->getTextureInfo(handle, "channels", &firstUdimChannelCount);
            textureSampler->getTextureInfo(handle, "format", &firstUdimFileFormat);
        } else if (!checkUdimConsistencies(textureSampler, handle, firstUdimChannelCount,
                firstUdimFileFormat, udimFileName, errorMsg)) {
            return false;
        }
            
        if (!checkTextureWindow(textureSampler, handle, udimFileName, errorMsg)) {
            return false;
        }

        if (idx >= static_cast<int>(mTextureHandles.size())) {
            errorMsg = "FATAL: udim index exceed max supported udim, please adjust maxVdim accordingly.";
            return false;
        }

        mTextureHandles[idx] = handle;
        mTextureHandleIndices.push_back(idx);
        textureSampler->
                registerMapForInvalidation(udimFileName, mShader, true /* multifile */);

        // retrieve resolution and pixel aspect ratio information. This
        // info may be needed in certain contexts, see ProjectCameraMap_v2 shader
        OIIO::ImageSpec spec;
        OIIO::ustring ufilename(udimFileName.c_str());
        OIIO::TextureSystem *textureSystem = textureSampler->getTextureSystem();
        textureSystem->get_imagespec(ufilename, 0, spec);
        mIs8bit = (spec.format == OIIO::TypeDesc::UINT8);
        mWidths[idx] = spec.width;
        mHeights[idx] = spec.height;

        // Don't apply gamma if any of the images are single channel
        applyGamma = applyGamma ? getApplyGamma(gammaMode, spec.nchannels) : false;

        mPixelAspectRatios[idx] = spec.get_float_attribute("PixelAspectRatio", 1.0f);

        return true;
    }

    bool
    prepareUdimTextureHandles(const std::string &filename,
                              const std::size_t uDimPos,
                              const int maxVdim,
                              scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry,
                              std::string &errorMsg,
                              ispc::TEXTURE_GammaMode gammaMode,
                              bool& applyGamma)
    {
        std::string udimFileName = filename;
        udimFileName.replace(uDimPos, 6, "UDIM");

        // to eliminate negative filesystem lookups, we check the strings in the UDIM directory
        // the directory will be processed once into an unordered set for O(1) lookup, then closed
        const size_t directoryEnd = udimFileName.find_last_of("/\\");
        const std::string udimDirectoryName = udimFileName.substr(0, directoryEnd);
        DIR *directory = opendir(udimDirectoryName.c_str());

        struct dirent *dirEntry;
        std::unordered_set<std::string> allEntries;
        if (directory != NULL) {
            while ((dirEntry = readdir(directory)) != NULL) {
                allEntries.insert(std::string(dirEntry->d_name));
            }
            closedir(directory);
        }

        int firstUdimChannelCount = 0;
        int firstUdimFileFormat = 0;
        for (int idx = 0; idx < mNumTextures; ++idx) {
            udimToStr(idx + sUdimStart, udimFileName, uDimPos);
            const std::string udimFileEntry = udimFileName.substr(directoryEnd + 1);

            const bool fileExists = allEntries.find(udimFileEntry) != allEntries.end();

            if (fileExists) {
                if (!prepareUdimTextureHandle(udimFileName,
                                              idx,
                                              firstUdimChannelCount,
                                              firstUdimFileFormat,
                                              errorMsg,
                                              gammaMode,
                                              applyGamma)) {
                    return false;
                }
            } else {
                // The UDIM file does not exist and we should use the error color when sampling
                mTextureHandles[idx] = nullptr;

                if (mUdimMissingTextureWarningSwitch) {
                    // Create a matching log event for this missing file
                    std::stringstream ss;
                    ss << "computed udim " << (1001 + idx) << " does not have corresponding texture: " << udimFileName;
                    mErrorUdimMissingTexture[idx] =
                        logEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL, ss.str());
                }
            }
        }

        return true;
    }

    bool
    prepareUdimTextureHandles(const std::vector<std::string> &filenameList,
                              const std::vector<int> &udimList,
                              scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry,
                              std::string &errorMsg,
                              ispc::TEXTURE_GammaMode gammaMode,
                              bool& applyGamma)
    {
        int firstUdimChannelCount = 0;
        int firstUdimFileFormat = 0;

        for (size_t i = 0; i < udimList.size(); i++) {
            unsigned int arrayIndex = udimList[i] - sUdimStart;
            std::string udimFileName = filenameList[i];
            if (file_resource::fileExists( udimFileName )) {
                if (!prepareUdimTextureHandle(udimFileName,
                                              arrayIndex,
                                              firstUdimChannelCount,
                                              firstUdimFileFormat,
                                              errorMsg,
                                              gammaMode,
                                              applyGamma)) {
                    return false;
                }
            } else {
                // The UDIM file does not exist and we should use the error color when sampling
                mTextureHandles[arrayIndex] = nullptr;

                if (mUdimMissingTextureWarningSwitch) {
                    // Create a matching log event for this missing file
                    std::stringstream ss;
                    ss << "computed udim " << (1001 + arrayIndex) << " does not have corresponding texture: " << udimFileName;
                    mErrorUdimMissingTexture[arrayIndex] =
                        logEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL, ss.str());
                }
            }
        }
        return true;
    }

    const texture::TextureHandle *
    getTextureHandle(int udim) const
    {
        if (udim >= mNumTextures) return nullptr;
        return static_cast<const texture::TextureHandle *>(mTextureHandles[udim]);
    }

    ispc::UDIM_TEXTURE_Data mIspc;

    scene_rdl2::rdl2::Shader *mShader;
    scene_rdl2::logging::LogEvent mErrorUdimOutOfRangeU;
    scene_rdl2::logging::LogEvent mErrorUdimOutOfRangeV;
    std::vector<scene_rdl2::logging::LogEvent> mErrorUdimMissingTexture; // log event per udim tile
    scene_rdl2::logging::LogEvent mErrorSampleFail;
    std::vector<texture::TextureHandle*> mTextureHandles;
    texture::TextureOptions mTextureOpt[QualityCount];
    std::vector<std::unique_ptr<texture::TextureOptions>> mTextureOptions;
    std::vector<int> mTextureHandleIndices;
    int mNumTextures;
    std::vector<int> mWidths;
    std::vector<int> mHeights;
    std::vector<float> mPixelAspectRatios;
    bool mIs8bit;

    static std::atomic<bool> mUdimMissingTextureWarningSwitch;
};

std::atomic<bool> UdimTexture::Impl::mUdimMissingTextureWarningSwitch { true };

void
UdimTexture::getDimensions(int udim, int &x, int& y) const
{
    mImpl->getDimensions(udim, x, y);
}

float UdimTexture::getPixelAspectRatio(int udim) const
{
    return mImpl->getPixelAspectRatio(udim);
}

UdimTexture::UdimTexture(
    scene_rdl2::rdl2::Shader *shader) :
    mImpl(fauxstd::make_unique<Impl>(shader))
{
}

UdimTexture::~UdimTexture()
{
    mImpl->init();
}

bool
UdimTexture::update(scene_rdl2::rdl2::Shader *shader,
                    scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry,
                    const std::string &filename,
                    ispc::TEXTURE_GammaMode gammaMode,
                    WrapType wrapS,
                    WrapType wrapT,
                    bool useDefaultColor,
                    const scene_rdl2::math::Color& defaultColor,
                    const scene_rdl2::math::Color& fatalColor,
                    int maxVdim,
                    const std::vector<int>& udimValues,
                    const std::vector<std::string>& udimFiles,
                    std::string &errorMsg)
{
    return mImpl->update(shader,
                         logEventRegistry,
                         filename,
                         gammaMode,
                         wrapS,
                         wrapT,
                         useDefaultColor,
                         defaultColor,
                         fatalColor,
                         maxVdim,
                         udimValues,
                         udimFiles,
                         errorMsg);
}

bool
UdimTexture::isValid() const
{
    return mImpl->isValid();
}

scene_rdl2::math::Color4
UdimTexture::sample(shading::TLState *tls,
                    const shading::State& state,
                    int udim,
                    const scene_rdl2::math::Vec2f& st,
                    float *derivatives) const
{
    return mImpl->sample(tls, state, udim, st, derivatives);
}

int
UdimTexture::computeUdim(const shading::TLState *tls,
                         const float u, const float v) const
{
    return mImpl->computeUdim(tls, u, v);
}

const ispc::UDIM_TEXTURE_Data&
UdimTexture::getUdimTextureData() const
{
    return mImpl->getUdimTextureData();
}

// static function
void
UdimTexture::setUdimMissingTextureWarningSwitch(bool flag)
{
    UdimTexture::Impl::setUdimMissingTextureWarningSwitch(flag);
}

// static function    
bool
UdimTexture::getUdimMissingTextureWarningSwitch()
{
    return UdimTexture::Impl::getUdimMissingTextureWarningSwitch();
}

void CPP_oiioUdimTexture(const ispc::UDIM_TEXTURE_Data *tx,
                         shading::TLState *tls,
                         const uint32_t displacement,
                         const int pathType,
                         const float *derivatives,
                         const int udim,
                         const float *st,
                         float *result)
{
    if (!tx->mIsValid && tx->mUseDefaultColor) {
        result[0] = tx->mDefaultColor.r;
        result[1] = tx->mDefaultColor.g;
        result[2] = tx->mDefaultColor.b;
        result[3] = 1.0f;
        return;
    }

    scene_rdl2::rdl2::Shader *shader = reinterpret_cast<scene_rdl2::rdl2::Shader *>(tx->mShader);

    const texture::TextureHandle *textureHandle = (udim >= tx->mNumTextures) ? 
        nullptr :
        (reinterpret_cast<const texture::TextureHandle **>(tx->mTextureHandles))[udim];

    if (textureHandle == nullptr) {
        // Can happen if the UDIM file couldn't be loaded for whatever reason,
        // eg. missing UDIM tex
        if (tx->mUseDefaultColor) {
            result[0] = tx->mDefaultColor.r;
            result[1] = tx->mDefaultColor.g;
            result[2] = tx->mDefaultColor.b;
            result[3] = 1.0f;
            return;
        } else {
            if (udim >= tx->mNumTextures) {
                scene_rdl2::rdl2::Shader::getLogEventRegistry().log(shader, tx->mErrorUdimOutOfRangeV);
            } else {
                scene_rdl2::rdl2::Shader::getLogEventRegistry().log(shader, tx->mErrorUdimMissingTexture[udim]);
            }
            result[0] = tx->mFatalColor.r;
            result[1] = tx->mFatalColor.g;
            result[2] = tx->mFatalColor.b;
            result[3] = 1.0f;
            return;
        }
    }

    texture::TLState::Perthread *threadInfo = tls->mOIIOThreadData;
    OIIO::TextureSystem *texSys = MNRY_VERIFY(tls->mTextureSystem);

    const int index = getTextureOptionIndex(displacement != 0,
        static_cast<shading::Intersection::PathType>(pathType));
    std::vector<std::unique_ptr<texture::TextureOptions>>& options = 
        *(reinterpret_cast<std::vector<std::unique_ptr<texture::TextureOptions>>*>(tx->mTextureOptions));

    float s = st[0];
    float t = st[1];
    float dsdx = derivatives[0];
    float dtdx = derivatives[1];
    float dsdy = derivatives[2];
    float dtdy = derivatives[3];
    const int nChannels = 4;

    bool res = texSys->texture(const_cast<texture::TextureHandle *>(textureHandle),
                               threadInfo,
                               *options[udim * QualityCount + index],
                               s, t,
                               dsdx, dsdy, dtdx, dtdy,
                               nChannels,
                               result);

    if (res) {
        if (tx->mApplyGamma && tx->mIs8bit) {
            result[0] = result[0] > 0.0f ? powf(result[0], 2.2f) : 0.0f;
            result[1] = result[1] > 0.0f ? powf(result[1], 2.2f) : 0.0f;
            result[2] = result[2] > 0.0f ? powf(result[2], 2.2f) : 0.0f;
            // don't gamma the alpha channel
        }
    } else {
        scene_rdl2::rdl2::Shader::getLogEventRegistry().log(shader, tx->mErrorSampleFail);
        result[0] = result[1] = result[2] = result[3] = 0.f;
    }
}

} // namespace shading
} // namespace moonray

//
