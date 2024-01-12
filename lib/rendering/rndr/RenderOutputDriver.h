// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file RenderOutputDriver.h

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include <memory>
#include <string>
#include <vector>

namespace moonray {

namespace pbr {
class AovSchema;
class CryptomatteBuffer;
class DeepBuffer;
class LightAovs;
class MaterialAovs;
}

namespace rndr {

class Film;
class ImageWriteCache;
class RenderContext;
class VariablePixelBuffer;

/// A class that can interpret the contents of a set of
/// RenderOutput objects (that are possibly inconsistent), resolve
/// them into a consistent set, query information about them such as
/// "needs heat map results" and ultimately output the results
/// using the OIIO API.
class RenderOutputDriver
{
public:
    /// Create a render output driver from an array of render output objects
    /// Activity and consistency is checked, resulting in an ordered list
    /// of active render output objects.
    explicit RenderOutputDriver(const RenderContext *renderContext);
    ~RenderOutputDriver();

    /// Storeet final maxSamplesPerPix count for output file metadata
    void setFinalMaxSamplesPerPix(unsigned v);

    /// How many valid/active render output objects do we have?
    unsigned int getNumberOfRenderOutputs() const;

    /// We have an ordered array of render outputs, get the indx value
    const scene_rdl2::rdl2::RenderOutput *getRenderOutput(unsigned int indx) const;

    /// Given a render output, return its indx
    /// @return -1 if not found
    int getRenderOutputIndx(const scene_rdl2::rdl2::RenderOutput *ro) const;

    /// Number of channels used by the render output
    /// @param indx return of getRenderOutputIndx()
    /// 0 if inactive or deactivated, otherwise 1, 2, or 3
    unsigned int getNumberOfChannels(unsigned int indx) const;

    /// A render output might use one of Film's aov buffers.  This function
    /// returns the indx into the aov buffers.
    /// @param indx return of getRenderOutputIndx()
    /// @return -1 if error or render output does not use an aov buffer
    int getAovBuffer(unsigned int indx) const;

    /// A render output might use one of Film's DisplayFilter buffers.  This function
    /// returns the indx into the DisplayFilter buffers.
    /// @param indx return of getRenderOutputIndx()
    /// @return -1 if error or render output does not use a DisplayFilter buffer
    int getDisplayFilterIndex(unsigned int indx) const;

    /// Given a render output index, return whether it's a visbility aov
    /// @param indx return of getRenderOutputIndx()
    bool isVisibilityAov(unsigned int indx) const;

    /// Get the geom primitive attribute keys that are needed
    /// by the the active render outputs
    const shading::AttributeKeySet &getPrimAttrs() const;

    /// Aov schema - This array contains the ordered list of aov schema Ids
    /// in the Film's aov buffer.  For example, if the schema contains
    /// three entries as:
    ///    schema[0].mId = AOV_SCHEMA_ID_STATE_P
    ///    schema[1].mId = AOV_SCHEMA_ID_STATE_ST
    ///    schema[2].mId = AOV_SCHEMA_ID_STATE_DSDX
    /// Then the aov buffer is expected to have 3 entries laid out
    /// as:
    ///    aovBuffer[0] = Float3Buffer (P.x, P.y, P.z)
    ///    aovBuffer[1] = Float2Buffer (St.x, St.y)
    ///    aovBuffer[2] = FloatBuffer  (dsdx)
    const pbr::AovSchema &getAovSchema() const;

    /// Return default value for Aov buffer based on aov filter definition
    /// @param indx return of getRenderOutputIndx()
    float getAovDefaultValue(const unsigned indx) const;

    /// Return condition of this Aov buffer requires scaled by weight or not
    /// @param indx return of getRenderOutputIndx()
    bool requiresScaledByWeight(const unsigned indx) const;

    /// Does this driver require DeepBuffer results?
    bool requiresDeepBuffer() const;

    /// Does this render output require the DeepBuffer?
    bool requiresDeepBuffer(unsigned int indx) const;

    /// Does this driver require cryptomatte results?
    bool requiresCryptomatteBuffer() const;

    /// Does this render output require cryptomatte?
    bool requiresCryptomatteBuffer(unsigned int indx) const;

    /// Number of cryptomatte layers to output
    int getCryptomatteNumLayers() const;

    /// Does this driver require RenderBuffer results?
    bool requiresRenderBuffer() const;

    /// Does this render output require the RenderBuffer?
    bool requiresRenderBuffer(unsigned int indx) const;

    /// Does this driver require RenderBufferOdd results?
    bool requiresRenderBufferOdd() const;

    /// Does this driver require RenderBuffer Odd results?
    bool requiresRenderBufferOdd(unsigned int indx) const;

    /// Does this driver require heat map results?
    bool requiresHeatMap() const;

    /// Does this render output require the HeatMap buffer
    bool requiresHeatMap(unsigned int indx) const;

    /// Does this driver require weight map results?
    bool requiresWeightBuffer() const;

    /// Does this render output require the WeightBuffer ?
    bool requiresWeightBuffer(unsigned int indx) const;

    /// Does this driver require DisplayFilter results?
    bool requiresDisplayFilter() const;

    /// Does this render output require a DisplayFilter buffer?
    bool requiresDisplayFilter(unsigned int indx) const;

    unsigned int getDisplayFilterCount() const;

    /// Does this driver require a wireframe result?
    bool requiresWireframe() const;

    /// Does this driver require a motion vector result?
    bool requiresMotionVector() const;

    /// Material Aovs - main material aov manager.  Maps material aov
    /// names and types to aov schema ids.  Also resonpsible for computing
    /// material aov values from bsdf and bsdfv structures.
    const pbr::MaterialAovs &getMaterialAovs() const;

    /// Light Aovs - main light aov manager.  Maps light aov
    /// expressions to aov schema ids.  Also holds the light
    /// path state machine updated by the integrator as path
    /// events occur.
    const pbr::LightAovs &getLightAovs() const;

    /// Write the outputs : final output and non checkpoint file
    /// Errors are checked via errors()
    /// renderBuffer, aovBuffer, heatMap can be null if no output requires them
    void writeFinal(const pbr::DeepBuffer *deepBuffer,
                    pbr::CryptomatteBuffer *cryptomatteBuffer,
                    const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                    const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                    const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                    const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                    const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                    ImageWriteCache *cache) const;

    /// Write the checkpoint outputs
    /// Errors are checked via errors()
    /// renderBuffer, aovBuffer, heatMap can be null if no output requires them
    void writeCheckpointEnq(const bool checkpointMultiVersion,
                            const pbr::DeepBuffer *deepBuffer,
                            pbr::CryptomatteBuffer *cryptomatteBuffer,
                            const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                            const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                            const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                            const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                            const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                            const unsigned tileSampleTotals,
                            ImageWriteCache *outCache) const;
    void writeCheckpointDeq(ImageWriteCache *cache,
                            const bool checkpointMultiVersion) const;

    /// Errors and Infos during writing  are reported via a vector of strings. If empty, no error.
    /// In general, the class tries to output all the results it can sensibly
    /// figure out.  When it encounters a problem, it pushes an error and
    /// forges ahead.
    const std::vector<std::string> &getErrors() const;
    void resetErrors() const;

    /// An info message exists for each written render output.
    const std::vector<std::string> &getInfos() const;
    void resetInfos() const;

    /// When cache pointer is null, logging all errors go to Logger::error(),
    /// infos go to Logger::info(). Switch to the cache internal erros and infos
    /// if you set cache pointer instead RenderOutputDriver. errors and infos are clear after logging.
    /// return true if no error. otherwise false.
    bool loggingErrorAndInfo(ImageWriteCache *cache = nullptr) const;

    /// If the render output requires the renderBuffer, heatMapBuffer weightBuffer or
    /// renderBufferOdd as input, this function performs any needed final
    /// processing and transfers the result to buffer.
    void finishSnapshot(scene_rdl2::fb_util::VariablePixelBuffer *destBuffer, unsigned int indx,
                        const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                        const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                        const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                        const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                        bool parallel) const;

    /// Get the RenderOutput index to use as the albedo denoiser input
    /// Returns -1 if none.
    int getDenoiserAlbedoInput() const;

    /// Get the RenderOutput index to use as the normal input
    /// Returns -1 if none.
    int getDenoiserNormalInput() const;

    /// Revert Film data from resume file (just read and NOT DO any denormalize/zeroWeight operation)
    /// Returns false if error and call getErrors() for detail.
    bool revertFilmData(Film &film,
                        unsigned &resumeTileSamples, int &resumeNumConsistentSamples, bool &zeroWeightMask,
                        bool &adaptiveSampling, float adaptiveSampleParam[3]);

    void setLastCheckpointRenderTileSamples(const unsigned samples);

    //------------------------------

    scene_rdl2::grid_util::Parser& getParser();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace rndr
} // namespace moonray

