// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "DisplayFilterDriver.h"

#include "Film.h"
#include "RenderOutputDriver.h"

#include <moonray/rendering/displayfilter/DisplayFilter.h>
#include <moonray/rendering/displayfilter/InputBuffer.h>

#include <scene_rdl2/common/math/ispc/Typesv.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Viewport.h>
#include <scene_rdl2/scene/rdl2/DisplayFilter.h>

#define TILE_WIDTH 8

namespace moonray {
namespace rndr {

using namespace displayfilter;

class DisplayFilterDriver::Impl
{
public:
    Impl() = default;
    ~Impl();
    void init(rndr::Film* film,
              const rndr::RenderOutputDriver *roDriver,
              const std::vector<scene_rdl2::fb_util::Tile>* tiles,
              const uint32_t *tileIndices,
              const scene_rdl2::math::Viewport& viewport,
              unsigned int threadCount);

    bool hasDisplayFilters() const { return mRenderOutputDisplayFilterCount > 0; }

    // For snapshotting aovs.
    bool isAovRequired(unsigned int aovIdx) const { return mAovBuffers[aovIdx] != nullptr; }
    scene_rdl2::fb_util::VariablePixelBuffer * getAovBuffer(unsigned int aovIdx) const { return mAovBuffers[aovIdx]; }

    void runDisplayFilters(unsigned int tileIdx,
                           unsigned int threadId) const;

    void requestTileUpdate(unsigned int tileIdx) const;

private:

    // The DisplayFilterDriver has an internal DAG to organize which
    // display filters or input aovs need to be updated first. The DAG
    // contains nodes of 3 types:
    // DISPLAYFILTER Nodes are either root Nodes or interior nodes.
    // AOV Nodes are the leaf nodes of the DAG.
    // UNKNOWN Nodes are nodes created by a bad user input, such as a null
    // SceneObject, or invalid RenderOutput. These are also leaf nodes.
    // There should be either 0 or 1 UNKNOWN Nodes in the DAG.
    enum Type
    {
        UNKNOWN,
        DISPLAYFILTER,
        AOV
    };

    struct Node
    {
        Node() : mType(UNKNOWN),
                 mAltIndex(-1),
                 mDisplayFilter(nullptr),
                 mBuffer(nullptr)
        {
        }

        Node(Node&& other) : mType(other.mType),
                             mAltIndex(other.mAltIndex),
                             mDisplayFilter(other.mDisplayFilter),
                             mBuffer(std::move(other.mBuffer)),
                             mChildren(std::move(other.mChildren))
        {
        }

        void addChild(unsigned int index, int child)
        {
            if (index >= mChildren.size()) {
                mChildren.resize(index + 1);
            }
            mChildren[index] = child;
        }

        void clearChildren()
        {
            mChildren.clear();
        }

        void print()
        {
            std::cout << "type: " << mType << std::endl;
            std::cout << "alternative index: " << mAltIndex << std::endl;
            if (mDisplayFilter) std::cout << "display filter: " << mDisplayFilter->getName() << std::endl;
            std::cout << "children: ";
            for (int child : mChildren) std::cout << child << " ";
            std::cout << std::endl;
            std::cout << "buffer: " << mBuffer.get() << std::endl;
        }


        Type mType;
        // The index corresponding to a Film class pixel buffer.
        // It is either an Aov index or DisplayFilter index.
        int mAltIndex;
        // This is null for AOV (and UNKNOWN) Nodes
        const scene_rdl2::rdl2::DisplayFilter * mDisplayFilter;
        // Permanent storage of all post snapshot AOV and DisplayFilter pixel buffers.
        // These are full frame buffer that exist for the life of the DisplayFilterDriver.
        // They are owned by the DisplayFilterDriver, but can be shared with the InputBuffers
        // That are also owned by the DisplayFilterDriver.
        std::shared_ptr<scene_rdl2::fb_util::VariablePixelBuffer> mBuffer;
        // List of child node indices in DAG
        std::vector<int> mChildren;
    };

    /////////////// METHODS ///////////////
    typedef std::unordered_map<const scene_rdl2::rdl2::DisplayFilter *, displayfilter::InputData> InputDataMap;

    // initialization
    void clear();
    void createIndices(const uint32_t *tileIndices);
    void validateTiles(const scene_rdl2::math::Viewport& viewport);
    void initDisplayFilters(const rndr::RenderOutputDriver *roDriver, InputDataMap& inputDataMap);
    void createDAG(const rndr::RenderOutputDriver *roDriver,
                   const InputDataMap& inputDataMap,
                   std::vector<scene_rdl2::fb_util::VariablePixelBuffer *>& aovBuffers);
    void validateDAG(const rndr::RenderOutputDriver *roDriver,
                     const InputDataMap& inputDataMap);
    bool hasBadInputs(const Node& node) const;
    void prepareFatalBuffers();
    void createUpdateMask();
    void createInputBuffers(const InputDataMap& inputDataMap, unsigned int threadCount);

    // update mask
    void haltTileUpdate(int dfIdx, unsigned int tileIdx) const;
    bool updateRequested(int dfIdx, unsigned int tileIdx) const;

    // running display filters
    void grabAovInput(InputBuffer* destBuffer,
                      const scene_rdl2::fb_util::VariablePixelBuffer& sourceBuffer,
                      unsigned int tileIdx) const;
    void grabDisplayFilterInput(InputBuffer* destBuffer,
                                int dfIdx,
                                unsigned int tileIdx,
                                unsigned int threadId) const;

    void runDisplayFilter(int dfIdx,
                          unsigned int tileIdx,
                          unsigned int threadId) const;

    // static helper functions for initializing DAG
    static void initDisplayFilterRecurse(const scene_rdl2::rdl2::DisplayFilter * df,
                                         const moonray::displayfilter::InitializeData& initData,
                                         InputDataMap& inputDataMap);
    static void createChildDisplayFilterNodes(unsigned int nodeIdx,
                                              std::vector<Node>& dag,
                                              const rndr::RenderOutputDriver * roDriver,
                                              const rndr::Film& film,
                                              const InputDataMap& inputDataMap);
    static void createChildAovNodes(unsigned int nodeIdx,
                                    std::vector<Node>& dag,
                                    const rndr::RenderOutputDriver * roDriver,
                                    const rndr::Film& film,
                                    const InputDataMap& inputDataMap);
    static void createEmptyChildNode(unsigned int nodeIdx,
                                     std::vector<Node>& dag,
                                     const rndr::RenderOutputDriver * roDriver,
                                     const InputDataMap& inputDataMap);

    // getters
    inline unsigned int getWidth() const { return mWidth; }
    inline unsigned int getHeight() const { return mHeight; }

    /////////////// MEMBERS ///////////////

    // The layout of the DAG is
    //  RenderOutput DisplayFilters (roots)    intermediate DisplayFilters    AOVs (leaves)      empty node (leaf)
    // |************************************|******************************|********************|*
    // |                                    |                              |                    |
    // |  mRenderOutputDisplayFilterCount   |   mTotalDisplayFilterCount   | mDAG.size() - 1 -  |1
    //                                                                       mTotalDisplayFilterCount
    std::vector<Node> mDAG;
    std::vector<uint32_t> mLinearToPermutedTileIndices;
    std::vector<uint32_t> mPermutedToLinearTileIndices;
    // per DisplayFilter per tile update mask.
    // Indicates which tiles of which display filters need to be updated.
    mutable std::vector<std::vector<bool>> mUpdateMask;
    // Temporary buffers that are the inputs to the display filters.
    // per thread, per DisplayFilter, per input
    mutable std::vector<std::vector<std::vector<InputBuffer *>>> mInputBuffers;
    // snapshotted values of input aov buffers
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer *> mAovBuffers;

    // DisplayFilterDriver does not own these pointers
    rndr::Film *mFilm;
    const std::vector<scene_rdl2::fb_util::Tile> * mTiles;

    // critical indicies
    unsigned int mRenderOutputDisplayFilterCount;
    unsigned int mTotalDisplayFilterCount;

    // Frame stuff
    unsigned int mWidth;
    unsigned int mHeight;
    int mNumTilesX;
    int mNumTilesY;
};

DisplayFilterDriver::Impl::~Impl()
{
    clear();
}

void
DisplayFilterDriver::Impl::init(rndr::Film *film,
                                const rndr::RenderOutputDriver *roDriver,
                                const std::vector<scene_rdl2::fb_util::Tile>* tiles,
                                const uint32_t *tileIndices,
                                const scene_rdl2::math::Viewport& viewport,
                                unsigned int threadCount)
{
    clear();

    mFilm = film;
    mTiles = tiles;
    mWidth = mFilm->getAlignedWidth();
    mHeight = mFilm->getAlignedHeight();
    // This is the formula for the number of tiles defined in the TileScheduler.
    // See TileScheduler::generateTiles().
    mNumTilesX = (viewport.mMaxX / TILE_WIDTH) - (viewport.mMinX / TILE_WIDTH) + 1;
    mNumTilesY = (viewport.mMaxY / TILE_WIDTH) - (viewport.mMinY / TILE_WIDTH) + 1;

    InputDataMap inputDataMap;
    createIndices(tileIndices);
    validateTiles(viewport);
    initDisplayFilters(roDriver, inputDataMap);
    createDAG(roDriver, inputDataMap, mAovBuffers);
    validateDAG(roDriver, inputDataMap);
    prepareFatalBuffers();
    createUpdateMask();
    createInputBuffers(inputDataMap, threadCount);
}

void
DisplayFilterDriver::Impl::requestTileUpdate(unsigned int tileIdx) const
{
    for (int dfIdx = 0; dfIdx < mTotalDisplayFilterCount; ++dfIdx) {
        mUpdateMask[dfIdx][tileIdx] = true;
    }
}

void
DisplayFilterDriver::Impl::haltTileUpdate(int dfIdx, unsigned int tileIdx) const
{
    MNRY_ASSERT(dfIdx < mTotalDisplayFilterCount);
    mUpdateMask[dfIdx][tileIdx] = false;
}

bool
DisplayFilterDriver::Impl::updateRequested(int dfIdx, unsigned int tileIdx) const
{
    MNRY_ASSERT(dfIdx < mTotalDisplayFilterCount);
    return mUpdateMask[dfIdx][tileIdx];
}

void
DisplayFilterDriver::Impl::clear()
{
    for (auto& perThreadInputBuffers : mInputBuffers) {
        for (auto& inputBuffers : perThreadInputBuffers) {
            for (InputBuffer * inputBuffer : inputBuffers) {
                if (inputBuffer) {
                    // inputBuffers are not smart pointers to make it easier
                    // to pass them to ispc. So we must manually delete them.
                    delete inputBuffer;
                }
            }
        }
    }
    mInputBuffers.clear();

    mDAG.clear();
    mUpdateMask.clear();
    mLinearToPermutedTileIndices.clear();
    mPermutedToLinearTileIndices.clear();
    mAovBuffers.clear();

    mFilm = nullptr;
    mTiles = nullptr;
    mRenderOutputDisplayFilterCount = 0;
    mTotalDisplayFilterCount = 0;
}

void
DisplayFilterDriver::Impl::createIndices(const uint32_t *tileIndices)
{
    MNRY_ASSERT(mTiles);
    // Set tile indices
    mLinearToPermutedTileIndices.resize(mTiles->size());
    mPermutedToLinearTileIndices.resize(mTiles->size());
    memcpy(mLinearToPermutedTileIndices.data(), tileIndices, mLinearToPermutedTileIndices.size() * sizeof(uint32_t));
    for (unsigned int i = 0 ; i < mPermutedToLinearTileIndices.size(); ++i) {
        mPermutedToLinearTileIndices[mLinearToPermutedTileIndices[i]] = i;
    }
}

void
DisplayFilterDriver::Impl::validateTiles(const scene_rdl2::math::Viewport& viewport)
{
    for (const int& tileIdx : mLinearToPermutedTileIndices) {
        MNRY_ASSERT_REQUIRE(tileIdx < mTiles->size());
        const scene_rdl2::fb_util::Tile& tile = (*mTiles)[tileIdx];
        MNRY_ASSERT_REQUIRE(tile.mMinX % TILE_WIDTH == 0 || tile.mMinX == viewport.mMinX);
        MNRY_ASSERT_REQUIRE(tile.mMinY % TILE_WIDTH == 0 || tile.mMinY == viewport.mMinY);
        // tile max coordinate is exclusive. viewport max coordinate is inclusive.
        MNRY_ASSERT_REQUIRE(tile.mMaxX % TILE_WIDTH == 0 || tile.mMaxX == viewport.mMaxX + 1);
        MNRY_ASSERT_REQUIRE(tile.mMaxY % TILE_WIDTH == 0 || tile.mMaxY == viewport.mMaxY + 1);
        MNRY_ASSERT_REQUIRE(tile.mMaxX - tile.mMinX <= TILE_WIDTH);
        MNRY_ASSERT_REQUIRE(tile.mMaxY - tile.mMinY <= TILE_WIDTH);
    }
}

void
DisplayFilterDriver::Impl::initDisplayFilters(const rndr::RenderOutputDriver *roDriver,
                                              InputDataMap& inputDataMap)
{
    moonray::displayfilter::InitializeData initData(getWidth(), getHeight());

    unsigned int roCount = roDriver->getNumberOfRenderOutputs();
    for (unsigned int roIdx = 0; roIdx < roCount; ++roIdx) {
        if (roDriver->requiresDisplayFilter(roIdx)) {
            const scene_rdl2::rdl2::DisplayFilter *df = roDriver->getRenderOutput(roIdx)->getDisplayFilter();
            initDisplayFilterRecurse(df, initData, inputDataMap);
        }
    }
}

void
DisplayFilterDriver::Impl::initDisplayFilterRecurse(const scene_rdl2::rdl2::DisplayFilter * df,
                                                    const moonray::displayfilter::InitializeData& initData,
                                                    DisplayFilterDriver::Impl::InputDataMap& inputDataMap)
{
    if (inputDataMap.find(df) != inputDataMap.end()) {
        // already initialized
        return;
    }
    InputData inputData;
    df->getInputData(initData, inputData);
    MNRY_ASSERT(inputData.mInputs.size() == inputData.mWindowWidths.size());
    inputDataMap[df] = inputData;
    const scene_rdl2::rdl2::SceneObjectVector& inputs = inputData.mInputs;

    for (scene_rdl2::rdl2::SceneObject * input : inputs) {
        if (input == nullptr) {
            continue;
        }
        const scene_rdl2::rdl2::DisplayFilter * inputDf = nullptr;
        if (input->isA<scene_rdl2::rdl2::RenderOutput>()) {
            scene_rdl2::rdl2::RenderOutput * ro = input->asA<scene_rdl2::rdl2::RenderOutput>();
            if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                inputDf = ro->getDisplayFilter();
            }
        } else if (input->isA<scene_rdl2::rdl2::DisplayFilter>()) {
            inputDf = input->asA<scene_rdl2::rdl2::DisplayFilter>();
        }

        if (inputDf != nullptr) {
            initDisplayFilterRecurse(inputDf, initData, inputDataMap);
        }
    }
}

void
DisplayFilterDriver::Impl::createDAG(const rndr::RenderOutputDriver *roDriver,
                                     const InputDataMap& inputDataMap,
                                     std::vector<scene_rdl2::fb_util::VariablePixelBuffer *>& aovBuffers)
{
    MNRY_ASSERT(mDAG.empty());

    // RenderOutput DisplayFilters go first
    unsigned int roCount = roDriver->getNumberOfRenderOutputs();
    for (unsigned int roIdx = 0; roIdx < roCount; ++roIdx) {
        if (roDriver->requiresDisplayFilter(roIdx)) {
            Node node;
            node.mType = DISPLAYFILTER;
            node.mAltIndex = roDriver->getDisplayFilterIndex(roIdx);
            node.mDisplayFilter = roDriver->getRenderOutput(roIdx)->getDisplayFilter();
            node.mBuffer.reset(new scene_rdl2::fb_util::VariablePixelBuffer());
            node.mBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3,
                               mFilm->getAlignedWidth(),
                               mFilm->getAlignedHeight());
            mDAG.push_back(std::move(node));
        }
    }
    mRenderOutputDisplayFilterCount = mDAG.size();

    // intermediate DisplayFilters go next
    for (unsigned int ro = 0; ro < mRenderOutputDisplayFilterCount; ++ro) {
        createChildDisplayFilterNodes(ro, mDAG, roDriver, *mFilm, inputDataMap);
    }
    mTotalDisplayFilterCount = mDAG.size();

    // AOVs go last
    for (unsigned int df = 0; df < mTotalDisplayFilterCount; ++df) {
        createChildAovNodes(df, mDAG, roDriver, *mFilm, inputDataMap);
    }

    // set aovBuffers
    aovBuffers.clear();
    unsigned int aovCount = 0;
    for(unsigned int ro = 0; ro < roDriver->getNumberOfRenderOutputs(); ++ro) {
        if (roDriver->getAovBuffer(ro) >= 0) {
            aovCount++;
        }
    }
    aovBuffers.resize(aovCount, nullptr);
    for (unsigned int aov = mTotalDisplayFilterCount; aov < mDAG.size(); ++aov) {
        const Node& node = mDAG[aov];
        MNRY_ASSERT(node.mType == AOV);
        aovBuffers[node.mAltIndex] = node.mBuffer.get();
    }

    // Empty input goes very last if needed. And Empty input might occur due to
    // bad user input.
    for (unsigned int df = 0; df < mTotalDisplayFilterCount; ++df) {
        createEmptyChildNode(df, mDAG, roDriver, inputDataMap);
    }
}

void
DisplayFilterDriver::Impl::createChildDisplayFilterNodes(unsigned int nodeIdx,
                                                         std::vector<Node>& dag,
                                                         const rndr::RenderOutputDriver * roDriver,
                                                         const rndr::Film& film,
                                                         const InputDataMap& inputDataMap)
{
    if (dag[nodeIdx].mType != DISPLAYFILTER) {
        return;
    }

    // get input SceneObjects
    const scene_rdl2::rdl2::DisplayFilter * df = dag[nodeIdx].mDisplayFilter;
    const scene_rdl2::rdl2::SceneObjectVector& inputs = inputDataMap.at(df).mInputs;

    for (unsigned int inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
        const scene_rdl2::rdl2::SceneObject * input = inputs[inputIndex];
        if (input == nullptr) {
            // invalid input
            continue;
        }
        Node& currentNode = dag[nodeIdx];
        const scene_rdl2::rdl2::DisplayFilter * df = nullptr;
        int dfIdx = -1;
        if (input->isA<scene_rdl2::rdl2::RenderOutput>()) {
            const scene_rdl2::rdl2::RenderOutput * ro = input->asA<scene_rdl2::rdl2::RenderOutput>();
            if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                df = ro->getDisplayFilter();
                dfIdx = roDriver->getDisplayFilterIndex(roDriver->getRenderOutputIndx(ro));
            } else {
                // probably an aov input
                continue;
            }
        } else if (input->isA<scene_rdl2::rdl2::DisplayFilter>()) {
            df = input->asA<scene_rdl2::rdl2::DisplayFilter>();
        } else {
            // invalid input
            continue;
        }

        MNRY_ASSERT(df);

        // check if input is already in dag and update children indices
        const auto it = std::find_if(dag.begin(), dag.end(), [&](const Node& node) {
                                        return node.mDisplayFilter == df;
                                    });
        if (it != dag.end()) {
            // child already exists in dag
            currentNode.addChild(inputIndex, it - dag.begin());
            continue;
        }

        // create new node
        currentNode.addChild(inputIndex, dag.size());
        Node newNode;
        newNode.mType = DISPLAYFILTER;
        newNode.mAltIndex = dfIdx;
        newNode.mDisplayFilter = df;
        newNode.mBuffer.reset(new scene_rdl2::fb_util::VariablePixelBuffer());
        newNode.mBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3,
                              film.getAlignedWidth(),
                              film.getAlignedHeight());
        dag.push_back(std::move(newNode));

        // recurse on input DisplayFilter Node
        createChildDisplayFilterNodes(dag.size() - 1, dag, roDriver, film, inputDataMap);
    }
}

void
DisplayFilterDriver::Impl::createChildAovNodes(unsigned int nodeIdx,
                                               std::vector<Node>& dag,
                                               const rndr::RenderOutputDriver * roDriver,
                                               const rndr::Film& film,
                                               const InputDataMap& inputDataMap)
{
    if (dag[nodeIdx].mType != DISPLAYFILTER) {
        return;
    }

    // get input SceneObjects
    const scene_rdl2::rdl2::DisplayFilter * df = dag[nodeIdx].mDisplayFilter;
    const scene_rdl2::rdl2::SceneObjectVector& inputs = inputDataMap.at(df).mInputs;

    for (unsigned int inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
        const scene_rdl2::rdl2::SceneObject * input = inputs[inputIndex];
        if (input == nullptr) {
            // invalid input
            continue;
        }
        if (!input->isA<scene_rdl2::rdl2::RenderOutput>()) {
            // This was not a RenderOutput object, likely a DisplayFilterObject.
            continue;
        }
        Node& currentNode = dag[nodeIdx];
        const scene_rdl2::rdl2::RenderOutput * ro = input->asA<scene_rdl2::rdl2::RenderOutput>();

        // check if input is already in dag and update children indexes
        const auto it = std::find_if(dag.begin(), dag.end(), [&](const Node& node) {
                                        int roIdx = roDriver->getRenderOutputIndx(ro);
                                        return node.mType == AOV &&
                                               node.mAltIndex == roDriver->getAovBuffer(roIdx);;
                                    });
        if (it != dag.end()) {
            // child already exists in dag
            currentNode.addChild(inputIndex, it - dag.begin());
            continue;
        }


        if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
            continue;
        }
        int aovIdx = roDriver->getAovBuffer(roDriver->getRenderOutputIndx(ro));
        if (aovIdx < 0) {
            continue;
        }

        // create new node
        currentNode.addChild(inputIndex, dag.size());
        Node newNode;
        newNode.mType = AOV;
        newNode.mAltIndex = aovIdx;
        newNode.mDisplayFilter = nullptr;
        const scene_rdl2::fb_util::VariablePixelBuffer& filmAovBuf = film.getAovBuffer(newNode.mAltIndex);
        newNode.mBuffer.reset(new scene_rdl2::fb_util::VariablePixelBuffer());
        scene_rdl2::fb_util::VariablePixelBuffer::Format format = filmAovBuf.getFormat();
        if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV) {
            // special cases where snapshot format does not match film buffer format
            format = scene_rdl2::fb_util::VariablePixelBuffer::FLOAT;
        }
        newNode.mBuffer->init(format, filmAovBuf.getWidth(), filmAovBuf.getHeight());
        dag.push_back(std::move(newNode));
    }
}

void
DisplayFilterDriver::Impl::createEmptyChildNode(unsigned int nodeIdx,
                                                std::vector<Node>& dag,
                                                const rndr::RenderOutputDriver * roDriver,
                                                const InputDataMap& inputDataMap)
{
    if (dag[nodeIdx].mType != DISPLAYFILTER) {
        return;
    }

    // get input SceneObjects
    const scene_rdl2::rdl2::DisplayFilter * df = dag[nodeIdx].mDisplayFilter;
    const scene_rdl2::rdl2::SceneObjectVector& inputs = inputDataMap.at(df).mInputs;

    for (unsigned int inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
        const scene_rdl2::rdl2::SceneObject * input = inputs[inputIndex];
        bool addEmptyChildNode = (input == nullptr); // null input
        if (!addEmptyChildNode) {
            if (input->isA<scene_rdl2::rdl2::RenderOutput>()) {
                const scene_rdl2::rdl2::RenderOutput* ro = input->asA<scene_rdl2::rdl2::RenderOutput>();
                if (ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER &&
                    roDriver->getAovBuffer(roDriver->getRenderOutputIndx(ro)) < 0) {
                    // input is not a display filter nor a valid aov
                    addEmptyChildNode = true;
                }
            }
        }

        if (addEmptyChildNode) {
            Node& currentNode = dag[nodeIdx];
            // check if empty input is already in dag and update children indexes
            const auto it = std::find_if(dag.begin(), dag.end(), [&](const Node& node) {
                                            return node.mType == UNKNOWN;
                                        });

            if (it != dag.end()) {
                // child already exists in dag
                currentNode.addChild(inputIndex, it - dag.begin());
                continue;
            }

            currentNode.addChild(inputIndex, dag.size());
            Node newNode;
            newNode.mType = UNKNOWN;
            newNode.mAltIndex = -1;
            newNode.mDisplayFilter = nullptr;
            newNode.mBuffer.reset();
            dag.push_back(std::move(newNode));
        }
    }
}

void
DisplayFilterDriver::Impl::validateDAG(const rndr::RenderOutputDriver * roDriver,
                                       const InputDataMap& inputDataMap)
{
    for (unsigned int i = 0; i < mDAG.size(); ++i) {
        const Node& node = mDAG[i];
        // make sure node is correct type
        MNRY_ASSERT_REQUIRE((node.mType == DISPLAYFILTER) ^ (node.mDisplayFilter == nullptr));
        MNRY_ASSERT_REQUIRE((node.mType == DISPLAYFILTER && i < mTotalDisplayFilterCount) ||
                           (node.mType == AOV && i >= mTotalDisplayFilterCount) ||
                           (node.mType == UNKNOWN && i == mDAG.size() - 1));

        if (node.mType != DISPLAYFILTER) {
            // This node should not have any children if it is not a display filter.
            MNRY_ASSERT_REQUIRE(node.mChildren.size() == 0);
            continue;
        }

        // If node is a Display Filter, we must check the children / inputs.
        // An "input" is the raw rdl SceneObject that is the input to the scene_rdl2::rdl2::DisplayFilter object.
        // A "child" is a node in the dag that is a child of another node in the dag. Its data
        // is taken from the "input", but sometimes a programmer error can jumble the data.
        // We are validating the the data in the "input" agrees with the data in the "child".
        // The order of the "children" must match the order of the "inputs".
        const std::vector<int> children = node.mChildren;
        const scene_rdl2::rdl2::SceneObjectVector& inputs = inputDataMap.at(node.mDisplayFilter).mInputs;

        // There should be the same number of children in the node as there are inputs to the
        // display filter.
        MNRY_ASSERT_REQUIRE(children.size() == inputs.size());
        unsigned int numChildren = children.size();

        for (unsigned int j = 0; j < numChildren; ++j) {
            const scene_rdl2::rdl2::SceneObject * input = inputs[j];
            int childIndex = children[j];
            // child must be in the DAG
            MNRY_ASSERT_REQUIRE(childIndex >= 0 && childIndex < mDAG.size());
            const Node& child = mDAG[childIndex];

            if (child.mType == UNKNOWN) {
                // An unknown child must be the last element of the dag.
                MNRY_ASSERT_REQUIRE(childIndex == mDAG.size() - 1);

            } else if (child.mType == AOV) {
                // An aov child must come from a RenderOutput with a valid aov
                MNRY_ASSERT_REQUIRE(input->isA<scene_rdl2::rdl2::RenderOutput>());
                const scene_rdl2::rdl2::RenderOutput * ro = input->asA<scene_rdl2::rdl2::RenderOutput>();
                int aovIdx = roDriver->getAovBuffer(roDriver->getRenderOutputIndx(ro));
                // The "input" and the child must share the same index
                MNRY_ASSERT_REQUIRE(aovIdx >= 0 && aovIdx == child.mAltIndex);

            } else if (child.mType == DISPLAYFILTER) {
                // A display filter child can come from a RenderOutput with a valid DsiplayFilter
                // or directly from a DisplayFilter. The diplay filter in the child node and the
                // display filter from the "input" must have the same name (i.e. be the same
                // display filter).
                if (input->isA<scene_rdl2::rdl2::RenderOutput>()) {
                    const scene_rdl2::rdl2::RenderOutput * ro = input->asA<scene_rdl2::rdl2::RenderOutput>();
                    MNRY_ASSERT_REQUIRE(child.mDisplayFilter &&
                                       ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER &&
                                       child.mDisplayFilter->getName() == ro->getDisplayFilter()->getName());

                } else {
                    MNRY_ASSERT_REQUIRE(input->isA<scene_rdl2::rdl2::DisplayFilter>() &&
                                       child.mDisplayFilter &&
                                       child.mDisplayFilter->getName() == input->asA<scene_rdl2::rdl2::DisplayFilter>()->getName());
                }
            }
        }
    }
}

bool
DisplayFilterDriver::Impl::hasBadInputs(const Node& node) const
{
    for (int inputIdx : node.mChildren) {
        if (mDAG[inputIdx].mType == UNKNOWN) {
            return true;
        }
    }

    return false;
}

void
DisplayFilterDriver::Impl::prepareFatalBuffers()
{
    for (unsigned int nodeIdx = 0; nodeIdx < mTotalDisplayFilterCount; ++nodeIdx) {
        Node& node = mDAG[nodeIdx];
        MNRY_ASSERT(node.mType == DISPLAYFILTER);
        if (hasBadInputs(node)) {
            // if this node has bad inputs, then its buffer is filled with the fatal color.
            const scene_rdl2::math::Color fatalColor = node.mDisplayFilter->getSceneClass().getSceneContext()->
                getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFatalColor);
            node.mBuffer->getFloat3Buffer().clear(scene_rdl2::math::Vec3f(fatalColor.r, fatalColor.g, fatalColor.b));
        }
    }
}

void
DisplayFilterDriver::Impl::createUpdateMask()
{
    MNRY_ASSERT(mTiles);
    mUpdateMask.resize(mTotalDisplayFilterCount);
    for (std::vector<bool>& mask : mUpdateMask) {
        mask.resize(mTiles->size(), true);
    }
}

void
DisplayFilterDriver::Impl::createInputBuffers(const DisplayFilterDriver::Impl::InputDataMap& inputDataMap,
                                              unsigned int threadCount)
{
    // Each thread get local input buffer storage.
    mInputBuffers.resize(threadCount);
    for (std::vector<std::vector<InputBuffer *>>& perThreadInputBuffers : mInputBuffers) {
        // Each display filter has its own set of input buffers
        perThreadInputBuffers.resize(mTotalDisplayFilterCount);

        for (unsigned int dfIdx = 0; dfIdx < mTotalDisplayFilterCount; ++dfIdx) {
            const Node& node = mDAG[dfIdx];
            MNRY_ASSERT(node.mType == DISPLAYFILTER && node.mDisplayFilter);

            const std::vector<int>& windowWidths = inputDataMap.at(node.mDisplayFilter).mWindowWidths;
            const std::vector<int>& children = node.mChildren;
            MNRY_ASSERT(children.size() == windowWidths.size());
            std::vector<InputBuffer *>& inputBuffers = perThreadInputBuffers[dfIdx];
            inputBuffers.resize(children.size());

            // initialize each input buffer
            for (unsigned int idx = 0; idx < children.size(); ++idx) {
                const Node& childNode = mDAG[children[idx]];
                if (childNode.mType == UNKNOWN) {
                    // invalid input
                    inputBuffers[idx] = nullptr;
                    continue;
                }
                inputBuffers[idx] = new InputBuffer();
                if (windowWidths[idx] == 0) {
                    // special case if request window width is 0.
                    // We want the entire frame buffer so we assign
                    // the Node's pixel buffer to the InputBuffer.
                    inputBuffers[idx]->init(childNode.mBuffer);
                    inputBuffers[idx]->setStartPixel(0,0);
                    continue;
                }

                int windowWidth = windowWidths[idx] - 1;
                // Round to nearest tile width. Tile width is 8 pixels.
                // VLEN pixels are submitted to a display filter at a time.
                // If there are n tiles to either side of the center tile(s)
                // that fit VLEN pixels, then windowWidth must be (VLEN / 8 + 2n) * 8 pixels.
                windowWidth = ((VLEN / TILE_WIDTH) /* number of tiles needed for VLEN pixels */
                            + 2 * (windowWidth / TILE_WIDTH + (windowWidth % TILE_WIDTH > 0))) /* 2n */
                            * TILE_WIDTH; /* 8 pixels */

                scene_rdl2::fb_util::VariablePixelBuffer::Format format = childNode.mBuffer->getFormat();
                inputBuffers[idx]->init(format, windowWidth, windowWidth);
            }
        }
    }
}

void
DisplayFilterDriver::Impl::runDisplayFilters(unsigned int tileIdx,
                                             unsigned int threadId) const
{
    if(mDAG.empty()) {
        return;
    }

    for (int dfIdx = 0; dfIdx < mRenderOutputDisplayFilterCount; ++dfIdx) {
        // execute filter function for this DisplayFilter
        runDisplayFilter(dfIdx, tileIdx, threadId);

        // update DisplayFilter buffer stored in rndr::Film.
        const scene_rdl2::fb_util::Tile& tile = (*mTiles)[tileIdx];
        for (unsigned int y = tile.mMinY; y < tile.mMaxY; ++y) {
            const auto& dfBuffer = mDAG[dfIdx].mBuffer;
            // Add entire row of pixels at once.
            mFilm->addTileSamplesToDisplayFilterBuffer(mDAG[dfIdx].mAltIndex,
                tile.mMinX, y,
                tile.mMaxX - tile.mMinX,
                reinterpret_cast<const uint8_t *>(&dfBuffer->getFloat3Buffer().getPixel(tile.mMinX, y)[0]));
        }
    }
}

void
DisplayFilterDriver::Impl::runDisplayFilter(int dfIdx,
                                            unsigned int tileIdx,
                                            unsigned int threadId) const
{
    MNRY_ASSERT(dfIdx < mTotalDisplayFilterCount);

    if (!updateRequested(dfIdx, tileIdx)) {
        return;
    }

    if(hasBadInputs(mDAG[dfIdx])) {
        // No need to do anything. Buffer is already filled with fatal color.
        return;
    }

    const Node& node = mDAG[dfIdx];
    const scene_rdl2::rdl2::DisplayFilter * const df = node.mDisplayFilter;
    MNRY_ASSERT(df);
    // get inputs
    const std::vector<InputBuffer *>& inputBuffers = mInputBuffers[threadId][dfIdx];
    MNRY_ASSERT(node.mChildren.size() == inputBuffers.size());

    for (unsigned int idx = 0; idx < node.mChildren.size(); ++idx) {
        int inputIdx = node.mChildren[idx];
        const Node& inputNode = mDAG[inputIdx];
        if (inputNode.mType == DISPLAYFILTER) {
            InputBuffer* inputBuffer = inputBuffers[idx];
            grabDisplayFilterInput(inputBuffer, inputIdx, tileIdx, threadId);
        } else if (inputNode.mType == AOV) {
            InputBuffer* inputBuffer = inputBuffers[idx];
            grabAovInput(inputBuffer, *inputNode.mBuffer, tileIdx);
        } else {
            MNRY_ASSERT(0);
        }
    }

    // run filter
    const scene_rdl2::fb_util::Tile& tile = (*mTiles)[tileIdx];
    const unsigned int numRows = tile.mMaxY - tile.mMinY;
    const unsigned int numColumns = tile.mMaxX - tile.mMinX;
    for (unsigned int row = 0; row < numRows; ++row) {
        DisplayFilterStatev statev;
        for (int i = 0; i < VLEN; ++i) {
            int outputX = i < numColumns ? tile.mMinX + i : tile.mMaxX -1;
            statev.mOutputPixelX[i] = outputX;
            statev.mOutputPixelY[i] = tile.mMinY + row;
            statev.mImageWidth[i] = getWidth();
            statev.mImageHeight[i] = getHeight();
        }

        scene_rdl2::math::Colorv result;
        memset(&result, 0, sizeof(scene_rdl2::math::Colorv));

        df->filterv(reinterpret_cast<const scene_rdl2::rdl2::DisplayFilterInputBufferv * const *>(inputBuffers.data()),
                    reinterpret_cast<const scene_rdl2::rdl2::DisplayFilterStatev *>(&statev),
                    reinterpret_cast<scene_rdl2::rdl2::Colorv *>(&result));

        for (int i = 0; i < numColumns; ++i) {
            node.mBuffer->getFloat3Buffer().setPixel(tile.mMinX + i,  tile.mMinY + row,
                scene_rdl2::math::Vec3f(result.r[i], result.g[i], result.b[i]));
        }
    }

    // The update for this tile of this DisplayFilter is complete.
    haltTileUpdate(dfIdx, tileIdx);
}

// If destination buffer goes out of bounds of
// source buffer, clamp the pixels to the edge.
void clampRow(const uint8_t * srcData,
              const int length,
              InputBuffer* destBuffer,
              unsigned int& offset)
{
    for (int i = 0; i < length; ++i) {
        destBuffer->setData(offset, 1, srcData);
        ++offset;
    }
}

void
DisplayFilterDriver::Impl::grabDisplayFilterInput(InputBuffer* destBuffer,
                                                  int dfIdx,
                                                  unsigned int tileIdx,
                                                  unsigned int threadId) const
{
    const scene_rdl2::fb_util::VariablePixelBuffer& srcBuffer = *mDAG[dfIdx].mBuffer;
    MNRY_ASSERT(destBuffer->getFormat() == srcBuffer.getFormat());

    const unsigned int width = srcBuffer.getWidth();
    const unsigned int sizeOfPixel = srcBuffer.getSizeOfPixel();

    // find tileIdx of adjacent tiles.
    const unsigned int tileIdxLinear = mPermutedToLinearTileIndices[tileIdx];
    const int tileX = tileIdxLinear % mNumTilesX;
    const int tileY = tileIdxLinear / mNumTilesX;

    const int windowWidth = destBuffer->getWidth();
    // Compute number of tiles on either side of this tile.
    // A windowWidth, measured in pixels, is contained within
    // 2n + 1 tiles. Compute n. Assumes 8 pixels per tile.
    const int tileBuffer = (windowWidth - TILE_WIDTH ) / (2 * TILE_WIDTH);
    const int minTileX = std::max(0, tileX - tileBuffer);
    const int maxTileX = std::min(mNumTilesX - 1, tileX + tileBuffer); // inclusive
    const int minTileY = std::max(0, tileY - tileBuffer);
    const int maxTileY = std::min(mNumTilesY - 1, tileY + tileBuffer); // inclusive

    unsigned offsetY = 0;
    for (int ty = tileY - tileBuffer; ty <= tileY + tileBuffer; ++ty) {
        unsigned y = ty;
        if (ty < minTileY) { y = minTileY; }
        if (ty > maxTileY) { y = maxTileY; }

        unsigned offsetX = 0;
        for (int tx = tileX - tileBuffer; tx <= tileX + tileBuffer; ++tx) {
            unsigned x = tx;
            if (tx < minTileX) { x = minTileX; }
            if (tx > maxTileX) { x = maxTileX; }

            unsigned int currTileIdx = mLinearToPermutedTileIndices[y * mNumTilesX + x];
            if (updateRequested(dfIdx, currTileIdx)) {
                // The display filter at this tile has outdated data.
                // Run the display filter before gathering its output pixels.
                runDisplayFilter(dfIdx, currTileIdx, threadId);
            }

            if (destBuffer->getPixelBuffer() == &srcBuffer) {
                // destBuffer is already assigned to srcBuffer.
                // No need to copy pixels. This occurs when
                // the requested window width is the entire image frame.
            } else {
                // Copy pixels from global source buffer to local per thread destination buffer.
                const scene_rdl2::fb_util::Tile& tile = (*mTiles)[currTileIdx];
                const unsigned int px = tile.mMinX;
                const unsigned int length = tile.mMaxX - tile.mMinX;
                for (unsigned int py = tile.mMinY; py < tile.mMinY + TILE_WIDTH; ++py) {
                    unsigned int pixelY = py < tile.mMaxY ? py : tile.mMaxY - 1;
                    const uint8_t * data = srcBuffer.getData() + (pixelY * width + px) * sizeOfPixel;
                    // get pixel offset into local destination buffer
                    unsigned int offset = (offsetY + py - tile.mMinY) * windowWidth + offsetX;
                    destBuffer->setData(offset, length, data);
                    if (length < TILE_WIDTH) {
                        offset += length;
                        clampRow(data + length * sizeOfPixel, TILE_WIDTH - length, destBuffer, offset);
                    }
                }
            }
            offsetX += TILE_WIDTH;
        }
        offsetY += TILE_WIDTH;
    }

    if (destBuffer->getPixelBuffer() == &srcBuffer) {
        // destBuffer is already assigned to srcBuffer.
        // No need to set start pixel. This occurs when
        // the requested window width is the entire image frame.
    } else {
        destBuffer->setStartPixel((int)(*mTiles)[tileIdx].mMinX - TILE_WIDTH * tileBuffer,
                                  (int)(*mTiles)[tileIdx].mMinY - TILE_WIDTH * tileBuffer);
    }
}

void
DisplayFilterDriver::Impl::grabAovInput(InputBuffer* destBuffer,
                                        const scene_rdl2::fb_util::VariablePixelBuffer& srcBuffer,
                                        unsigned int tileIdx) const
{
    MNRY_ASSERT(destBuffer->getFormat() == srcBuffer.getFormat());
    if (destBuffer->getPixelBuffer() == &srcBuffer) {
        // destBuffer is already assigned to srcBuffer.
        // No need to copy pixels. This occurs when
        // the requested window width is the entire image frame.
        return;
    }

    const unsigned width = srcBuffer.getWidth();
    const unsigned height = srcBuffer.getHeight();
    const unsigned sizeOfPixel = srcBuffer.getSizeOfPixel();
    const int windowBuffer = (destBuffer->getWidth() - TILE_WIDTH) / 2; // pixels

    // get adjacent tiles if needed.
    const scene_rdl2::fb_util::Tile& tile = (*mTiles)[tileIdx];
    // values are in pixels
    const int minX = tile.mMinX - windowBuffer;
    const int maxX = tile.mMinX + TILE_WIDTH + windowBuffer;
    const int minY = tile.mMinY - windowBuffer;
    const int maxY = tile.mMinY + TILE_WIDTH + windowBuffer;

    unsigned offset = 0;
    for (int iy = minY; iy < maxY; ++iy) {
        unsigned int ry = iy;
        if (iy < 0) ry = 0;
        if (iy >= height) ry = height - 1;
        int rx = minX;

        if (rx < 0) {
            // If the window goes past the left edge of the image, fill
            // the region of the window that does not overlap the image with
            // the pixel value at x = 0. (essentially a clamping operation)
            const int length = -rx;
            // get first element of row
            const uint8_t * data = srcBuffer.getData() + ry * width * sizeOfPixel;
            clampRow(data, length, destBuffer, offset);
            rx = 0;
        }

        const unsigned rowLength = maxX > width ? width - rx : maxX - rx;

        // copy row
        const uint8_t * data = srcBuffer.getData() + (ry * width + rx) * sizeOfPixel;
        destBuffer->setData(offset, rowLength, data);
        offset += rowLength;

        // fill in end
        if (maxX > width) {
            // If window goes past the right edge of the image, fill
            // the region of the window that does not overlap the image with
            // the pixel value at x = width. (essentially a clamping operation)
            const unsigned length = maxX - width;
            const uint8_t * data = srcBuffer.getData() + (ry * width + width - 1) * sizeOfPixel;
            clampRow(data, length, destBuffer, offset);
        }
    }

    destBuffer->setStartPixel(minX, minY);
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

DisplayFilterDriver::DisplayFilterDriver()
{
    mImpl.reset(new Impl());
}

DisplayFilterDriver::~DisplayFilterDriver()
{
    mImpl.reset();
}

void
DisplayFilterDriver::init(rndr::Film *film,
                          const rndr::RenderOutputDriver *roDriver,
                          const std::vector<scene_rdl2::fb_util::Tile>* tiles,
                          const uint32_t *tileIndices,
                          const scene_rdl2::math::Viewport& viewport,
                          unsigned int threadCount)
{
    mImpl->init(film, roDriver, tiles, tileIndices, viewport, threadCount);
}

bool
DisplayFilterDriver::hasDisplayFilters() const
{
    return mImpl->hasDisplayFilters();
}

bool
DisplayFilterDriver::isAovRequired(uint32_t aovIdx) const
{
    return mImpl->isAovRequired(aovIdx);
}

scene_rdl2::fb_util::VariablePixelBuffer *
DisplayFilterDriver::getAovBuffer(uint32_t aovIdx) const
{
    return mImpl->getAovBuffer(aovIdx);
}

void
DisplayFilterDriver::requestTileUpdate(unsigned int tileIdx) const
{
    mImpl->requestTileUpdate(tileIdx);
}

void
DisplayFilterDriver::runDisplayFilters(unsigned int tileIdx,
                                       unsigned int threadId) const
{
    mImpl->runDisplayFilters(tileIdx, threadId);
}

} // namespace displayfilter
} // namespace moonray

