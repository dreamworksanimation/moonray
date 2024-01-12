// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/bvh/shading/AttributeTable.h>

#include <scene_rdl2/scene/rdl2/RootShader.h>


namespace moonray {
namespace shading {

class RootShader : public scene_rdl2::rdl2::SceneObject::Extension
{
public:
    explicit RootShader(const scene_rdl2::rdl2::SceneObject & owner) :
        mOwner(*owner.asA<scene_rdl2::rdl2::RootShader>()) {}

    const scene_rdl2::rdl2::RootShader &getOwner() const { return mOwner; }

    //
    // Attribute APIs.
    //

    AttributeTable * setAttributeTable(
        std::unique_ptr<AttributeTable> table)
    {
        mAttributeTable.swap(table);
        return mAttributeTable.get();
    }

    AttributeTable * getAttributeTable()
    { return mAttributeTable.get(); }

    const AttributeTable * getAttributeTable() const
    { return mAttributeTable.get(); }

    // This function is only valid for bundled configurations. It returns a globally
    // unique id for this material which is constant over any given frame and will be
    // no larger than the number of currently active materials in the scene.
    // Zero is reserved for a null material.
    uint32_t getMaterialId() const { MNRY_ASSERT(mMaterialId > 0); return mMaterialId; }

private:
    const scene_rdl2::rdl2::RootShader& mOwner;
    std::unique_ptr<AttributeTable> mAttributeTable;

protected:
    // TODO: We don't support deleting materials currently. We'll need to add
    // some extra logic to maintain a valid bundled id when that gets implemented.
    uint32_t mMaterialId;
};

} // namespace shading
} // namespace moonray

