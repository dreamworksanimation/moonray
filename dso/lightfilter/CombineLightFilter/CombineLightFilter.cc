// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include "attributes.cc"

using namespace scene_rdl2;

RDL2_DSO_CLASS_BEGIN(CombineLightFilter, rdl2::LightFilter)

public:
    RDL2_DSO_DEFAULT_CTOR(CombineLightFilter)

    bool isOn() const override
    {
        const rdl2::SceneObjectVector& rdlLightFilters =
            get<rdl2::SceneObjectVector>(attrLightFilters);
        return get(sOnKey) && rdlLightFilters.size() > 0;
    }

    void getReferencedLightFilters(std::unordered_set<const rdl2::LightFilter *>& filters) const override
    {
         const rdl2::SceneObjectVector& rdlLightFilters =
             get<rdl2::SceneObjectVector>(attrLightFilters);

         for (rdl2::SceneObject* sceneObject : rdlLightFilters) {
             const rdl2::LightFilter *rdlLightFilter = sceneObject->asA<rdl2::LightFilter>();
             if (rdlLightFilter == nullptr) {
                 // not a light filter, ignore
                 continue;
             }
             if (rdlLightFilter->isOn()) {
                 filters.insert(rdlLightFilter);
                 rdlLightFilter->getReferencedLightFilters(filters);
             }
         }
    }

RDL2_DSO_CLASS_END(CombineLightFilter)

