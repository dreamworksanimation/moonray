// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TextureColorManagement.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace moonray {
namespace shading {
namespace texture_color_management {
namespace {

constexpr const char* sRenderingColorSpaceEnv = "MOONRAY_OCIO_RENDERING_COLOR_SPACE";

std::string
trim(std::string value)
{
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

std::string
normalized(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       if (c == '-' || c == ' ' || c == '.') return '_';
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool
isHardBypassToken(const std::string& value)
{
    const std::string key = normalized(trim(value));
    // These tokens describe data, not a color space to be resolved through
    // OCIO.  In particular, do not depend on a config's optional `Raw` color
    // space being marked as data: scalar maps (roughness, metallic, masks,
    // displacement) must preserve their sampled values exactly.
    return key == "raw" || key == "data" || key == "none";
}

std::string
ocioPath()
{
    const char* ocio = std::getenv("OCIO");
    return ocio ? std::string(ocio) : std::string();
}

std::string
colorSpaceName(const OCIO::ConstColorSpaceRcPtr& colorSpace)
{
    if (!colorSpace) {
        return {};
    }
    const char* name = colorSpace->getName();
    return name ? std::string(name) : std::string();
}

std::string
resolveColorSpace(const OCIO::ConstConfigRcPtr& config,
                  const std::string& token)
{
    if (!config || token.empty()) {
        return {};
    }

    try {
        return colorSpaceName(config->getColorSpace(token.c_str()));
    } catch (const OCIO::Exception&) {
    }
    return {};
}

bool
isRoleDataColorSpace(const OCIO::ConstConfigRcPtr& config,
                     const std::string& colorSpaceName)
{
    if (!config || colorSpaceName.empty()) {
        return false;
    }

    const std::string dataRole = resolveColorSpace(config, OCIO::ROLE_DATA);
    return !dataRole.empty() && dataRole == colorSpaceName;
}

bool
isDataColorSpace(const OCIO::ConstConfigRcPtr& config,
                 const std::string& colorSpaceName)
{
    if (!config || colorSpaceName.empty()) {
        return false;
    }

    if (isRoleDataColorSpace(config, colorSpaceName)) {
        return true;
    }

    try {
        OCIO::ConstColorSpaceRcPtr colorSpace = config->getColorSpace(colorSpaceName.c_str());
        if (colorSpace) {
            if (colorSpace->isData()) {
                return true;
            }
        }
    } catch (const OCIO::Exception&) {
    }

    return false;
}

std::string
roleColorSpace(const OCIO::ConstConfigRcPtr& config, const char* role)
{
    return resolveColorSpace(config, role ? role : "");
}

std::string
renderingColorSpace(const OCIO::ConstConfigRcPtr& config,
                    std::string* method)
{
    const char* authoredTarget = std::getenv(sRenderingColorSpaceEnv);
    if (authoredTarget && authoredTarget[0]) {
        const std::string resolved = resolveColorSpace(config, authoredTarget);
        if (!resolved.empty() && !isDataColorSpace(config, resolved)) {
            if (method) {
                *method = std::string("render-setting:") + sRenderingColorSpaceEnv;
            }
            return resolved;
        }
    }

    const std::pair<const char*, const char*> roles[] = {
        {OCIO::ROLE_RENDERING, "role:rendering"},
        {OCIO::ROLE_SCENE_LINEAR, "role:scene_linear"},
        {"default_float", "role:default_float"},
        {"reference", "role:reference"},
        {OCIO::ROLE_DEFAULT, "role:default"}
    };

    for (const auto& entry : roles) {
        std::string name = roleColorSpace(config, entry.first);
        if (!name.empty() && !isDataColorSpace(config, name)) {
            if (method) {
                *method = entry.second;
            }
            return name;
        }
    }

    if (method) {
        method->clear();
    }
    return {};
}

std::string
sourceColorSpaceForTexture(const OCIO::ConstConfigRcPtr& config,
                           const std::string& filename,
                           const std::string& authoredSource,
                           std::string* method,
                           std::string* reason)
{
    const std::string source = trim(authoredSource);
    const std::string key = normalized(source.empty() ? std::string("auto") : source);

    if (isHardBypassToken(key)) {
        if (method) {
            *method = "bypass";
        }
        if (reason) {
            *reason = "authored none token";
        }
        return {};
    }

    if (!config) {
        if (method) {
            *method = "disabled";
        }
        if (reason) {
            *reason = "no active OCIO config";
        }
        return {};
    }

    if (key == "auto") {
        try {
            const char* fileRuleSpace = config->getColorSpaceFromFilepath(filename.c_str());
            if (!fileRuleSpace || isHardBypassToken(fileRuleSpace)) {
                if (method) {
                    *method = "file-rule:bypass";
                }
                if (reason) {
                    *reason = fileRuleSpace ? fileRuleSpace : "no file-rule color space";
                }
                return {};
            }
            std::string resolved = resolveColorSpace(config, fileRuleSpace);
            if (!resolved.empty()) {
                if (isDataColorSpace(config, resolved)) {
                    if (method) {
                        *method = "file-rule:data";
                    }
                    if (reason) {
                        *reason = resolved;
                    }
                    return {};
                }
                if (method) {
                    *method = "file-rule";
                }
                return resolved;
            }
        } catch (const OCIO::Exception& e) {
            if (method) {
                *method = "file-rule:error";
            }
            if (reason) {
                *reason = e.what();
            }
            return {};
        }

        if (method) {
            *method = "file-rule:unresolved";
        }
        if (reason) {
            *reason = "file rule returned an unresolved color space";
        }
        return {};
    }

    std::string resolved = resolveColorSpace(config, source);
    if (!resolved.empty()) {
        if (isDataColorSpace(config, resolved)) {
            if (method) {
                *method = "explicit:data";
            }
            if (reason) {
                *reason = resolved;
            }
            return {};
        }
        if (method) {
            *method = "explicit";
        }
        return resolved;
    }

    if (key == "raw" || key == "data") {
        const std::string dataRole = resolveColorSpace(config, OCIO::ROLE_DATA);
        if (!dataRole.empty() && isDataColorSpace(config, dataRole)) {
            if (method) {
                *method = "explicit:data-role";
            }
            if (reason) {
                *reason = dataRole;
            }
            return {};
        }
        if (method) {
            *method = "explicit:unresolved";
        }
        if (reason) {
            *reason = "authored data token is not a name, role, or alias in the active OCIO config";
        }
        return {};
    }

    if (method) {
        *method = "explicit:unresolved";
    }
    if (reason) {
        *reason = "authored source color space is not a name, role, or alias in the active config";
    }
    return {};
}

std::string
baseDiagnostic(const std::string& filename,
               const std::string& authoredSource,
               const std::string& method)
{
    std::ostringstream out;
    out << "filename=\"" << filename << "\""
        << " authored=\"" << authoredSource << "\""
        << " method=" << method
        << " ocio=\"" << ocioPath() << "\""
        << " version=" << OCIO::GetVersion();
    return out.str();
}

} // namespace

ProcessorResult
createTextureProcessor(const std::string& filename,
                       const std::string& sourceColorSpace)
{
    ProcessorResult result;

    OCIO::ConstConfigRcPtr config;
    try {
        config = OCIO::GetCurrentConfig();
    } catch (const OCIO::Exception& e) {
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, "disabled") +
            " reason=\"OCIO config load failed: " + e.what() + "\"";
        return result;
    }

    std::string sourceMethod;
    std::string sourceReason;
    const std::string source = sourceColorSpaceForTexture(config,
                                                          filename,
                                                          sourceColorSpace,
                                                          &sourceMethod,
                                                          &sourceReason);
    if (source.empty()) {
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, sourceMethod) +
            " reason=\"" + sourceReason + "\"";
        return result;
    }

    std::string targetMethod;
    const std::string target = renderingColorSpace(config, &targetMethod);
    if (target.empty()) {
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, sourceMethod) +
            " source=\"" + source + "\" reason=\"no render/working color space resolved\"";
        return result;
    }

    if (source == target) {
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, "identity") +
            " source=\"" + source + "\" target=\"" + target + "\" targetMethod=" + targetMethod;
        return result;
    }

    try {
        OCIO::ConstProcessorRcPtr processor =
            config->getProcessor(source.c_str(), target.c_str());
        result.mProcessor = processor ? processor->getDefaultCPUProcessor() : OCIO::ConstCPUProcessorRcPtr();
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, sourceMethod) +
            " source=\"" + source + "\" target=\"" + target + "\" targetMethod=" + targetMethod;
        return result;
    } catch (const OCIO::Exception& e) {
        result.mDiagnostic = baseDiagnostic(filename, sourceColorSpace, "failed") +
            " source=\"" + source + "\" target=\"" + target + "\" reason=\"" + e.what() + "\"";
    }

    return result;
}

void
applyProcessor(intptr_t processorPtr, float* rgba)
{
    if (!processorPtr || !rgba) {
        return;
    }
    const OCIO::CPUProcessor* processor =
        reinterpret_cast<const OCIO::CPUProcessor*>(processorPtr);
    try {
        processor->applyRGB(rgba);
    } catch (const OCIO::Exception&) {
    }
}

} // namespace texture_color_management
} // namespace shading
} // namespace moonray
