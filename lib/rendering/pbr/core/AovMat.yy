/* Copyright 2023-2024 DreamWorks Animation LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file AovMat.yy
 */

%{
#include <moonray/rendering/pbr/core/Aov.h>

#include <string>

// this parser is not thread-safe!
#if defined (__ICC)
__pragma(warning(disable:1711))
#endif

// not sure about these, but icc complains
// "zero used for undefined preprocessing identifier":
//   # if YYENABLE_NLS
//   # if YYLTYPE_IS_TRIVIAL
#if defined (__ICC)
__pragma(warning(disable:193))
#endif


#undef yylex
#define yylex aovMat_yylex
extern int aovMat_yylex();

void yyerror(const char *err);

static moonray::pbr::ParsedMaterialExpression *mataov = nullptr;

static std::string
trim(const std::string &in)
{
    // strip leading and trailing ticks and white space
    const std::string ws = "' \t";
    const std::size_t beg = in.find_first_not_of(ws);
    if (beg == std::string::npos) {
        return "";
    }
    const std::size_t end = in.find_last_not_of(ws);
    return in.substr(beg, end - beg + 1);
}

static void
setSelectorBits(const std::string &in)
{
    for (auto c: in) {
        switch (c) {
        case 'R':
            mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::REFLECTION;
            break;
        case 'T':
            mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::TRANSMISSION;
            break;
        case 'D':
            mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::DIFFUSE;
            break;
        case 'G':
            mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::GLOSSY;
            break;
        case 'M':
            mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::MIRROR;
            break;
        default:
            MNRY_ASSERT(0 && "unexpected char in lobe selector\n");
        }
    }
}

static void
setPrimitiveAttributeFloat(const std::string &in)
{
    const std::string TYPE_STR = "float:";
    mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_FLOAT;
    mataov->mPrimitiveAttribute = in.substr(TYPE_STR.length());
}

static void
setPrimitiveAttributeVec2(const std::string &in)
{
    const std::string TYPE_STR = "vec2:";
    mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_VEC2;
    mataov->mPrimitiveAttribute = in.substr(TYPE_STR.length());
}

static void
setPrimitiveAttributeVec3(const std::string &in)
{
    const std::string TYPE_STR = "vec3:";
    mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_VEC3;
    mataov->mPrimitiveAttribute = in.substr(TYPE_STR.length());
}

static void
setPrimitiveAttributeRgb(const std::string &in)
{
    const std::string TYPE_STR = "rgb:";
    mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_RGB;
    mataov->mPrimitiveAttribute = in.substr(TYPE_STR.length());
}

%}

%union
{
    const char *s;
}

/* terminals */
/* labels */
%token <s> LABEL
%token     DOT

/* bsdf lobe (and sub-surface) selectors */
%token <s> LOBE_SELECTOR
%token     SUBSURFACE_SELECTOR
%token     FRESNEL_SELECTOR

/* values */
%token     ALBEDO
%token     COLOR
%token     EMISSION
%token     FACTOR
%token     NORMAL
%token     RADIUS
%token     ROUGHNESS
%token     MATTE
%token     PBR_VALIDITY
%token     STATE_VARIABLE_P
%token     STATE_VARIABLE_N
%token     STATE_VARIABLE_NG
%token     STATE_VARIABLE_ST
%token     STATE_VARIABLE_DPDS
%token     STATE_VARIABLE_DPDT
%token     STATE_VARIABLE_DSDX
%token     STATE_VARIABLE_DSDY
%token     STATE_VARIABLE_DTDX
%token     STATE_VARIABLE_DTDY
%token     STATE_VARIABLE_WP
%token     STATE_VARIABLE_DEPTH
%token     STATE_VARIABLE_MOTION
%token <s> PRIMITIVE_ATTRIBUTE_TYPE_FLOAT
%token <s> PRIMITIVE_ATTRIBUTE_TYPE_VEC2
%token <s> PRIMITIVE_ATTRIBUTE_TYPE_VEC3
%token <s> PRIMITIVE_ATTRIBUTE_TYPE_RGB

/* non-terminals */
%type <s> labels

%start mataovexp

%%
mataovexp : bsdf_what
          | labels DOT bsdf_what
          ;

bsdf_what : what
          | bsdf_selector DOT what
          ;

bsdf_selector : lobe_selector
              | subsurface_selector
              | lobe_selector bsdf_selector
              | subsurface_selector bsdf_selector
              ;

lobe_selector : LOBE_SELECTOR { setSelectorBits($1); }

subsurface_selector : SUBSURFACE_SELECTOR { mataov->mSelector |= moonray::pbr::ParsedMaterialExpression::SUBSURFACE; }

what : property
     | FRESNEL_SELECTOR DOT property { mataov->mFresnelProperty = true; }

property : ALBEDO                { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_ALBEDO; }
         | COLOR                 { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_COLOR; }
         | EMISSION              { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_EMISSION; }
         | FACTOR                { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_FACTOR; }
         | NORMAL                { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_NORMAL; }
         | RADIUS                { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_RADIUS; }
         | ROUGHNESS             { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_ROUGHNESS; }
         | MATTE                 { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_MATTE; }
         | PBR_VALIDITY          { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_PBR_VALIDITY; }
         | STATE_VARIABLE_P      { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_P; }
         | STATE_VARIABLE_N      { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_N; }
         | STATE_VARIABLE_NG     { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_NG; }
         | STATE_VARIABLE_ST     { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_ST; }
         | STATE_VARIABLE_DPDS   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DPDS; }
         | STATE_VARIABLE_DPDT   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DPDT; }
         | STATE_VARIABLE_DSDX   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DSDX; }
         | STATE_VARIABLE_DSDY   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DSDY; }
         | STATE_VARIABLE_DTDX   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DTDX; }
         | STATE_VARIABLE_DTDY   { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DTDY; }
         | STATE_VARIABLE_WP     { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_WP; }
         | STATE_VARIABLE_DEPTH  { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DEPTH; }
         | STATE_VARIABLE_MOTION { mataov->mProperty = moonray::pbr::ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_MOTION; }
         | PRIMITIVE_ATTRIBUTE_TYPE_FLOAT { setPrimitiveAttributeFloat($1); }
         | PRIMITIVE_ATTRIBUTE_TYPE_VEC2  { setPrimitiveAttributeVec2($1); }
         | PRIMITIVE_ATTRIBUTE_TYPE_VEC3  { setPrimitiveAttributeVec3($1); }
         | PRIMITIVE_ATTRIBUTE_TYPE_RGB   { setPrimitiveAttributeRgb($1); }
         ;

labels : labels LABEL { mataov->mLabels.top().push_back(trim($2)); }
       | labels DOT   { mataov->mLabels.push(std::vector<std::string>()); }
       | LABEL        { mataov->mLabels.top().push_back(trim($1)); }
       ;
%%

void
yyerror(const char *err)
{
    mataov->mError = err;
}


namespace moonray {
namespace pbr {

/* used by lex to grab the next char */
int
aovMatParserGetC()
{
    if (mataov->mNextLex < mataov->mExpression.length()) {
        return mataov->mExpression[mataov->mNextLex++];
    } else {
        return 0;
    }
}

/* main entry point for the parser */
void
aovParseMaterialExpression(ParsedMaterialExpression *m)
{
    mataov = m;
    aovMat_yyparse();
}

} // namespace pbr
} // namespace moonray

