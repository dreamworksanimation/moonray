/* Copyright 2023-2024 DreamWorks Animation LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file AovMat.ll
 */

%option noyywrap

%{
/* bison forces the header to be .hh if the parser code extension is .cc */
#include "AovMatParser.hh"

// this lexer is not thread-safe!
#if defined (__ICC)
__pragma(warning(disable:1711))
#endif

// note sure about these, but icc complains
#if defined(__ICC)
__pragma(warning(disable:111))
__pragma(warning(disable:177))
#endif

namespace moonray {
namespace pbr {

extern int aovMatParserGetC();

} // namespace pbr
} // namespace moonray

#define YY_INPUT(buf, result, max_size)          \
{                                                \
     if ((max_size) == 0) {                      \
         (result) = YY_NULL;                     \
     } else {                                    \
         int c = moonray::pbr::aovMatParserGetC(); \
         if (c == 0) {                           \
             (result) = YY_NULL;                 \
         } else {                                \
             (buf)[0] = c;                       \
             (result) = 1;                       \
         }                                       \
     }                                           \
}

#define YY_NO_UNPUT

%}


%%
'[a-zA-Z0-9 _-]*'   { aovMat_yylval.s = yytext; return LABEL; }
\.                  { aovMat_yylval.s = yytext; return DOT; }
[RTDGM]+            { aovMat_yylval.s = yytext; return LOBE_SELECTOR; }
SS                  { aovMat_yylval.s = yytext; return SUBSURFACE_SELECTOR; }
fresnel             { aovMat_yylval.s = yytext; return FRESNEL_SELECTOR; }
albedo              { aovMat_yylval.s = yytext; return ALBEDO; }
color               { aovMat_yylval.s = yytext; return COLOR; }
emission            { aovMat_yylval.s = yytext; return EMISSION; }
factor              { aovMat_yylval.s = yytext; return FACTOR; }
normal              { aovMat_yylval.s = yytext; return NORMAL; }
radius              { aovMat_yylval.s = yytext; return RADIUS; }
roughness           { aovMat_yylval.s = yytext; return ROUGHNESS; }
matte               { aovMat_yylval.s = yytext; return MATTE; }
pbr_validity        { aovMat_yylval.s = yytext; return PBR_VALIDITY; }
P             	    { aovMat_yylval.s = yytext; return STATE_VARIABLE_P; }
N             	    { aovMat_yylval.s = yytext; return STATE_VARIABLE_N; }
Ng            	    { aovMat_yylval.s = yytext; return STATE_VARIABLE_NG; }
St            	    { aovMat_yylval.s = yytext; return STATE_VARIABLE_ST; }
dPds                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DPDS; }
dPdt                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DPDT; }
dSdx                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DSDX; }
dSdy                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DSDY; }
dTdx                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DTDX; }
dTdy                { aovMat_yylval.s = yytext; return STATE_VARIABLE_DTDY; }
Wp                  { aovMat_yylval.s = yytext; return STATE_VARIABLE_WP; }
depth               { aovMat_yylval.s = yytext; return STATE_VARIABLE_DEPTH; }
motionvec           { aovMat_yylval.s = yytext; return STATE_VARIABLE_MOTION; }
float:[a-zA-Z0-9_]+ { aovMat_yylval.s = yytext; return PRIMITIVE_ATTRIBUTE_TYPE_FLOAT; }
vec2:[a-zA-Z0-9_]+  { aovMat_yylval.s = yytext; return PRIMITIVE_ATTRIBUTE_TYPE_VEC2; }
vec3:[a-zA-Z0-9_]+  { aovMat_yylval.s = yytext; return PRIMITIVE_ATTRIBUTE_TYPE_VEC3; }
rgb:[a-zA-Z0-9_]+   { aovMat_yylval.s = yytext; return PRIMITIVE_ATTRIBUTE_TYPE_RGB; }
.                 { /* no match - syntax error, must be last rule  */ aovMat_yylval.s = yytext; return -1; }
%%
