// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightStage.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

#include <atomic>
#include <cstring>

using namespace scene_rdl2::math;

namespace moonray {
namespace shading {


#define PBR_LIGHTSTAGE_DEBUG 0


//---------------------------------------------------------------------------

// Light-Stage X is made of the following lights. Light positions are given
// in ICT world space (real-world cm) and are converted to DWA world space in
// initLightStage()
static const int sLightCount = 346;
static float sLightPosition[sLightCount][3] = {
    { 0.00, 132.32, 0.00 },
    { 0.00, 130.68, -23.45 },
    { -22.30, 130.68, -7.25 },
    { -13.78, 130.68, 18.97 },
    { 13.78, 130.68, 18.97 },
    { 22.30, 130.68, -7.25 },
    { 0.00, 123.65, -47.11 },
    { -22.86, 126.15, -31.46 },
    { -44.81, 123.65, -14.56 },
    { -36.98, 126.15, 12.02 },
    { -27.69, 123.65, 38.11 },
    { 0.00, 126.15, 38.89 },
    { 27.69, 123.65, 38.11 },
    { 36.98, 126.15, 12.02 },
    { 44.81, 123.65, -14.56 },
    { 22.86, 126.15, -31.46 },
    { 0.00, 112.78, -69.70 },
    { -24.19, 116.68, -57.22 },
    { -46.95, 116.68, -40.69 },
    { -66.29, 112.78, -21.54 },
    { -61.90, 116.68, 5.32 },
    { -53.20, 116.68, 32.08 },
    { -40.97, 112.78, 56.39 },
    { -14.07, 116.68, 60.51 },
    { 14.07, 116.68, 60.51 },
    { 40.97, 112.78, 56.39 },
    { 53.20, 116.68, 32.08 },
    { 61.90, 116.68, 5.32 },
    { 66.29, 112.78, -21.54 },
    { 46.95, 116.68, -40.69 },
    { 24.19, 116.68, -57.22 },
    { 0.00, 97.43, -89.52 },
    { -24.19, 103.36, -78.77 },
    { -47.21, 105.15, -64.98 },
    { -67.44, 103.36, -47.34 },
    { -85.14, 97.43, -27.66 },
    { -82.39, 103.36, -1.34 },
    { -76.39, 105.15, 24.82 },
    { -65.87, 103.36, 49.51 },
    { -52.62, 97.43, 72.43 },
    { -26.73, 103.36, 77.94 },
    { 0.00, 105.15, 80.32 },
    { 26.73, 103.36, 77.94 },
    { 52.62, 97.43, 72.43 },
    { 65.87, 103.36, 49.51 },
    { 76.39, 105.15, 24.82 },
    { 82.39, 103.36, -1.34 },
    { 85.14, 97.43, -27.66 },
    { 67.44, 103.36, -47.34 },
    { 47.21, 105.15, -64.98 },
    { 24.19, 103.36, -78.77 },
    { 0.00, 79.42, -106.40 },
    { -22.86, 84.56, -98.77 },
    { -46.95, 88.57, -86.16 },
    { -67.44, 88.57, -71.28 },
    { -86.87, 84.56, -52.26 },
    { -101.19, 79.42, -32.88 },
    { -100.99, 84.56, -8.78 },
    { -96.45, 88.57, 18.02 },
    { -88.63, 88.57, 42.11 },
    { -76.54, 84.56, 66.47 },
    { -62.54, 79.42, 86.08 },
    { -39.56, 84.56, 93.34 },
    { -12.66, 88.57, 97.30 },
    { 12.66, 88.57, 97.30 },
    { 39.56, 84.56, 93.34 },
    { 62.54, 79.42, 86.08 },
    { 76.54, 84.56, 66.47 },
    { 88.63, 88.57, 42.11 },
    { 96.45, 88.57, 18.02 },
    { 100.99, 84.56, -8.78 },
    { 101.19, 79.42, -32.88 },
    { 86.87, 84.56, -52.26 },
    { 67.44, 88.57, -71.28 },
    { 46.95, 88.57, -86.16 },
    { 22.86, 84.56, -98.77 },
    { 0.00, 59.17, -118.35 },
    { -22.30, 64.92, -113.65 },
    { -44.81, 68.32, -104.08 },
    { -66.29, 69.70, -91.24 },
    { -85.14, 68.32, -74.78 },
    { -101.19, 64.92, -56.33 },
    { -112.56, 59.17, -36.57 },
    { -114.98, 64.92, -13.91 },
    { -112.83, 68.32, 10.45 },
    { -107.26, 69.70, 34.85 },
    { -97.43, 68.32, 57.87 },
    { -84.84, 64.92, 78.83 },
    { -69.56, 59.17, 95.75 },
    { -48.76, 64.92, 105.05 },
    { -24.93, 68.32, 110.54 },
    { 0.00, 69.70, 112.78 },
    { 24.93, 68.32, 110.54 },
    { 48.76, 64.92, 105.05 },
    { 69.56, 59.17, 95.75 },
    { 84.84, 64.92, 78.83 },
    { 97.43, 68.32, 57.87 },
    { 107.26, 69.70, 34.85 },
    { 112.83, 68.32, 10.45 },
    { 114.98, 64.92, -13.91 },
    { 112.56, 59.17, -36.57 },
    { 101.19, 64.92, -56.33 },
    { 85.14, 68.32, -74.78 },
    { 66.29, 69.70, -91.24 },
    { 44.81, 68.32, -104.08 },
    { 22.30, 64.92, -113.65 },
    { -13.78, 41.48, -125.37 },
    { -36.98, 45.67, -118.21 },
    { -61.90, 47.42, -106.74 },
    { -82.39, 47.42, -91.85 },
    { -100.99, 45.67, -71.70 },
    { -114.98, 41.48, -51.85 },
    { -123.49, 41.48, -25.63 },
    { -123.85, 45.67, -1.35 },
    { -120.64, 47.42, 25.88 },
    { -112.81, 47.42, 49.97 },
    { -99.40, 45.67, 73.89 },
    { -84.84, 41.48, 93.33 },
    { -62.54, 41.48, 109.53 },
    { -39.56, 45.67, 117.37 },
    { -12.66, 47.42, 122.73 },
    { 12.66, 47.42, 122.73 },
    { 39.56, 45.67, 117.37 },
    { 62.54, 41.48, 109.53 },
    { 84.84, 41.48, 93.33 },
    { 99.40, 45.67, 73.89 },
    { 112.81, 47.42, 49.97 },
    { 120.64, 47.42, 25.88 },
    { 123.85, 45.67, -1.35 },
    { 123.49, 41.48, -25.63 },
    { 114.98, 41.48, -51.85 },
    { 100.99, 45.67, -71.70 },
    { 82.39, 47.42, -91.85 },
    { 61.90, 47.42, -106.74 },
    { 36.98, 45.67, -118.21 },
    { 13.78, 41.48, -125.37 },
    { 0.00, 21.64, -130.23 },
    { -27.69, 21.21, -127.64 },
    { -53.20, 23.49, -118.70 },
    { -76.39, 24.82, -105.15 },
    { -96.45, 23.49, -87.28 },
    { -112.83, 21.21, -65.78 },
    { -123.85, 21.64, -40.24 },
    { -129.95, 21.21, -13.11 },
    { -129.33, 23.49, 13.92 },
    { -123.61, 24.82, 40.16 },
    { -112.81, 23.49, 64.76 },
    { -97.43, 21.21, 86.98 },
    { -76.54, 21.64, 105.35 },
    { -52.62, 21.21, 119.54 },
    { -26.73, 23.49, 127.30 },
    { 0.00, 24.82, 129.97 },
    { 26.73, 23.49, 127.30 },
    { 52.62, 21.21, 119.54 },
    { 76.54, 21.64, 105.35 },
    { 97.43, 21.21, 86.98 },
    { 112.81, 23.49, 64.76 },
    { 123.61, 24.82, 40.16 },
    { 129.33, 23.49, 13.92 },
    { 129.95, 21.21, -13.11 },
    { 123.85, 21.64, -40.24 },
    { 112.83, 21.21, -65.78 },
    { 96.45, 23.49, -87.28 },
    { 76.39, 24.82, -105.15 },
    { 53.20, 23.49, -118.70 },
    { 27.69, 21.21, -127.64 },
    { -14.07, -1.94, -131.42 },
    { -40.97, 0.00, -126.09 },
    { -65.87, 1.94, -114.59 },
    { -88.63, 1.94, -98.05 },
    { -107.26, 0.00, -77.93 },
    { -120.64, -1.94, -53.99 },
    { -129.33, -1.94, -27.23 },
    { -132.58, 0.00, 0.00 },
    { -129.33, 1.94, 27.23 },
    { -120.64, 1.94, 53.99 },
    { -107.26, 0.00, 77.93 },
    { -88.63, -1.94, 98.05 },
    { -65.87, -1.94, 114.59 },
    { -40.97, 0.00, 126.09 },
    { -14.07, 1.94, 131.42 },
    { 14.07, 1.94, 131.42 },
    { 40.97, 0.00, 126.09 },
    { 65.87, -1.94, 114.59 },
    { 88.63, -1.94, 98.05 },
    { 107.26, -0.00, 77.93 },
    { 120.64, 1.94, 53.99 },
    { 129.33, 1.94, 27.23 },
    { 132.58, 0.00, 0.00 },
    { 129.33, -1.94, -27.23 },
    { 120.64, -1.94, -53.99 },
    { 107.26, 0.00, -77.93 },
    { 88.63, 1.94, -98.05 },
    { 65.87, 1.94, -114.59 },
    { 40.97, 0.00, -126.09 },
    { 14.07, -1.94, -131.42 },
    { 0.00, -24.82, -129.97 },
    { -26.73, -23.49, -127.30 },
    { -52.62, -21.21, -119.54 },
    { -76.54, -21.64, -105.35 },
    { -97.43, -21.21, -86.98 },
    { -112.81, -23.49, -64.76 },
    { -123.61, -24.82, -40.16 },
    { -129.33, -23.49, -13.92 },
    { -129.95, -21.21, 13.11 },
    { -123.85, -21.64, 40.24 },
    { -112.83, -21.21, 65.78 },
    { -96.45, -23.49, 87.28 },
    { -76.39, -24.82, 105.15 },
    { -53.20, -23.49, 118.70 },
    { -27.69, -21.21, 127.64 },
    { 0.00, -21.64, 130.23 },
    { 27.69, -21.21, 127.64 },
    { 53.20, -23.49, 118.70 },
    { 76.39, -24.82, 105.15 },
    { 96.45, -23.49, 87.28 },
    { 112.83, -21.21, 65.78 },
    { 123.85, -21.64, 40.24 },
    { 129.95, -21.21, 13.11 },
    { 129.33, -23.49, -13.92 },
    { 123.61, -24.82, -40.16 },
    { 112.81, -23.49, -64.76 },
    { 97.43, -21.21, -86.98 },
    { 76.54, -21.64, -105.35 },
    { 52.62, -21.21, -119.54 },
    { 26.73, -23.49, -127.30 },
    { -12.66, -47.42, -122.73 },
    { -39.56, -45.67, -117.37 },
    { -62.54, -41.48, -109.53 },
    { -84.84, -41.48, -93.33 },
    { -99.40, -45.67, -73.89 },
    { -112.81, -47.42, -49.97 },
    { -120.64, -47.42, -25.88 },
    { -123.85, -45.67, 1.35 },
    { -123.49, -41.48, 25.63 },
    { -114.98, -41.48, 51.85 },
    { -100.99, -45.67, 71.70 },
    { -82.39, -47.42, 91.85 },
    { -61.90, -47.42, 106.74 },
    { -36.98, -45.67, 118.21 },
    { -13.78, -41.48, 125.37 },
    { 13.78, -41.48, 125.37 },
    { 36.98, -45.67, 118.21 },
    { 61.90, -47.42, 106.74 },
    { 82.39, -47.42, 91.85 },
    { 100.99, -45.67, 71.70 },
    { 114.98, -41.48, 51.85 },
    { 123.49, -41.48, 25.63 },
    { 123.85, -45.67, 1.35 },
    { 120.64, -47.42, -25.88 },
    { 112.81, -47.42, -49.97 },
    { 99.40, -45.67, -73.89 },
    { 84.84, -41.48, -93.33 },
    { 62.54, -41.48, -109.53 },
    { 39.56, -45.67, -117.37 },
    { 12.66, -47.42, -122.73 },
    { 0.00, -69.70, -112.78 },
    { -24.93, -68.32, -110.54 },
    { -48.76, -64.92, -105.05 },
    { -69.56, -59.17, -95.75 },
    { -84.84, -64.92, -78.83 },
    { -97.43, -68.32, -57.87 },
    { -107.26, -69.70, -34.85 },
    { -112.83, -68.32, -10.45 },
    { -114.98, -64.92, 13.91 },
    { -112.56, -59.17, 36.57 },
    { -101.19, -64.92, 56.33 },
    { -85.14, -68.32, 74.78 },
    { -66.29, -69.70, 91.24 },
    { -44.81, -68.32, 104.08 },
    { -22.30, -64.92, 113.65 },
    { 0.00, -59.17, 118.35 },
    { 22.30, -64.92, 113.65 },
    { 44.81, -68.32, 104.08 },
    { 66.29, -69.70, 91.24 },
    { 85.14, -68.32, 74.78 },
    { 101.19, -64.92, 56.33 },
    { 112.56, -59.17, 36.57 },
    { 114.98, -64.92, 13.91 },
    { 112.83, -68.32, -10.45 },
    { 107.26, -69.70, -34.85 },
    { 97.43, -68.32, -57.87 },
    { 84.84, -64.92, -78.83 },
    { 69.56, -59.17, -95.75 },
    { 48.76, -64.92, -105.05 },
    { 24.93, -68.32, -110.54 },
    { -12.66, -88.57, -97.30 },
    { -39.56, -84.56, -93.34 },
    { -62.54, -79.42, -86.08 },
    { -76.54, -84.56, -66.47 },
    { -88.63, -88.57, -42.11 },
    { -96.45, -88.57, -18.02 },
    { -100.99, -84.56, 8.78 },
    { -101.19, -79.42, 32.88 },
    { -86.87, -84.56, 52.26 },
    { -67.44, -88.57, 71.28 },
    { -46.95, -88.57, 86.16 },
    { -22.86, -84.56, 98.77 },
    { 0.00, -79.42, 106.40 },
    { 22.86, -84.56, 98.77 },
    { 46.95, -88.57, 86.16 },
    { 67.44, -88.57, 71.28 },
    { 86.87, -84.56, 52.26 },
    { 101.19, -79.42, 32.88 },
    { 100.99, -84.56, 8.78 },
    { 96.45, -88.57, -18.02 },
    { 88.63, -88.57, -42.11 },
    { 76.54, -84.56, -66.47 },
    { 62.54, -79.42, -86.08 },
    { 39.56, -84.56, -93.34 },
    { 12.66, -88.57, -97.30 },
    { 0.00, -105.15, -80.32 },
    { -26.73, -103.36, -77.94 },
    { -52.62, -97.43, -72.43 },
    { -65.87, -103.36, -49.51 },
    { -76.39, -105.15, -24.82 },
    { -82.39, -103.36, 1.34 },
    { -85.14, -97.43, 27.66 },
    { -67.44, -103.36, 47.34 },
    { -47.21, -105.15, 64.98 },
    { -24.19, -103.36, 78.77 },
    { 0.00, -24.82, -129.97},        // Original value was { 0.00, -97.43, 89.52 },
    { 24.19, -103.36, 78.77 },
    { 47.21, -105.15, 64.98 },
    { 67.44, -103.36, 47.34 },
    { 85.14, -97.43, 27.66 },
    { 82.39, -103.36, 1.34 },
    { 76.39, -105.15, -24.82 },
    { 65.87, -103.36, -49.51 },
    { 52.62, -97.43, -72.43 },
    { 26.73, -103.36, -77.94 },
    { -14.13, -116.72, -60.42 },
    { -40.97, -112.78, -56.39 },
    { -53.10, -116.72, -32.11 },
    { -61.83, -116.72, -5.24 },
    { -64.92, -112.78, 25.77 },
    { -46.94, -116.72, 40.58 },
    { -24.09, -116.72, 57.18 },
    { -4.45, -112.78, 69.70 },
    { 24.09, -116.72, 57.18 },
    { 46.94, -116.72, 40.58 },
    { 66.29, -112.78, 21.54 },
    { 61.83, -116.72, -5.24 },
    { 53.10, -116.72, -32.11 },
    { 40.97, -112.78, -56.39 },
    { 14.13, -116.72, -60.42 }
};


// 1-based light index for the spiraling OLAT pattern for separated cross
// and parallel polarized. The index is converted to 0-based index
// in initLightStage()
const int LightStage::sSeparatedLightCount;
static int sSeparatedLightIndex[LightStage::sSeparatedLightCount] =
   {138,
    166,
    77,
    225,
    199,
    140,
    79,
    105,
    164,
    286,
    258,
    32,
    223,
    284,
    260,
    201,
    142,
    81,
    34,
    7,
    50,
    103,
    162,
    330,
    312,
    314,
    221,
    282,
    262,
    203,
    83,
    36,
    9,
    1,
    15,
    48,
    101,
    160,
    328,
    316,
    144,
    219,
    280,
    264,
    205,
    85,
    38,
    11,
    13,
    46,
    99,
    158,
    326,
    318,
    266,
    146,
    217,
    278,
    207,
    87,
    40,
    42,
    44,
    97,
    156,
    276,
    324,
    148,
    215,
    320,
    268,
    89,
    91,
    93,
    95,
    154,
    274,
    322,
    209,
    150,
    213,
    270,
    152,
    272,
    211};



// 1-based light index for the spiraling OLAT pattern for non-polarized.
// The index is converted to 0-based index in initLightStage()
const int LightStage::sCombinedLightCount;
static int sCombinedLightIndex[LightStage::sCombinedLightCount] =
   {137,
    138,
    107,
    136,
    166,
    195,
    226,
    198,
    168,
    139,
    108,
    78,
    77,
    106,
    135,
    165,
    194,
    225,
    255,
    256,
    227,
    228,
    199,
    169,
    140,
    109,
    79,
    53,
    52,
    76,
    105,
    134,
    164,
    193,
    224,
    254,
    285,
    286,
    257,
    258,
    259,
    229,
    200,
    170,
    141,
    110,
    80,
    54,
    33,
    32,
    51,
    75,
    104,
    133,
    163,
    192,
    223,
    253,
    284,
    309,
    310,
    311,
    287,
    288,
    289,
    260,
    230,
    201,
    171,
    142,
    111,
    81,
    55,
    34,
    18,
    7,
    17,
    31,
    50,
    74,
    103,
    132,
    162,
    191,
    222,
    252,
    283,
    308,
    330,
    331,
    312,
    313,
    314,
    290,
    261,
    231,
    202,
    172,
    143,
    112,
    82,
    56,
    35,
    19,
    8,
    16,
    30,
    49,
    73,
    102,
    131,
    161,
    190,
    221,
    251,
    282,
    307,
    329,
    315,
    291,
    262,
    232,
    203,
    83,
    57,
    36,
    20,
    9,
    1,
    15,
    29,
    48,
    72,
    101,
    130,
    160,
    220,
    250,
    281,
    306,
    328,
    316,
    292,
    263,
    233,
    204,
    173,
    144,
    113,
    84,
    58,
    37,
    47,
    71,
    100,
    129,
    219,
    280,
    264,
    205,
    114,
    85,
    38,
    11,
    13,
    46,
    99,
    158,
    326,
    318,
    266,
    146,
    217,
    278,
    207,
    87,
    40,
    42,
    44,
    97,
    156,
    276,
    324,
    148,
    215,
    320,
    268,
    89,
    91,
    93,
    95,
    154,
    274,
    322,
    209,
    150,
    213,
    270,
    152,
    272,
    211};


// 1-based lightOrderIndex are converted to 0-based indices in initLightStage()
static const int sFaceCount = 177;
static int sFaces[sFaceCount][3] = {
    {1, 2, 3},
    {1, 2, 78},
    {1, 3, 7},
    {1, 5, 78},
    {1, 6, 5},
    {1, 6, 7},
    {10, 11, 25},
    {10, 11, 78},
    {10, 14, 24},
    {10, 14, 4},
    {10, 24, 25},
    {10, 4, 78},
    {11, 15, 26},
    {11, 15, 5},
    {11, 25, 26},
    {11, 5, 78},
    {12, 19, 20},
    {12, 19, 7},
    {12, 20, 21},
    {12, 21, 8},
    {12, 3, 7},
    {12, 3, 8},
    {13, 14, 28},
    {13, 14, 4},
    {13, 23, 27},
    {13, 23, 9},
    {13, 27, 28},
    {13, 4, 9},
    {14, 24, 28},
    {15, 16, 29},
    {15, 16, 5},
    {15, 26, 29},
    {16, 17, 30},
    {16, 17, 6},
    {16, 29, 30},
    {16, 5, 6},
    {17, 18, 31},
    {17, 18, 6},
    {17, 30, 41},
    {17, 31, 41},
    {18, 19, 32},
    {18, 19, 7},
    {18, 31, 32},
    {18, 6, 7},
    {19, 20, 33},
    {19, 32, 33},
    {2, 3, 8},
    {2, 4, 78},
    {2, 4, 9},
    {2, 8, 9},
    {20, 21, 35},
    {20, 33, 34},
    {20, 34, 35},
    {20, 35, 21},
    {21, 20, 35},
    {21, 22, 36},
    {21, 22, 8},
    {21, 35, 20},
    {21, 35, 36},
    {22, 23, 37},
    {22, 23, 9},
    {22, 36, 37},
    {22, 8, 9},
    {23, 27, 38},
    {23, 37, 38},
    {24, 25, 39},
    {24, 28, 39},
    {25, 26, 40},
    {25, 39, 40},
    {26, 29, 40},
    {27, 28, 43},
    {27, 38, 42},
    {27, 42, 43},
    {28, 39, 43},
    {29, 30, 44},
    {29, 40, 44},
    {30, 41, 45},
    {30, 44, 45},
    {31, 32, 46},
    {31, 41, 46},
    {32, 33, 47},
    {32, 46, 47},
    {33, 34, 48},
    {33, 47, 48},
    {34, 35, 49},
    {34, 48, 49},
    {35, 20, 21},
    {35, 21, 20},
    {35, 36, 50},
    {35, 49, 50},
    {36, 37, 51},
    {36, 50, 51},
    {37, 38, 51},
    {38, 42, 52},
    {38, 51, 52},
    {39, 25, 67},
    {39, 43, 53},
    {39, 53, 67},
    {40, 25, 70},
    {40, 44, 54},
    {40, 54, 70},
    {41, 45, 56},
    {41, 46, 56},
    {42, 43, 58},
    {42, 52, 57},
    {42, 57, 58},
    {43, 53, 58},
    {44, 45, 55},
    {44, 54, 55},
    {45, 55, 59},
    {45, 56, 59},
    {46, 47, 60},
    {46, 56, 60},
    {47, 48, 61},
    {47, 60, 61},
    {48, 49, 62},
    {48, 61, 62},
    {49, 50, 63},
    {49, 62, 63},
    {5, 1, 6},
    {5, 6, 1},
    {50, 51, 64},
    {50, 63, 64},
    {51, 52, 64},
    {52, 57, 65},
    {52, 64, 65},
    {53, 58, 66},
    {53, 66, 67},
    {54, 55, 71},
    {54, 70, 71},
    {55, 59, 71},
    {56, 59, 68},
    {56, 60, 68},
    {57, 58, 66},
    {57, 65, 69},
    {57, 66, 69},
    {59, 68, 79},
    {59, 71, 79},
    {6, 1, 5},
    {6, 5, 1},
    {60, 61, 72},
    {60, 68, 72},
    {61, 62, 73},
    {61, 72, 73},
    {62, 63, 74},
    {62, 73, 74},
    {63, 64, 75},
    {63, 74, 75},
    {64, 65, 75},
    {65, 69, 76},
    {65, 75, 76},
    {66, 67, 77},
    {66, 69, 77},
    {67, 39, 70},
    {67, 70, 84},
    {67, 77, 84},
    {68, 72, 80},
    {68, 79, 80},
    {69, 76, 81},
    {69, 77, 81},
    {70, 40, 67},
    {70, 71, 82},
    {70, 82, 84},
    {71, 79, 82},
    {72, 73, 80},
    {73, 74, 83},
    {73, 80, 83},
    {74, 75, 76},
    {74, 76, 83},
    {76, 81, 83},
    {77, 81, 84},
    {79, 80, 85},
    {79, 82, 85},
    {80, 83, 85},
    {81, 83, 85},
    {81, 84, 85},
    {82, 84, 85}
};


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

LightStage::LightStage()
{
    MOONRAY_START_THREADSAFE_STATIC_WRITE;

    static std::atomic_flag sInitialized = ATOMIC_FLAG_INIT; // Deprecated in C++20. Just use the default constructor.
    if (sInitialized.test_and_set()) {
        return;
    }

    // Convert ids to 0-based indices
    for (int i=0; i < sSeparatedLightCount; i++) {
        sSeparatedLightIndex[i]--;
    }
    for (int i=0; i < sCombinedLightCount; i++) {
        sCombinedLightIndex[i]--;
    }

    // Convert light position from ICT world space to DWA world space by
    // flipping sign on x and z.
    for (int i=0; i < sLightCount; i++) {
        sLightPosition[i][0] = -sLightPosition[i][0];
        sLightPosition[i][2] = -sLightPosition[i][2];
    }

    // Convert face indices from 1-based to 0-based
    for (int i=0; i < sFaceCount; i++) {
        sFaces[i][0]--;
        sFaces[i][1]--;
        sFaces[i][2]--;
    }

    //printLightStageLightsInOrder()

    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE;
}


void
LightStage::printLightStageLightsInOrder()
{
    for (int i = 0; i < sSeparatedLightCount; i++) {
        int lightIndex = getSeparatedLightIndex(i);
        Vec3f lightPosition = getLightPosition(lightIndex);
        std::cout << "Light " << i << ": " << lightPosition << std::endl;
    }
}


int
LightStage::getSeparatedLightIndex(int lightOrderIndex) const
{
    MNRY_ASSERT(lightOrderIndex >= 0  &&  lightOrderIndex < sSeparatedLightCount);
    return sSeparatedLightIndex[lightOrderIndex];
}


int
LightStage::getCombinedLightIndex(int lightOrderIndex) const
{
    MNRY_ASSERT(lightOrderIndex >= 0  &&  lightOrderIndex < sCombinedLightCount);
    return sCombinedLightIndex[lightOrderIndex];
}


Vec3f
LightStage::getLightPosition(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < sLightCount);
    return Vec3f(sLightPosition[index]);
}


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

LightStageCylinderBsdfSlice::LightStageCylinderBsdfSlice(bool top, int sizeThetaO,
        int lightCount, float cylRadius, float cylZCenter, float cylAlpha) :
    mTop(top),
    mSizeThetaO(sizeThetaO),
    mLightCount(lightCount),
    mCylRadius(cylRadius),
    mCylZCenter(cylZCenter),
    mCylAlpha(cylAlpha),
    mData(nullptr)
{
    init();
}


void
LightStageCylinderBsdfSlice::init()
{
    int floatCount = getFloatCount();
    mData = new float[floatCount];
    for (int i = 0; i < floatCount; i++) {
        mData[i] = 0.0f;
    }

    // Cylinder <--> world transform
    mCyl2LS = Mat4f::rotate(Vec4f(0.0f, 0.0f, 1.0f, 0.0f), mCylAlpha)
           * Mat4f::translate(Vec4f(0.0f, 0.0f, mCylZCenter, 0.0f));
    mLS2Cyl = mCyl2LS.inverse();
}


LightStageCylinderBsdfSlice::~LightStageCylinderBsdfSlice()
{
    delete [] mData;
}


//---------------------------------------------------------------------------

// cppcheck-suppress uninitMemberVar // note these is an embree file so we are ignoring these
LightStageCylinderBsdfSlice::LightStageCylinderBsdfSlice(const std::string &filename)
{
    FILE *file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    int top = 0;
    if (fread(&top, sizeof(int), 1, file) != 1  ||
        fread(&mSizeThetaO, sizeof(int), 1, file) != 1  ||
        fread(&mLightCount, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }
    mTop = top;

    if (fread(&mCylRadius, sizeof(float), 1, file) != 1  ||
        fread(&mCylZCenter, sizeof(float), 1, file) != 1  ||
        fread(&mCylAlpha, sizeof(float), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read cylinder properties in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    init();

    int floatCount = getFloatCount();
    if (fread(mData, sizeof(float), floatCount, file) != floatCount) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }
    fclose(file);
}


void
LightStageCylinderBsdfSlice::saveAs(const std::string &filename) const
{
    FILE *file = fopen(filename.c_str(), "wb");
    if (file == nullptr) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    int top = mTop;
    if (fwrite(&top, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizeThetaO, sizeof(int), 1, file) != 1  ||
        fwrite(&mLightCount, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    if (fwrite(&mCylRadius, sizeof(float), 1, file) != 1  ||
        fwrite(&mCylZCenter, sizeof(float), 1, file) != 1  ||
        fwrite(&mCylAlpha, sizeof(float), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write cylinder properties in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    int floatCount = getFloatCount();
    if (fwrite(mData, sizeof(float), floatCount, file) != floatCount) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write table in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }
    fclose(file);
}


//---------------------------------------------------------------------------

// rotate vector along one axis
static Vec3f rotate(const Vec3f &vector, const Vec3f &axis, float angle)
{
    Mat4f mat = Mat4f::rotate(Vec4f(axis.x, axis.y, axis.z, 0.0f), angle);
    Vec3f result = transformVector(mat, vector);

    return result;
}


void
LightStageCylinderBsdfSlice::computeCylLightStagePNT(float thetaO, Vec3f &P,
        Vec3f &N, Vec3f &T) const
{
    N = Vec3f(sin(thetaO), 0.0f, cos(thetaO));
    P = N * getCylRadius();
    T = Vec3f(cos(thetaO), 0.0f, -sin(thetaO));
    P = transformPoint(getCyl2W(), P);
    T = transformVector(getCyl2W(), T);
    N = transformNormal(getW2Cyl(), N);
    N = normalize(N);
}


#if 0

static int
getClosestSeparatedLightOrderIndex(const Vec3f &P, const Vec3f &N, const Vec3f &wi)
{
    int closestIndex = -1;
    float closestDot = 0.0f;
    for (int i = 0; i < sSeparatedLightCount; i++) {
        Vec3f L = getLightStageLightPosition(getLightStageSeparatedLightIndex(i));
        Vec3f testWi = normalize(L - P);
        if (dot(N, testWi) > sEpsilon) {
            float testDot = dot(wi, testWi);
            if (testDot > closestDot) {
                closestDot = testDot;
                closestIndex = i;
            }
        }
    }

    return closestIndex;
}


static void
getThreeClosestSeparatedLightOrderIndex(const Vec3f &P, const Vec3f &N,
        const Vec3f &wi, Vec3i &index)
{
    index = Vec3i(-1);
    Vec3f closestDot(0.0f);

    // Search for three closest light directions
    for (int i = 0; i < sSeparatedLightCount; i++) {
        Vec3f Plight = getLightStageLightPosition(getLightStageSeparatedLightIndex(i));
        Vec3f testWi = normalize(Plight - P);
        float testDot = dot(wi, testWi);
        for (int j = 0; j < 3; j++) {
            if (testDot > closestDot[j]) {
                for (int k = 2; k > j; k--) {
                    closestDot[k] = closestDot[k-1];
                    index[k] = index[k-1];
                }
                closestDot[j] = testDot;
                index[j] = i;
                break;
            }
        }
    }
    MNRY_ASSERT(index[0] >= 0  &&  index[1] >= 0  &&  index[2] >= 0);

    std::cout << "Triangle: " << index[0] + 1 << " " << index[1] + 1 << " " << index[2] + 1 << std::endl;
}

#endif


static bool
intersect(const Vec3f &P, const Vec3f &wi, const Vec3f Plight[3],
        Vec3f &weights)
{
    Vec3f mV0(Plight[0]);
    Vec3f mE1 = Plight[1] - mV0;
    Vec3f mE2 = Plight[2] - mV0;
    Vec3f mE1CrossE2 = cross(mE1, mE2);

    float det = -dot(wi, mE1CrossE2);
    if (scene_rdl2::math::abs(det) < sEpsilon) {
        return false;
    }
    float invDet = 1.0f / det;

    Vec3f tvec = P - mV0;
    float distance = dot(mE1CrossE2, tvec) * invDet;
    if (distance < 0.0f) {
        return false;
    }

    Vec3f txr = cross(tvec, wi);

    float alpha =  dot(txr, mE2) * invDet;   // alpha
    if (alpha < 0.0f  ||  alpha > 1.0f) {
        return false;
    }

    float beta = -dot(txr, mE1) * invDet;   // beta
    if (beta < 0.0f  ||  beta > 1.0f) {
        return false;
    }

    if (alpha + beta > 1.0f) {
        return false;
    }

    weights[0] = 1.0f - (alpha + beta);
    weights[1] = alpha;
    weights[2] = beta;

    return true;
}


static bool
getFaceLightOrderIndex(const Vec3f &P, const Vec3f &wi,
        Vec3i &lightOrderIndex, Vec3f &weights)
{
    static const LightStage &ls = LightStage::singleton();

    // TODO: Use spatial index to speedup the lookup

    // Search the face that wi points to
    for (int f=0; f < sFaceCount; f++) {
        lightOrderIndex = Vec3i(sFaces[f][0],
                                sFaces[f][1],
                                sFaces[f][2]);
        Vec3f Plight[3];
        Plight[0] = ls.getLightPosition(ls.getSeparatedLightIndex(lightOrderIndex[0]));
        Plight[1] = ls.getLightPosition(ls.getSeparatedLightIndex(lightOrderIndex[1]));
        Plight[2] = ls.getLightPosition(ls.getSeparatedLightIndex(lightOrderIndex[2]));
        if (intersect(P, wi, Plight, weights)) {
            return true;
        }
    }

    return false;
}


Color
LightStageCylinderBsdfSlice::getBsdf(const Vec3f &localWo, const Vec3f &localWi,
        const Vec2f &smoothThetaWoRange, bool useLerp) const
{
    const float thetaWo = acos(max(localWo.z, 0.0f));
    if (thetaWo >= smoothThetaWoRange[1]) {
        return getBsdfSample(localWo, localWi, useLerp);
    }

    int count = 32;
    int i = 0;
    Color c(0.0f);
    for (float phi = 0.0f; phi < 360.0f; i++, phi += 360.0f / count) {
        Vec3f lwo = rotate(localWo, Vec3f(0.0f, 0.0f, 1.0f), phi / 180.0f * sPi);
        c += getBsdfSample(lwo, localWi, useLerp);
    }
    c /= i;

    float t = (thetaWo < smoothThetaWoRange[0]  ?  0.0f  :
            (thetaWo - smoothThetaWoRange[0]) / (smoothThetaWoRange[1] - smoothThetaWoRange[0]));

    c = lerp(c, getBsdfSample(localWo, localWi, useLerp), t);

    return c;
}


Color
LightStageCylinderBsdfSlice::getBsdfSample(const Vec3f &localWo,
        const Vec3f &localWi, bool useLerp) const
{
    static const float sFloatIndexEpsilon = 1e-3f;

    // Compute the bin for thetaO
    float cosThetaO = localWo.z;
    if (cosThetaO < sEpsilon) {
        return Color(0.0f);
    }
    float thetaO = acos(cosThetaO);
    float findexThetaO = thetaO / sHalfPi * (mSizeThetaO - sFloatIndexEpsilon);
    int indexThetaO = int(floor(findexThetaO));
    MNRY_ASSERT(indexThetaO >= 0  &&  indexThetaO < mSizeThetaO);

    // Compute P, N, T on the cylinder in light-stage space
    Vec3f P, N, T;
    computeCylLightStagePNT((mTop  ?  -thetaO  :  thetaO), P, N, T);
    scene_rdl2::math::ReferenceFrame frame(N, T);

    // Convert localWi to cylinder local space (see computeCylLightStagePNT()):
    // We rotate around N such that phiO = pi:
    // phiO' = pi;  phiI' = phiI + (pi - phiO)
    float phiO = atan2(localWo.y, localWo.x);
    Vec3f cylWi = rotate(localWi, Vec3f(0.0f, 0.0f, 1.0f),
            (mTop  ?  -phiO  :  sPi - phiO));

    // Convert wi to light-stage space
    Vec3f wi = frame.localToGlobal(cylWi);

#if PBR_LIGHTSTAGE_DEBUG
    //wi = localWi;
    std::cout << "localWo = " << localWo << ", localWi = " << localWi;
    std::cout << ", thetaO = " << thetaO/sPi*180.0f << ", phiO = " << phiO/sPi*180.0f;
    std::cout << ", indexThetaO = " << indexThetaO << std::endl;
    std::cout << "P = " << P << ", N = " << N << ", T = " << T << std::endl;
    std::cout << "cylWi = " << cylWi << ", lsWi = " << wi << std::endl;
#endif

    // Get three surrounding light directions and interpolate
    Vec3i lightOrderIndex;
    Vec3f weights;
    if (getFaceLightOrderIndex(P, wi, lightOrderIndex, weights)) {
        Color color;
        if (useLerp) {
            // Use linear interpolation across thetaO
            // Use barycentric interpolation across light directions
            int indexThetaONext = min(indexThetaO + 1, mSizeThetaO - 1);
            float t = findexThetaO - indexThetaO;

            Vec3i index[2];
            index[0][0] = getIndex(lightOrderIndex[0], indexThetaO);
            index[0][1] = getIndex(lightOrderIndex[1], indexThetaO);
            index[0][2] = getIndex(lightOrderIndex[2], indexThetaO);
            index[1][0] = getIndex(lightOrderIndex[0], indexThetaONext);
            index[1][1] = getIndex(lightOrderIndex[1], indexThetaONext);
            index[1][2] = getIndex(lightOrderIndex[2], indexThetaONext);

            Color c0 = lerp(Color(mData[index[0][0]], mData[index[0][0] + 1], mData[index[0][0] + 2]),
                            Color(mData[index[1][0]], mData[index[1][0] + 1], mData[index[1][0] + 2]), t);
            Color c1 = lerp(Color(mData[index[0][1]], mData[index[0][1] + 1], mData[index[0][1] + 2]),
                            Color(mData[index[1][1]], mData[index[1][1] + 1], mData[index[1][1] + 2]), t);
            Color c2 = lerp(Color(mData[index[0][2]], mData[index[0][2] + 1], mData[index[0][2] + 2]),
                            Color(mData[index[1][2]], mData[index[1][2] + 1], mData[index[1][2] + 2]), t);

            color = c0 * weights[0] + c1 * weights[1] + c2 * weights[2];

        } else {

            // Uses closest vertex according to barycentric coordinates
            int index;
            if (weights[0] > weights[1]  &&  weights[0] > weights[2]) {
                index = getIndex(lightOrderIndex[0], indexThetaO);
            } else if (weights[1] > weights[2]) {
                index = getIndex(lightOrderIndex[1], indexThetaO);
            } else {
                index = getIndex(lightOrderIndex[2], indexThetaO);
            }
            color = Color(mData[index], mData[index + 1], mData[index + 2]);
        }

#if PBR_LIGHTSTAGE_DEBUG
        std::cout << "lightOrderIndex = " << lightOrderIndex << ", weights = " << weights;
        std::cout << ", color = " << color << std::endl;
#endif

        return color;
    } else {
        return Color(0.0f);
    }
}


void
LightStageCylinderBsdfSlice::setBsdf(int lightOrderIndex, int indexThetaO,
        const Color &color)
{
    int index = getIndex(lightOrderIndex, indexThetaO);
    mData[index] = color.r;
    mData[index + 1] = color.g;
    mData[index + 2] = color.b;
}


//---------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

