// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderDriver.h"

#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/render/util/StrUtil.h>

namespace {

//
// This directive is used for debug of internal table dump of qstep loop finding main logic.
//
//#define DEBUG_NFILE_QSTEP

#ifdef DEBUG_NFILE_QSTEP
std::string
showAllStepGroup(int min, int max, int numFiles, int steps, const std::vector<unsigned> *tbl)
//
// Dump each steps value for each group.
// This function is mainly used for debugging purposes.
//    
{
    auto showV = [](int j, int v, int min, int max, int w) -> std::string {
        std::ostringstream ostr;
        if (v == max) {
            ostr << std::setw(w) << v << "<<";
        } else if (j == 0) {
            if (v < min) ostr << "-" << std::setw(w) << std::setfill('0') << v << '-';
            else         ostr << "(" << std::setw(w) << std::setfill('0') << v << ')';
        } else {
            ostr << std::setw(w) << v;
        }
        return ostr.str();
    };

    int sequenceSize = (tbl) ? tbl->size() : max;
    int w0 = std::to_string((int)std::ceil((float)sequenceSize / (float)steps)).size();
    int w1 = std::to_string(max).size();

    std::ostringstream ostr;
    ostr << "stepGroup (max:" << max << " numFiles:" << numFiles << " steps:" << steps << ") {\n";
    unsigned ii = 0;
    unsigned end = (tbl) ? tbl->size() : max;
    for (int i = 0; ; ++i) {
        for (int j = 0; j < steps; ++j) {
            if (j == 0) ostr << "  stepGroup-" << std::setw(w0) << std::setfill('0') << i << ": ";
            unsigned currV =
                (tbl) ?
                (*tbl)[ii++] : // return tbl items from index 0 (adaptive sampling case)
                ++ii; // return integer number sequence start from 1 (uniform sampling case)
            ostr << showV(j, currV, min, max, w1) << ' ';
            if (j == (steps - 1)) ostr << '\n';
            if (ii >= end) break;
        }
        if (ii >= end) break;
    }
    if (ii % steps != 0) ostr << '\n';
    ostr << "}";
    return ostr.str();
}
#endif // end DEBUG_NFILE_QSTEP

std::string
showCheckpointSampleTable(int userDefinedNumFiles, int min, int max, int steps,
                          const std::vector<unsigned> *sequenceTbl,
                          int *totalFilesForVerify = nullptr)
//
// This function creates strings which show conversion results from checkpoint total files
// to checkpoint quality steps. This information is useful to understand what kind of
// quality steps are used finally.
//    
// userDefinedNumFiles includes end sample data. See comment of
// RenderDriver::convertTotalCheckpointToQualitySteps() function.
//
{
    std::vector<unsigned> sampleTbl;
    unsigned ii = 0;
    unsigned end = (sequenceTbl) ? sequenceTbl->size() : max;
    while (1) {
        for (int j = 0; j < steps; ++j) {
            unsigned currSample =
                (sequenceTbl) ?
                (*sequenceTbl)[ii++] : // return sequenceTbl items from index 0 (adaptive sampling)
                ++ii; // return integer number sequence start from 1 (uniform sampling)
            if (j == 0 && ii < end ) {
                if (min <= (int)currSample) sampleTbl.push_back(currSample);
            }
            if (ii >= end) break;
        }
        if (ii >= end) break;
    }
    sampleTbl.push_back(max);   // we have to include end sample

    if (totalFilesForVerify) {
        *totalFilesForVerify = sampleTbl.size();
    }

    std::ostringstream ostr;
    ostr << "checkpoint_total_files:" << userDefinedNumFiles << " was converted ... "
         << "(minSPP:" << min << " qSteps:" << steps << ") SPP:{";
    for (size_t i = 0; i < sampleTbl.size(); ++i) {
        if (i != 0) ostr << ", ";
        ostr << sampleTbl[i];
    }
    ostr << "} checkpointFiles:" << sampleTbl.size();
    return ostr.str();
}

int
calcStepsBlockCount(int totalSteps, int steps)
//
// Calculate block number from given totalSteps and single block steps
//    
{
    // We don't include block which start at last steps item  (= totalSteps - 1)
    /* This is a naive version to understand the idea behind.
    int totalFiles = 0;
    for (int i = 0; i < totalSteps - 1; i += steps) {
        totalFiles++;
    }
    return totalFiles;
    */

    return (totalSteps <= 1) ? 0 : ((totalSteps - 1) / steps + (((totalSteps - 1) % steps == 0) ? 0 : 1));
}

int
calcQualitySteps(int totalSteps, int requestedNumFiles)
//
// Return quality steps of each checkpoint render based on the total steps and
// requested total number of files.
// Computed steps might not create the requested number of files due to steps being
// integer values but result steps create the closest number of files and best guess.
// totalSteps value is related to the pixel samples value when under uniform
// sampling and total table size of sample sequence (i.e. KJ sequence) when under
// adaptive sampling.
// requestedNumFiles is a number about how many checkpoint files you want to create
// from totalSteps of sampling sequence.
//    
{
#   ifdef DEBUG_NFILE_QSTEP
    std::ostringstream ostr;
    ostr << "calcQualitySteps() :"
         << " totalSteps:" << totalSteps
         << " reqNumFiles:" << requestedNumFiles;
#   endif // end DEBUG_NFILE_QSTEP

    int steps = 0;
    if (totalSteps <= requestedNumFiles) {
        steps = 1;              // minimum bound of steps
#       ifdef DEBUG_NFILE_QSTEP
        ostr << " => numBound steps:" << steps;
#       endif // end DEBUG_NFILE_QSTEP

    } else if (totalSteps % requestedNumFiles == 0) {
        steps = totalSteps / requestedNumFiles; // The remainder is 0, simply use the result as steps.
#       ifdef DEBUG_NFILE_QSTEP
        ostr << " => divisible steps:" << steps;
#       endif // end DEBUG_NFILE_QSTEP

    } else {
        // Not a divisible case,
        // we evaluate both of 2 possible steps and pick closer numbers to requestedNumFiles.
        int stepsA = totalSteps / requestedNumFiles;
        int stepsB = stepsA + 1;
        int fileTotalA = calcStepsBlockCount(totalSteps, stepsA);
        int fileTotalB = calcStepsBlockCount(totalSteps, stepsB);
        if (abs(fileTotalA - requestedNumFiles) < abs(fileTotalB - requestedNumFiles)) {
            steps = stepsA;
        } else {
            steps = stepsB;
        }
#       ifdef DEBUG_NFILE_QSTEP
        ostr << " => non-divisible {\n"
             << "  stepsA:" << stepsA << '\n'
             << "  stepsB:" << stepsB << '\n'
             << "  fileTotalA:" << fileTotalA
             << "  fileTotalB:" << fileTotalB
             << "  steps:" << steps << '\n'
             << "}";
#       endif // end DEBUG_NFILE_QSTEP
    }
#   ifdef DEBUG_NFILE_QSTEP
    std::cerr << ostr.str() << '\n';
#   endif // end DEBUG_NFILE_QSTEP
    return steps;
}

int
calcTotalCheckpoint(int totalSteps, int qualitySteps, int minSPP, const std::vector<unsigned> *sequenceTbl)
//
// Return total checkpoint files based on the totalSteps qualityStep minStep and sequenceTbl.
//
// totalSteps : This value is related to the pixel samples value when under uniform
//              sampling and total table size of sample sequence (i.e. KJ sequence) when under
//              adaptive sampling.
//
// qualitySteps : Quality steps value which computed by calcQualitySteps()
//
// minSPP : Minimum SPP which we have to include for checkpoint output.
//          Less than this minSPP value does not create any checkpoint files.
//
// sequenceTbl : specify KJ sequence table pointer for adaptive sampling, set nullptr for uniform sampling
//
{
    int totalFiles = calcStepsBlockCount(totalSteps, qualitySteps);

    if (minSPP <= 0) {
        // no minimum steps control case, we can use entire step sequence range.
        return totalFiles; // early exit
    }

    // We have to think about minSPP and count total skip checkpoint files
    // in order to calculate actual total number of checkpoint files.
    int skipFiles = 0;
    for (int i = 0; i < totalFiles; ++i) {
        int currCheckpointStartSPP = 0;
        if (!sequenceTbl) {
            // no sequenceTbl implies uniform sampling
            currCheckpointStartSPP = qualitySteps * i + 1;
        } else {
            // this is adaptive sampling case and get value from sequenceTbl
            currCheckpointStartSPP = (*sequenceTbl)[qualitySteps * i];
        }
        
        if (currCheckpointStartSPP < minSPP) {
            skipFiles++; // We have to skip this checkpoint file output
        } else {
            break; // We do regular checkpoint file output
        }
    }
    /* useful debug message
    std::cerr << "(totalSteps:" << totalSteps
              << " qualitySteps:" << qualitySteps
              << " minSPP:" << minSPP
              << " totalFiles:" << totalFiles
              << " skipFiles:" << skipFiles << "):"
              << totalFiles - skipFiles << '\n';
    */
    return totalFiles - skipFiles;
}

int
calcQualityStepsWithSampleStartControl(int minSPP, int maxSPP, int requestedNumFiles,
                                       const std::vector<unsigned> *sequenceTbl)
//
// You can specify checkpoint start sample as minSPP
// maxSPP is only used under uniform-sampling case (i.e. sequenceTbl == nullptr)
//
{
    int totalSteps;
    if (!sequenceTbl) {         // uniform-sampling
        totalSteps = maxSPP;
    } else {                    // adaptive-sampling
        totalSteps = sequenceTbl->size();
    }

    if (requestedNumFiles == 0) {
        return totalSteps;      // special case.
    }

    if (minSPP <= 0) {
        return calcQualitySteps(totalSteps, requestedNumFiles); // no minimum steps control
    }
    
    //
    // We have to consider minSPP value. In order to find the best qualitySteps,
    // we use a brute force solution here which is loop finding by increasing requestNumFiles
    // value. This is only executed once at the beginning of renderPrep and efficiency is not
    // so important.
    //        
    int currReqNumFiles = requestedNumFiles;
    int currSteps = calcQualitySteps(totalSteps, currReqNumFiles);
    int currTotalFiles = calcTotalCheckpoint(totalSteps, currSteps, minSPP, sequenceTbl);

    int bestSteps = currSteps;
    int bestTotalFiles = currTotalFiles;

#   ifdef DEBUG_NFILE_QSTEP
    {
        int maxSteps = (sequenceTbl) ? (*sequenceTbl)[totalSteps - 1] : totalSteps;
        std::cerr << ">> search loop for best param initial :"
                  << " maxSteps:" << maxSteps
                  << " currReqNumFiles:" << currReqNumFiles
                  << " currSteps:" << currSteps
                  << " minSPP:" << minSPP << ' '
                  << showAllStepGroup(minSPP, maxSteps, currReqNumFiles, currSteps, sequenceTbl)
                  << " currTotalFiles:" << currTotalFiles << '\n';
    }
#   endif // end DEBUG_NFILE_QSTEP    

    auto evalTotalFiles = [&]() {
        // Evaluate current total files condition regarding to improve the situation or worse.

        int absA = std::abs(currTotalFiles - requestedNumFiles);
        int absB = std::abs(bestTotalFiles - requestedNumFiles);
#       ifdef DEBUG_NFILE_QSTEP
        std::cerr << "evalTotalFiles() : currTotalFiles:" << currTotalFiles
                  << " reqNumFiles:" << requestedNumFiles
                  << " absA:" << absA
                  << " absB:" << absB
                  << " currSteps:" << currSteps
                  << " oldBestSteps:" << bestSteps;
#       endif // DEBUG_NFILE_QSTEP
        if (absA < absB || (absA == absB && bestSteps == currSteps)) {
            bestTotalFiles = currTotalFiles;
            bestSteps = currSteps;
#           ifdef DEBUG_NFILE_QSTEP
            std::cerr << " => bestTotalFiles:" << bestTotalFiles
                      << " bestSteps:" << bestSteps << '\n';
#           endif // DEBUG_NFILE_QSTEP            
        } else {
#           ifdef DEBUG_NFILE_QSTEP
            std::cerr << " => skip\n";
#           endif // DEBUG_NFILE_QSTEP            
        }
    };

    for (++currReqNumFiles; currReqNumFiles <= totalSteps; ++currReqNumFiles) {
        currSteps = calcQualitySteps(totalSteps, currReqNumFiles);
        currTotalFiles = calcTotalCheckpoint(totalSteps, currSteps, minSPP, sequenceTbl);
        if (currTotalFiles > bestTotalFiles + requestedNumFiles) {
            // Search all the cases would be overkill and too costly.
            // We already have best guess value and best result should be some where between
            // current bestTotalFiles to bestTotalFiles + requestedNumFiles.
            break;
        }
#       ifdef DEBUG_NFILE_QSTEP
        {
            int maxSteps = (sequenceTbl) ? (*sequenceTbl)[totalSteps - 1] : totalSteps;
            std::cerr << ">> search loop for best param :"
                      << " maxSteps:" << maxSteps
                      << " currReqNumFiles:" << currReqNumFiles
                      << " currSteps:" << currSteps
                      << " minSPP:" << minSPP << ' '
                      << showAllStepGroup(minSPP, maxSteps, currReqNumFiles, currSteps, sequenceTbl)
                      << " currTotalFiles:" << currTotalFiles << '\n';
        }
#       endif // end DEBUG_NFILE_QSTEP
        evalTotalFiles();
    }

    return bestSteps;
}

bool    
verifyConvertTotalCheckpointToQStepsMainLogic(int checkpointStartSPP,
                                              int maxSPP,
                                              int userDefinedTotalCheckpointFiles,
                                              int verifyQSteps,
                                              const std::vector<unsigned> *sequenceTbl,
                                              std::string *verifyMsg)
{
    //
    // To calculate quality steps value required totalCheckpointFiles count which does not
    // include the last checkpoint file (i.e. final output file). In order to do this we do
    // subtract 1 here. userDefinedTotalCheckpointFiles includes the last checkpoint file.
    //
    int internalTotalCheckpointFiles = std::max(userDefinedTotalCheckpointFiles - 1, 0);

    int totalSteps;
    if (!sequenceTbl) {         // uniform-sampling
        totalSteps = maxSPP;
    } else {                    // adaptive-sampling
        totalSteps = sequenceTbl->size();
    }

    std::ostringstream ostr;
    if (verifyMsg) {
        ostr << "verifyConvertTotalCheckpointToQStepsMainLogic = {\n"
             << "  checkpointStartSPP:" << checkpointStartSPP
             << " maxSPP:" << maxSPP
             << " userDefinedTotalCheckpointFiles:" << userDefinedTotalCheckpointFiles
             << " verifyQSteps:" << verifyQSteps
             << " sequenceTbl:0x" << std::hex << (uintptr_t)sequenceTbl << std::dec << '\n'
             << "  totalSteps:" << totalSteps << '\n';
    }

    int currTotal = calcTotalCheckpoint(totalSteps, verifyQSteps, checkpointStartSPP, sequenceTbl);
    if (currTotal == internalTotalCheckpointFiles) {
        if (verifyMsg) {
            ostr << "  currTotal:" << currTotal
                 << " == internalTotalCheckpointFiles:" << internalTotalCheckpointFiles << '\n'
                 << "} = Verify OK-0";
            (*verifyMsg) = ostr.str();
        }
        return true;
    }
    
    if (verifyMsg) {
        ostr << "  verify logic stageB start (currTotal:" << currTotal
             << " internalTotalCheckpointFiles:" << internalTotalCheckpointFiles << ")\n";
    }
    
    int bestTotalDelta = std::abs(currTotal - internalTotalCheckpointFiles);
    std::vector<int> bestQSteps;
    bestQSteps.push_back(verifyQSteps);

    if (verifyMsg) {
        ostr << "  initBestTotalDelta:" << bestTotalDelta << '\n'
             << "  qStep verify loop = {\n";
    }
    for (int qSteps = 1; qSteps <= totalSteps; ++qSteps) {
        currTotal = calcTotalCheckpoint(totalSteps, qSteps, checkpointStartSPP, sequenceTbl);
        int currTotalDelta = std::abs(currTotal - internalTotalCheckpointFiles);
        if (verifyMsg) {
            ostr << "    qSteps:" << qSteps
                 << " currTotal:" << currTotal
                 << " currTotalDelta:" << currTotalDelta << " -> ";
        }
        if (currTotalDelta < bestTotalDelta) {
            bestTotalDelta = currTotalDelta;
            bestQSteps.clear();
            bestQSteps.push_back(qSteps);
            if (verifyMsg) {
                ostr << "reset best qSteps:" << qSteps << " bestTotalDelta:" << bestTotalDelta << '\n';
            }
        } else if (currTotalDelta == bestTotalDelta) {
            bestQSteps.push_back(qSteps);
            if (verifyMsg) {
                ostr << "push qSteps:" << qSteps << '\n';
            }
        } else {
            if (verifyMsg) {
                ostr << "skip\n";
            }
        }
    }
    if (verifyMsg) {
        ostr << "  }\n"
             << "  resultCandidate = (bestTotalDelta:" << bestTotalDelta << ") = {\n";
        for (int i = 0; i < bestQSteps.size(); ++i) {
            ostr << "    i:" << std::setw(std::to_string(bestTotalDelta).size()) << i
                 << " bestQSteps:" << bestQSteps[i] << '\n';
        }
        ostr << "  }\n";
    }

    for (int i = 0; i < bestQSteps.size(); ++i) {
        if (verifyQSteps == bestQSteps[i]) {
            if (verifyMsg) {
                ostr << "} = Verify OK-1";
                (*verifyMsg) = ostr.str();
            }
            return true;
        }
    }

    if (verifyMsg) {
        ostr << "} = Verify NG";
        (*verifyMsg) = ostr.str();
    }

    return false;
}

bool    
verifyConvertTotalCheckpointToQStepsLogMessage(int checkpointStartSPP,
                                               int maxSPP,
                                               int userDefinedTotalCheckpointFiles,
                                               int verifyQSteps,
                                               const std::vector<unsigned> *sequenceTbl,
                                               std::string *verifyMsg)
{
    int totalSteps;
    if (!sequenceTbl) {         // uniform-sampling
        totalSteps = maxSPP;
    } else {                    // adaptive-sampling
        totalSteps = sequenceTbl->size();
    }

    std::ostringstream ostr;
    if (verifyMsg) {
        ostr << "verifyConvertTotalCheckpointToQStepsLogMessage = {\n"
             << "  checkpointStartSPP:" << checkpointStartSPP
             << " maxSPP:" << maxSPP
             << " userDefinedTotalCheckpointFiles:" << userDefinedTotalCheckpointFiles
             << " verifyQSteps:" << verifyQSteps
             << " sequenceTbl:0x" << std::hex << (uintptr_t)sequenceTbl << std::dec << '\n'
             << "  totalSteps:" << totalSteps << '\n';
    }

    int totalFiles = calcTotalCheckpoint(totalSteps, verifyQSteps, checkpointStartSPP, sequenceTbl);

    if (verifyMsg) {
        ostr << "  totalFiles:" << totalFiles << " + 1 = " << totalFiles + 1 << '\n';
    }

    std::string workLogBuffer;
    int workTotalFiles;
    workLogBuffer = showCheckpointSampleTable(userDefinedTotalCheckpointFiles,
                                              checkpointStartSPP,
                                              maxSPP,
                                              verifyQSteps,
                                              sequenceTbl,
                                              &workTotalFiles);

    if (verifyMsg) {
        ostr << "  workLogBuffer = \"" << workLogBuffer << "\"\n"
             << "  workTotalFiles:" << workTotalFiles << '\n';
    }

    if (totalFiles + 1 == workTotalFiles) {
        if (verifyMsg) {
            ostr << "  totalFiles:" << totalFiles << " + 1 == " << "workTotalFiles:" << workTotalFiles << '\n'
                 << "} = Verify-OK";
            (*verifyMsg) = ostr.str();
        }
        return true;
    }
    if (verifyMsg) {
        ostr << "  totalFiles:" << totalFiles << " + 1 != " << "workTotalFiles:" << workTotalFiles << '\n'
             << "} = Verify-NG";
        (*verifyMsg) = ostr.str();
    }
    return false;
}

} // namespace

namespace moonray {
namespace rndr {

// static function
int
RenderDriver::convertTotalCheckpointToQualitySteps(SamplingMode mode,
                                                   int checkpointStartSPP,
                                                   int maxSPP,
                                                   int userDefinedTotalCheckpointFiles,
                                                   std::string &logMessage)
//
// Return qualitySteps value which converted from totalCheckpointFiles
//    
{
    //
    // To calculate quality steps value required totalCheckpointFiles count which does not
    // include the last checkpoint file (i.e. final output file). In order to do this we do
    // subtract 1 here. userDefinedTotalCahcekpointFiles includes the last checkpoint file.
    //
    int internalTotalCheckpointFiles = std::max(userDefinedTotalCheckpointFiles - 1, 0);

    std::vector<unsigned> tbl, *tblPtr;
    if (mode == SamplingMode::UNIFORM) {
        tblPtr = nullptr;
    } else {
        tbl = createKJSequenceTable(maxSPP);
        tblPtr = &tbl;
    }

    int qualitySteps =
        calcQualityStepsWithSampleStartControl(checkpointStartSPP,
                                               maxSPP,
                                               internalTotalCheckpointFiles,
                                               tblPtr);

    logMessage = showCheckpointSampleTable(userDefinedTotalCheckpointFiles,
                                           checkpointStartSPP,
                                           maxSPP,
                                           qualitySteps,
                                           tblPtr);
                                               
    return qualitySteps;
}

// static function
bool
RenderDriver::verifyKJSequenceTable(const unsigned maxSampleId, std::string *tblStr)
//
// This function is used for unitTest purposes.
// Verify KJ sequence table which is used to calculate tile sampleId start/end span for checkpoint interval.
//    
{
    std::vector<unsigned> table = RenderDriver::createKJSequenceTable(maxSampleId);

    auto dumpTblSimple = [&](const std::vector<unsigned> &tbl) -> std::string {
        std::ostringstream ostr;
        int w = std::to_string(tbl.back()).size();
        for (size_t i = 0; i < tbl.size() - 1; ++i) {
            ostr << std::setw(w) << tbl[i] << ',' << ((i % 10 < 9) ? ' ' : '\n');
        }
        ostr << std::setw(w) << tbl[tbl.size() - 1];
        return ostr.str();
    };
    if (tblStr) {
        std::ostringstream ostr;
        ostr << "KJsequence (total:" << table.size() << ") = {\n"
             << scene_rdl2::str_util::addIndent(dumpTblSimple(table)) << '\n'
             << "}";
        (*tblStr) = ostr.str();
    }

    if (table.size() <= 0) return false;
    if (table[0] != 1) return false; // KJ sequence need to start 1
    if (table[table.size() - 1] != maxSampleId) return false; // KJ sequence need to end maxSampleId

    if (table.size() > 2) {
        // This is a heuristic approach. The KJ sequence is monotonically increased but the last table
        // value is created by just adding maxSPP value. We have to check duplication of last value.
        // This test is expanded a bit more and verifying monotonically increasing with no duplication
        // over the entire table.
        for (size_t i = 1; i < table.size(); ++i) {
            if (table[i - 1] >= table[i]) return false;
        }
    }
    return true;
}

// static function
bool
RenderDriver::verifyTotalCheckpointToQualitySteps(SamplingMode mode,
                                                  int checkpointStartSPP,
                                                  int maxSPP,
                                                  int userDefinedTotalCheckpointFiles,
                                                  int verifyQSteps,
                                                  std::string *verifyErrorMessage,
                                                  std::string *logMessage)
//
// This function is used for unitTest purposes.
// Verify a single set of argument patterns.
// Usually unittest calls this function over and over (by verifyTotalCheckpointToQualityStepsExhaust()).
//
{
    std::vector<unsigned> tbl, *tblPtr;
    if (mode == SamplingMode::UNIFORM) {
        tblPtr = nullptr;
    } else {
        tbl = createKJSequenceTable(maxSPP);
        tblPtr = &tbl;
    }

    std::string verifyMsgMainLogic, verifyMsgLogMessage;
    bool mainLogicVerifyResult =
        verifyConvertTotalCheckpointToQStepsMainLogic(checkpointStartSPP,
                                                      maxSPP,
                                                      userDefinedTotalCheckpointFiles,
                                                      verifyQSteps,
                                                      tblPtr,
                                                      (verifyErrorMessage) ? &verifyMsgMainLogic : nullptr);
    bool logMessageVerifyResult =
        verifyConvertTotalCheckpointToQStepsLogMessage(checkpointStartSPP,
                                                       maxSPP,
                                                       userDefinedTotalCheckpointFiles,
                                                       verifyQSteps,
                                                       tblPtr,
                                                       (verifyErrorMessage) ? &verifyMsgLogMessage : nullptr);
    if (!mainLogicVerifyResult || !logMessageVerifyResult) {
        if (verifyErrorMessage) {
            std::ostringstream ostr;
            ostr << "verifyTotalCheckpointToQualitySteps result = {\n"
                 << scene_rdl2::str_util::addIndent(verifyMsgMainLogic) << '\n'
                 << scene_rdl2::str_util::addIndent(verifyMsgLogMessage) << '\n'
                 << "}";
            (*verifyErrorMessage) = ostr.str();
        }
        return false;
    }

    if (logMessage) {
        (*logMessage) = showCheckpointSampleTable(userDefinedTotalCheckpointFiles,
                                                  checkpointStartSPP,
                                                  maxSPP,
                                                  verifyQSteps,
                                                  tblPtr);
    }
    return true;
}
    
// static function
bool
RenderDriver::verifyTotalCheckpointToQualityStepsExhaust(SamplingMode mode,
                                                         int maxSPP,
                                                         int fileCountEndCap,
                                                         int startSPPEndCap,
                                                         bool liveMessage,
                                                         bool deepVerifyMessage,
                                                         std::string *verifyMessagePtr,
                                                         bool multiThreadA,
                                                         bool multiThreadB)
//
// This function is used by unitTest in order to verify RenderDriver::convertTotalCheckpointToQualitySteps()
// function by a very brute-force way.
// It requires a pretty long run to complete if we verify all possible cases.
// In order to reduce the cost of verification, we have to set each parameter carefully.
// (See rndr/unittest/TestCheckpoint.cc as well).
//
//   mode : SamplingMode::UNIFORM or SamplingMode::ADAPTIVE
//
//   maxSPP : max SPP for verify test
//
//   fileCountEndCap : end cap for checkpoint file count.
//     If you set 0 for fileCountEndCap, this function verifies all possible checkpoint file counts.
//     However this is pretty slow even if we use multi-threaded mode.
//     Typical production only uses a very small number of checkpoint file counts like less than 10.
//     You can set some cap number (= bigger than 0) and set maximum number of checkpoint file count
//     for verification. This reduces the cost of verification a lot.
//
//   startSPPEndCap : end cap for checkpoint start SPP.
//     If you set -1 for startSPPEndCap, this function verifies all possible checkpoint start sample control.
//     However this is pretty slow even if we use multi-threaded mode.
//     Typical production only uses a very small number of checkpoint start sample values like 0 or 1.
//     You can set some cap number (= 0 or positive) and set maximum number of testing checkpoint start
//     sample values. This reduces the cost of total verification a lot too.
// 
//  liveMessage : show execution live message for debug purpose
//
//  deepVerifyMessage : verify message volume control
//    set true If you need detail verify log (but verify operation is slow down).
//    set false if you are satisfied by a simplified version of the verified result message.
//
//  verifyMessagePtr : output verify message buffer
//    You need to set the verify message output string pointer. If you set nullptr, deepVerifyMessage
//    condition is ignored and no message is output. If you set output string address, output message
//    volume is changed depending on the deepVerifyMessage flag.
//                                                                
//   multiThreadA : thread on/off control for internal main loop. useful for debug when we hit verify error.
//
//   multiThreadB : thread on/off control for internal sub loop. useful for debug when we hit verify error.
// 
//   See unitTest code for current testing parameter (lib/rendering/rndr/unittest/TestCheckpoint.cc).
//                                                                    
{
    auto verifySingleSPP = [](int currMaxSPP, int fileCountEndCap, int startSPPEndCap,
                              bool multiThreadMode,
                              SamplingMode mode,
                              bool liveMessage,
                              bool deepVerifyMessage, std::string *verifyErrorMessagePtr,
                              float &totalConvertSec, float &totalVerifySec,
                              int &totalVerifyCount) -> bool {

        auto verifySingleSPPNumFile = [](int currFileCount, int startSPPEndCap, int currMaxSPP,
                                         SamplingMode mode,
                                         bool deepVerifyMessage, std::string *verifyErrorMessagePtr,
                                         float &totalConvertSec, float &totalVerifySec,
                                         int &totalVerifyCount) -> bool {
            scene_rdl2::rec_time::RecTime time;
            // This is a main loop of testing about checkpoint_start_sample control.
            // Typically, production does not use this control actively and only used start value as 0 or 1.
            // So probably we don't need to verify entire range.
            int startSPPEnd = currMaxSPP;
            if (startSPPEndCap >= 0 && currMaxSPP > startSPPEndCap) {
                startSPPEnd = startSPPEndCap;
            }
            for (int currStartSPP = 0; currStartSPP <= startSPPEnd; ++currStartSPP) {
                std::string logMessage;
                time.start();
                int qSteps =
                    convertTotalCheckpointToQualitySteps(mode, currStartSPP, currMaxSPP, currFileCount,
                                                         logMessage);
                totalConvertSec += time.end();

                std::string *verifyErrMsgPtr = (deepVerifyMessage) ? verifyErrorMessagePtr : nullptr;
                time.start();
                bool verifyResult =
                    verifyTotalCheckpointToQualitySteps(mode, currStartSPP, currMaxSPP, currFileCount,
                                                        qSteps, verifyErrMsgPtr);
                totalVerifySec += time.end();
                totalVerifyCount++;

                if (!verifyResult) {
                    if (verifyErrorMessagePtr && !deepVerifyMessage) {
                        std::ostringstream ostr;
                        ostr << "verifyTotalCheckpointToQualitySteps FAILED = {\n"
                             << "  mode:" << ((mode == SamplingMode::UNIFORM) ? "uniform" : "adaptive") << '\n'
                             << "  currStartSPP:" << currStartSPP << '\n'
                             << "  currMaxSPP:" << currMaxSPP << '\n'
                             << "  currFileCount:" << currFileCount << '\n'
                             << "  qSteps:" << qSteps << '\n'
                             << "}";
                        (*verifyErrorMessagePtr) = ostr.str();
                    }
                    return false;
                }
            }
            return true;
        }; // end verifySingleSPPNumFile()

        if (liveMessage) {
            std::cerr << "currMaxSPP:" << currMaxSPP << '\n';
        }

        int fileCountMax =
            (mode == SamplingMode::UNIFORM) ? currMaxSPP : createKJSequenceTable(currMaxSPP).size();
        if (fileCountEndCap >= 1 && fileCountMax > fileCountEndCap) {
            // It does not makesense production creates too many checkpoint files like 128 or more for
            // example. So it is a good idea to clip the test range by proper value of fileCountEndCap.
            fileCountMax = fileCountEndCap;
        }

        // This is a main loop of testing about checkpoint_total_files control at particular maxSPP
        // and sampling mode.
        bool verifyResult = true;
        if (!multiThreadMode) {
            //
            // single thread mode
            //
            for (int currFileCount = 0; currFileCount < fileCountMax; ++currFileCount) {
                if (!verifySingleSPPNumFile(currFileCount, startSPPEndCap,
                                            currMaxSPP, mode, deepVerifyMessage,
                                            verifyErrorMessagePtr,
                                            totalConvertSec,
                                            totalVerifySec,
                                            totalVerifyCount)) {
                    verifyResult = false;
                    break;
                }
            }

        } else {
            //
            // multi thread mode
            //
            std::vector<std::string> localVerifyErrorMessageArray;
            if (verifyErrorMessagePtr) {
                localVerifyErrorMessageArray.resize(fileCountMax);
            }
            std::vector<float> localTotalConvertSecArray(fileCountMax, 0.0f);
            std::vector<float> localTotalVerifySecArray(fileCountMax, 0.0f);
            std::vector<int> localTotalVerifyCountArray(fileCountMax, 0);

            tbb::blocked_range<size_t> range(0, fileCountMax, 1);
            tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t currFileCount = r.begin(); currFileCount < r.end(); ++currFileCount) {
                        if (!verifySingleSPPNumFile
                            (currFileCount,
                             startSPPEndCap,
                             currMaxSPP, mode, deepVerifyMessage,
                             (verifyErrorMessagePtr) ? &localVerifyErrorMessageArray[currFileCount] : nullptr,
                             localTotalConvertSecArray[currFileCount],
                             localTotalVerifySecArray[currFileCount],
                             localTotalVerifyCountArray[currFileCount])) {
                            verifyResult = false;
                            return;
                        }
                    }
                });

            // Gathering result
            for (int i = 0; i < fileCountMax; ++i) {
                if (verifyErrorMessagePtr && !localVerifyErrorMessageArray[i].empty()) {
                    (*verifyErrorMessagePtr) += localVerifyErrorMessageArray[i];
                }
                totalConvertSec += localTotalConvertSecArray[i];
                totalVerifySec += localTotalVerifySecArray[i];
                totalVerifyCount += localTotalVerifyCountArray[i];
            }
        }

        return verifyResult;
    }; // end verifySingleSPP()

    scene_rdl2::rec_time::RecTime timeAll;
    if (verifyMessagePtr) {
        // Initial log message
        auto boolStr = [](bool flag) -> std::string { return (flag) ? "true" : "false"; };
        std::ostringstream ostr;
        ostr << "verifyTotalCheckpointToQualityStepsExhaust() = {\n"
             << "  mode:" << ((mode == SamplingMode::UNIFORM) ? "uniform" : "adaptive") << '\n'
             << "  maxSPP:" << maxSPP << '\n'
             << "  fileCountEndCap:" << fileCountEndCap << '\n'
             << "  startSPPEndCap:" << startSPPEndCap << '\n'
             << "  deepVerifyMessage:" << boolStr(deepVerifyMessage) << '\n'
             << "  multiThreadA:" << boolStr(multiThreadA) << '\n'
             << "  multiThreadB:" << boolStr(multiThreadB) << '\n'
             << "} ";
        (*verifyMessagePtr) += ostr.str();

        timeAll.start();
    }

    // maxSPP table generation for verify testing
    std::vector<int> maxSPPTbl;
    if (mode == SamplingMode::UNIFORM) {
        // generate maxSPPTable here. We only have squared number of pixel sample control
        for (int spp = 1; spp * spp <= maxSPP; ++spp) {
            maxSPPTbl.push_back(spp * spp);
        }
    } else {
        // adaptive sampling case, table is simple.
        for (int spp = 1; spp <= maxSPP; ++spp) {
            maxSPPTbl.push_back(spp);
        }
    }
    /* useful debug message
    for (int i = 0; i < maxSPPTbl.size(); ++i) {
        std::cerr << maxSPPTbl[i] << ' ';
    }
    std::cerr << '\n';
    */

    bool multiThreadSPP = multiThreadA;
    bool multiThreadNumFile = multiThreadB;

    bool verifyResult = true;
    float totalConvertSec = 0.0f;
    float totalVerifySec = 0.0f;
    int totalVerifyCount = 0;

    // This is a main loop of testing for various different maxSPP values.
    if (!multiThreadSPP) {
        //
        // single thread mode
        //
        for (int taskId = 0; taskId < maxSPPTbl.size(); ++taskId) {
            int currMaxSPP = maxSPPTbl[taskId];
            if (!verifySingleSPP(currMaxSPP, fileCountEndCap, startSPPEndCap,
                                 multiThreadNumFile,
                                 mode,
                                 liveMessage,
                                 deepVerifyMessage, verifyMessagePtr,
                                 totalConvertSec, totalVerifySec, totalVerifyCount)) {
                verifyResult = false;
                break; // exit on verify error
            }
        }

    } else {
        //
        // multi thread mode
        //
        int size = maxSPPTbl.size();
        std::vector<std::string> verifyErrorMessageArray;
        if (verifyMessagePtr) {
            verifyErrorMessageArray.resize(size);
        }
        std::vector<float> totalConvertSecArray(size, 0.0f);
        std::vector<float> totalVerifySecArray(size, 0.0f);
        std::vector<int> totalVerifyCountArray(size, 0);

        tbb::blocked_range<size_t> range(0, maxSPPTbl.size(), 1);
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t taskId = r.begin(); taskId < r.end(); ++taskId) {
                    int currMaxSPP = maxSPPTbl[taskId];
                    if (!verifySingleSPP(currMaxSPP, fileCountEndCap, startSPPEndCap,
                                         multiThreadNumFile,
                                         mode,
                                         liveMessage,
                                         deepVerifyMessage,
                                         (verifyMessagePtr) ? &verifyErrorMessageArray[taskId] : nullptr,
                                         totalConvertSecArray[taskId], totalVerifySecArray[taskId],
                                         totalVerifyCountArray[taskId])) {
                        verifyResult = false;
                        return;
                    }
                }
            });

        // Gathering result
        for (int i = 0; i < size; ++i) {
            if (verifyMessagePtr && !verifyErrorMessageArray[i].empty()) {
                (*verifyMessagePtr) += verifyErrorMessageArray[i];
            }
            totalConvertSec += totalConvertSecArray[i];
            totalVerifySec += totalVerifySecArray[i];
            totalVerifyCount += totalVerifyCountArray[i];
        }
    }

    if (verifyMessagePtr) {
        // final log message
        float timeSecAll = timeAll.end();
        std::ostringstream ostr;
        ostr << "\nverifySummary = {\n"
             << "  accumulate-convert:" << totalConvertSec << " sec\n"
             << "   accumulate-verify:" << totalVerifySec << " sec\n"
             << "                 all:" << timeSecAll << " sec\n"
             << "                     " << (totalConvertSec + totalVerifySec) / timeSecAll << "x\n"
             << "         verifyTotal:" << totalVerifyCount << '\n'
             << "        verifyResult:" << ((verifyResult) ? "OK" : "NG") << '\n'
             << "}";
        (*verifyMessagePtr) += ostr.str();
    }

    return verifyResult;
}

} // namespace rndr
} // namespace moonray

