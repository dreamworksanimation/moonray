// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

class ProgressBar
{
public:
    explicit ProgressBar(unsigned total) : mStep(0), mTotal(total) {}

    void update(unsigned n = 1)
    {
        mStep += n;
    }

    void draw()
    {
        using namespace std::chrono_literals;

        const auto now = std::chrono::system_clock::now();
        if (mStep < mTotal && now - mLastTime < 5.0s) {
            return;
        }
        mLastTime = now;

        const unsigned percentage = calcPercentage(mStep, mTotal);

        constexpr char marker = '=';
        constexpr unsigned width = 50;
        const unsigned numDraw = calcNumDraw(mStep, mTotal, width);

        const std::string bar(numDraw, marker);
        std::cout << '\r'
                  << std::right << std::setfill(' ') << std::setw(4) << percentage << "% |"
                  << std::left  << std::setfill(' ') << std::setw(width) << bar << "> "
                  << mStep << '/' << mTotal
                  << std::flush;
    }

private:
    static unsigned calcPercentage(unsigned step, unsigned total)
    {
        return (total > 100) ? step/(total/100) : step*100/total;
    }

    static unsigned calcNumDraw(unsigned step, unsigned total, unsigned width)
    {
        return (total > width) ? step/(total/width) : step*width/total;
    }

    using TimePoint = std::chrono::system_clock::time_point;
    TimePoint mLastTime;
    unsigned mStep;
    unsigned mTotal;
};

