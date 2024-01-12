// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include <utility>

// Temp
#include <iostream>
#define prints(v) std::cout << #v << ": " << (v) << '\n'

template <typename Point>
class PointManager2D
{
    typedef std::list<Point> ListType;
    typedef NPoint<Point::sDims - 1u> ProjectedPoint;

public:
    typedef typename ListType::iterator iterator;
    typedef typename ListType::const_iterator const_iterator;
    typedef typename ListType::size_type size_type;

    template <typename PointIter, typename floatIter>
    PointManager2D(PointIter pointFirst, PointIter pointLast, floatIter floatFirst, floatIter floatLast)
    {
        for ( ; floatFirst != floatLast; ++floatFirst) {
            for (PointIter x = pointFirst; x != pointLast; ++x) {
                mPoints.emplace_front(*x, *floatFirst);
                auto p = std::make_shared<iterator>(mPoints.begin());
                mKeysX.insert(std::make_pair(*x, p));
                mKeysY.insert(std::make_pair(*floatFirst, p));
            }
            std::cout << "Outer\n";
        }
        std::cout << "Done\n";
    }

    void erase(const Point& p)
    {
        erase(p.strip(), p.back());
    }

    void erase(iterator it)
    {
        if (it != mPoints.end()) {
            erase(*it);
        }
    }

    void erase(const ProjectedPoint& x, float y)
    {
        auto xrange = mKeysX.equal_range(x);
        auto yrange = mKeysY.equal_range(y);

        for (auto xit = xrange.first; xit != xrange.second; ++xit) {
            if (xit->second && !xit->second.unique()) {
                mPoints.erase(*(xit->second));
            }
            xit->second.reset();
        }
        for (auto yit = yrange.first; yit != yrange.second; ++yit) {
            if (yit->second && !yit->second.unique()) {
                mPoints.erase(*(yit->second));
            }
            yit->second.reset();
        }

        mKeysY.erase(yrange.first, yrange.second);
        mKeysX.erase(xrange.first, xrange.second);
    }

    size_type size() const
    {
        return mPoints.size();
    }

    bool empty() const
    {
        return mPoints.empty();
    }

    iterator begin()
    {
        return mPoints.begin();
    }

    iterator end()
    {
        return mPoints.end();
    }

    const_iterator begin() const
    {
        return mPoints.cbegin();
    }

    const_iterator end() const
    {
        return mPoints.cend();
    }

    const_iterator cbegin() const
    {
        return mPoints.cbegin();
    }

    const_iterator cend() const
    {
        return mPoints.cend();
    }

private:
    typedef std::unordered_multimap<float, std::shared_ptr<iterator>> FloatKeyList;
    typedef std::unordered_multimap<ProjectedPoint, std::shared_ptr<iterator>> PointKeyList;

    ListType mPoints;
    PointKeyList mKeysX;
    FloatKeyList mKeysY;
};

