// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file State.h
/// $Id$
///

#ifndef GEOM_STATE_HAS_BEEN_INCLUDED
#define GEOM_STATE_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/Types.h>

#include <string>

namespace moonray {
namespace geom {

//----------------------------------------------------------------------------

///
/// @class State State.h <rendering/geom/State.h>
/// @brief The State is a tool that helps a set of procedurals and primitives
/// keep track of transforms and part sub-names hierarchically.
/// 
class State
{
public:
    /// Constructor / Destructor
    State() : mProc2Parent(scene_rdl2::math::one) {}
    ~State() = default;
    
    /// Set / Get transform from procedural space to parent space
    void setProc2Parent(const Mat43 &proc2parent) {
        mProc2Parent = proc2parent;
    }

    const Mat43 &getProc2Parent() const {
        return mProc2Parent;
    }

    void translate()  {   }
    void rotate()     {   }
    void scale()      {   }
    
    
    /// Set/get the name for this state, which will be used as a sub-name
    /// in the fully qualified part name
    void setName(const std::string &name)   {  mName = name;  }
    const std::string &getName() const      {  return mName;  }
    
    /// Convenience function to append a suffix to the name. push() returns a
    /// pop-hint == the length of the name before the push and should be passed
    /// to pop() to restore the name to its previous length.
    int pushNameSuffix(const std::string &suffix)
    {
        int popHint = mName.length();
        mName.append(suffix);
        return popHint;
    }
    void popNameSuffix(int popHint)     {  mName.resize(popHint);  }


private:
    Mat43 mProc2Parent;
    std::string mName;
};


//----------------------------------------------------------------------------

} // namespace geom
} // namespace moonray

#endif /* GEOM_STATE_HAS_BEEN_INCLUDED */

