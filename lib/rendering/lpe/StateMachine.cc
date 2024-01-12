// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file StateMachine.cc

#include "StateMachine.h"

#include "osl/automata.h"
#include "osl/oslclosure.h"
#include "osl/lpexp.h"
#include "osl/lpeparse.h"
#include "osl/optautomata.h"

#include <scene_rdl2/common/platform/Platform.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace moonray {
namespace lpe {

class StateMachine::Impl
{
public:
    Impl();
    ~Impl();

    int addExpression(std::string const &lp, int id);
    void addLabel(std::string const &label);
    int getLabelId(std::string const &label) const;
    void build();
    int transition(int stateId, EventType ev, EventScatteringType evs, int labelId) const;
    bool isValid(int stateId, int id) const;

private:
    typedef std::vector<std::tuple<std::string, int, osl::lpexp::LPexp *> > Expressions;
    typedef std::vector<osl::ustring> Labels;

    Expressions mExpressions;
    Labels mLabels;
    const osl::ustring mExtraLabel;
    const osl::ustring mMaterialLabel; // For material AOVs that have an LPE
    osl::DfOptimizedAutomata mOptFsm;
    bool mBuilt;
};

StateMachine::Impl::Impl():
    mExtraLabel("U"),
    mMaterialLabel("M"),
    mBuilt(false)
{
}

StateMachine::Impl::~Impl()
{
    if (!mBuilt) {
        // we need to free up the lpexp
        for (auto &exp: mExpressions) {
            osl::lpexp::LPexp *l = std::get<2>(exp);
            delete l;
        }
    }
}

static std::string
trim(const std::string &in)
{
    // trim leading and trailing white space
    const std::string ws = " \t";
    const std::size_t beg = in.find_first_not_of(ws);
    if (beg == std::string::npos) {
        return "";
    }
    const std::size_t end = in.find_last_not_of(ws);
    return in.substr(beg, end - beg + 1);
}

int
StateMachine::Impl::addExpression(std::string const &lp, int id)
{
    MNRY_ASSERT(!mBuilt);

    int result = 0;
    const std::vector<osl::ustring> userEvents { mExtraLabel, mMaterialLabel };

    osl::Parser p(&userEvents);
    osl::lpexp::LPexp *l = p.parse(lp.c_str());
    if (!l) {
        result = -1;
    } else {
#ifdef DEBUG
        // check that id is not currently in use
        for (auto const &exp: mExpressions) {
            MNRY_ASSERT(std::get<1>(exp) != id);
        }
#endif
        mExpressions.push_back(std::make_tuple(lp, id, l));
    }

    // scan lp for labels, any trimmed value between two tick chars
    // is a label
    auto t0 = lp.find_first_of('\'');
    while (t0 != std::string::npos) {
        auto t1 = lp.find_first_of('\'', t0 + 1);
        MNRY_ASSERT(t1 != std::string::npos); // otherwise p.parse would have failed
        addLabel(trim(lp.substr(t0 + 1, t1 - t0 - 1)));
        t0 = lp.find_first_of('\'', t1 + 1);
    }

    return result;
}

void
StateMachine::Impl::addLabel(std::string const &label)
{
    MNRY_ASSERT(!mBuilt);

    int result = getLabelId(label);
    if (result >= 0) return;

    // need to create it
    osl::ustring us(label);
    mLabels.push_back(us);
}

int
StateMachine::Impl::getLabelId(std::string const &label) const
{
    int result = -1;
    osl::ustring us(label);
    auto it = std::find(mLabels.begin(), mLabels.end(), us);
    if (it != mLabels.end()) {
        result = static_cast<int>(it - mLabels.begin());
    }
    return result;
}

void
StateMachine::Impl::build()
{
    /// In order to fix the exclusion symbol "^", we had to add this "placeholder" label internally. This is to combat
    /// the way the OSL automata works internally. For example, say we're at state 4 and want to progress to 5. 
    /// The automata has created a transition that looks like this:
    ///     4: excluded_label_name: -1  _stop_: -1   anything else: 5
    /// We only try to perform this transition for labels if they exist, otherwise we get a "_stop_" 
    /// (see StateMachine::transition()). This means that if there are no labels, we erroneously get a state id of -1. 
    /// I'm not sure whether this is a bug in OSL, but for now this label (which only exists internally) 
    /// will ensure we always have a label to process. 
    addLabel("placeholder");

    MNRY_ASSERT(!mBuilt);

    osl::NdfAutomata ndf;
    osl::DfAutomata  fsm;

    if (!mExpressions.empty()) {
        // build a single ndf
        for (auto const &exp: mExpressions) {
            osl::lpexp::LPexp *l = std::get<2>(exp);
            int id = std::get<1>(exp);
            osl::lpexp::Rule rule(l, reinterpret_cast<void *>(static_cast<intptr_t>(id)));
            rule.genAuto(ndf);
        }

        // build the fsm
        ndfautoToDfauto(ndf, fsm);

        // MNRY_ASSERT(mFsm.size() > 0);
        MNRY_ASSERT(fsm.size() > 0);

        // build the optimized fsm
        mOptFsm.compileFrom(fsm);

        mBuilt = true;
    }
}

int
StateMachine::Impl::transition(int stateId, EventType ev, EventScatteringType evs, int labelId) const
{
    MNRY_ASSERT(mBuilt);

    int newStateId = stateId;
    if (newStateId < 0) return newStateId;

    // event
    switch (ev) {
    case EVENT_TYPE_CAMERA:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::CAMERA);
        break;
    case EVENT_TYPE_REFLECTION:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::REFLECT);
        break;
    case EVENT_TYPE_TRANSMISSION:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::TRANSMIT);
        break;
    case EVENT_TYPE_VOLUME:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::VOLUME);
        break;
    case EVENT_TYPE_LIGHT:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::LIGHT);
        break;
    case EVENT_TYPE_EMISSION:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::OBJECT);
        break;
    case EVENT_TYPE_BACKGROUND:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::BACKGROUND);
        break;
    case EVENT_TYPE_EXTRA:
        newStateId = mOptFsm.getTransition(newStateId, mExtraLabel);
        break;
    case EVENT_TYPE_MATERIAL:
        newStateId = mOptFsm.getTransition(newStateId, mMaterialLabel);
        break;
    case EVENT_TYPE_NONE:
        break;
    default:
        newStateId = -1; // broken
        break;
    }

    if (newStateId < 0) return newStateId;

    // scattering event
    switch (evs) {
    case EVENT_SCATTERING_TYPE_NONE:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::NONE);
        break;
    case EVENT_SCATTERING_TYPE_DIFFUSE:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::DIFFUSE);
        break;
    case EVENT_SCATTERING_TYPE_GLOSSY:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::GLOSSY);
        break;
    case EVENT_SCATTERING_TYPE_MIRROR:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::SINGULAR);
        break;
    case EVENT_SCATTERING_TYPE_STRAIGHT:
        newStateId = mOptFsm.getTransition(newStateId, osl::Labels::STRAIGHT);
        break;
    default:
        newStateId = -1; // broken
    }

    if (newStateId < 0) return newStateId;

    // labelId
    if (labelId >= 0) {
        MNRY_ASSERT(static_cast<unsigned int>(labelId) < mLabels.size());
        newStateId = mOptFsm.getTransition(newStateId, mLabels[labelId]);
    } else {
        /// The "^" requires us to always process a label of some kind (see explanation in StateMachine::Impl::build)
        newStateId = mOptFsm.getTransition(newStateId, mLabels[mLabels.size()-1]);
    }

    if (newStateId < 0) return newStateId;

    // and finish evey hit event with a stop
    newStateId = mOptFsm.getTransition(newStateId, osl::Labels::STOP);

    return newStateId;
}

bool
StateMachine::Impl::isValid(int stateId, int id) const
{
    MNRY_ASSERT(mBuilt);

    bool result = false;
 
    if (stateId >= 0) {
        int nrules = 0;
        void * const * rules = mOptFsm.getRules(stateId, nrules);
        for (int i = 0; i < nrules; ++i) {
            intptr_t rule = reinterpret_cast<intptr_t>(rules[i]);
            if (rule == id) {
                result = true;
                break;
            }
        }
    }

    return result;
}

// ----------------------------------------------------------------------------
StateMachine::StateMachine():
    mImpl { new Impl }
{
}

StateMachine::~StateMachine()
{
}

int
StateMachine::addExpression(std::string const &lp, int id)
{
    return mImpl->addExpression(lp, id);
}

int
StateMachine::getLabelId(std::string const &label) const
{
    return mImpl->getLabelId(label);
}

void
StateMachine::build()
{
    mImpl->build();
}

int
StateMachine::transition(int stateId, EventType ev, EventScatteringType evs, int labelId) const
{
    return mImpl->transition(stateId, ev, evs, labelId);
}

bool
StateMachine::isValid(int stateId, int id) const
{
   return mImpl->isValid(stateId, id);
}


// ispc hook
extern "C" int
CPP_LpeStateMachine_transition(const uint8_t *stateMachine, int stateId, EventType ev, EventScatteringType evs, int labelId)
{
    // and now for a super un-safe cast
    const StateMachine *s = reinterpret_cast<const StateMachine *>(stateMachine);

    return s->transition(stateId, ev, evs, labelId);
}

} // namespace lpe
} // namespace moonray

