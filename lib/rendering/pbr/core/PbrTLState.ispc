// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "PbrTLState.isph"

// hooks into c++, must match declarations in TLState.cc
extern "C" void * uniform CPP_PbrTLState_poolAlloc(uniform PbrTLState * uniform pbrTls,
                                                   uniform unsigned int nybtes);
extern "C" void CPP_PbrTLState_poolFree(uniform PbrTLState * uniform pbrTls, void * uniform ptr);
extern "C" uniform uint32_t CPP_PBR_ptrToIndex(void * uniform ptr);
extern "C" void * uniform CPP_PBR_indexToPtr(uniform uint32_t index);

extern "C" uniform uint32_t
CPP_PbrTLState_allocList(uniform PbrTLState * uniform pbrTls,
                         uniform unsigned int itemSize,
                         uniform unsigned int numItems);

extern "C" void
CPP_PbrTLState_freeList(uniform PbrTLState * uniform pbrTls, uniform uint32_t listPtr);

extern "C" uniform unsigned int
CPP_PbrTLState_getNumListItems(uniform PbrTLState * uniform pbrTls, uniform uint32_t listPtr);

extern "C" void * uniform
CPP_PbrTLState_getListItem(uniform PbrTLState * uniform pbrTls,
                           uniform uint32_t listPtr,
                           uniform unsigned int item);

extern "C" void
CPP_PbrTLState_acquireDeepData(uniform PbrTLState * uniform pbrTls, uniform uint32_t deepPtr);

extern "C" void
CPP_PbrTLState_releaseDeepData(uniform PbrTLState * uniform pbrTls, uniform uint32_t deepPtr);

extern "C" void
CPP_PbrTLState_acquireCryptomatteData(uniform PbrTLState * uniform pbrTls, uniform uint32_t cryptomattePtr);

extern "C" void
CPP_PbrTLState_releaseCryptomatteData(uniform PbrTLState * uniform pbrTls, uniform uint32_t cryptomattePtr);

extern "C" uniform bool isRenderCanceled();

uniform bool
isCanceled(uniform PbrTLState * uniform pbrTls)
{
    if (pbrTls->mCancellationState <= CANCELLATION_STATE_CANCELED) {
        return pbrTls->mCancellationState != CANCELLATION_STATE_DISABLED;
    }

    switch (pbrTls->mCancellationState) {

    case CANCELLATION_STATE_WAITING_FOR_SAMPLE_PER_PIXEL:
        if (pbrTls->mCurrentPassIdx > 0) {
            pbrTls->mCancellationState = CANCELLATION_STATE_WAITING_FOR_CANCEL;
            if (isRenderCanceled()) {
                pbrTls->mCancellationState = CANCELLATION_STATE_CANCELED;
                return true;
            }
        }
        break;

    case CANCELLATION_STATE_WAITING_FOR_CANCEL:
        if (isRenderCanceled()) {
            pbrTls->mCancellationState = CANCELLATION_STATE_CANCELED;
            return true;
        }
        break;

    default:
        MNRY_ASSERT(0);
    }

    return false;
}

varying uint32_t
PbrTLState_allocList(uniform PbrTLState * uniform pbrTls,
                     uniform unsigned int itemSize,
                     varying unsigned int numItems)
{
    varying uint32_t result = PBR_nullHandle;
    foreach_active(lane) {
        const uniform unsigned int uNumItems = extract(numItems, lane);
        uniform uint32_t listPtr = CPP_PbrTLState_allocList(pbrTls, itemSize, uNumItems);
        result = insert(result, lane, listPtr);
    }
    return result;
}


void
PbrTLState_freeList(uniform PbrTLState * uniform pbrTls,
                    varying uint32_t listPtr)
{
    foreach_active(lane) {
        uniform uint32_t uListPtr = extract(listPtr, lane);
        CPP_PbrTLState_freeList(pbrTls, uListPtr);
    }
}

varying unsigned int
PbrTLState_getNumListItems(uniform PbrTLState * uniform pbrTls,
                           varying uint32_t listPtr)
{
    varying unsigned int numItems = 0;
    foreach_active(lane) {
        uniform unsigned int uNumItems = CPP_PbrTLState_getNumListItems(pbrTls, extract(listPtr, lane));
        numItems = insert(numItems, lane, uNumItems);
    }
    return numItems;
}

void * varying
PbrTLState_getListItem(uniform PbrTLState * uniform pbrTls,
                       varying uint32_t listPtr,
                       varying unsigned int item)
{
    varying intptr_t result = 0;
    foreach_active(lane) {
        uniform uint32_t uListPtr = extract(listPtr, lane);
        uniform unsigned int uItem = extract(item, lane);
        uniform intptr_t ptr = (uniform intptr_t) CPP_PbrTLState_getListItem(pbrTls, uListPtr, uItem);
        result = insert(result, lane, ptr);
    }
    return (void * varying) result;
}

void
PbrTLState_acquireDeepData(uniform PbrTLState * uniform pbrTls,
                           varying uint32_t deepPtr)
{
    foreach_active(lane) {
        uniform uint32_t uDeepPtr = extract(deepPtr, lane);
        CPP_PbrTLState_acquireDeepData(pbrTls, uDeepPtr);
    }
}

void
PbrTLState_releaseDeepData(uniform PbrTLState * uniform pbrTls,
                           varying uint32_t deepPtr)
{
    foreach_active(lane) {
        uniform uint32_t uDeepPtr = extract(deepPtr, lane);
        CPP_PbrTLState_releaseDeepData(pbrTls, uDeepPtr);
    }
}

void
PbrTLState_acquireCryptomatteData(uniform PbrTLState * uniform pbrTls,
                                  varying uint32_t cryptomattePtr)
{
    foreach_active(lane) {
        uniform uint32_t uCryptomattePtr = extract(cryptomattePtr, lane);
        CPP_PbrTLState_acquireCryptomatteData(pbrTls, uCryptomattePtr);
    }
}

void
PbrTLState_releaseCryptomatteData(uniform PbrTLState * uniform pbrTls,
                                  varying uint32_t cryptomattePtr)
{
    foreach_active(lane) {
        uniform uint32_t uCryptomattePtr = extract(cryptomattePtr, lane);
        CPP_PbrTLState_releaseCryptomatteData(pbrTls, uCryptomattePtr);
    }
}

export uniform uint32_t
PbrTLState_hudValidation(uniform bool verbose)
{
    PBR_TL_STATE_VALIDATION;
}


