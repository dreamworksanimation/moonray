// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once


// This is a way to mimic Python's finally construct, in which code is executed regardless of whether an exception was
// caught or not.
template <typename F>
class Finally
{
public:
    explicit Finally(F f)
    : m_f(std::move(f))
    {
    }

    ~Finally() noexcept
    {
        m_f();
    }

private:
    F m_f;
};

template <typename F>
Finally<F> finally(F f)
{
    return Finally<F>(std::move(f));
}

