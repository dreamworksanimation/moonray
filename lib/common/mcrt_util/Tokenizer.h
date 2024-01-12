// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

namespace moonray {
namespace util {

inline constexpr std::size_t length(const char* c)
{
    return *c == '\0' ? 0 : length(c + 1) + 1;
}

inline std::size_t length(const std::string& c)
{
    return c.length();
}

class Tokenizer
{
public:
    using value_type = std::string;

private:
    using container = std::vector<value_type>;

public:
    using iterator        = container::iterator;
    using const_iterator  = container::const_iterator;
    using reference       =       value_type&;
    using const_reference = const value_type&;
    using pointer         =       value_type*;
    using const_pointer   = const value_type*;

    explicit Tokenizer(const std::string& s)
    : mTokens(tokenize(s, [] (char c) { return isToken(c, sWhiteSpace, length(sWhiteSpace)); }))
    {
    }

    Tokenizer(const std::string& s, const std::string& tokens)
    : mTokens(tokenize(s, [&tokens] (char c) { return isToken(c, tokens.c_str(), length(tokens)); }))
    {
    }

    std::size_t size() const noexcept
    {
        return mTokens.size();
    }

    bool empty() const noexcept
    {
        return mTokens.empty();
    }

    reference operator[](std::size_t idx) noexcept
    {
        assert(idx < size());
        return mTokens[idx];
    }

    const_reference operator[](std::size_t idx) const noexcept
    {
        assert(idx < size());
        return mTokens[idx];
    }

    iterator begin()
    {
        return mTokens.begin();
    }

    iterator end()
    {
        return mTokens.end();
    }

    const_iterator begin() const
    {
        return mTokens.begin();
    }

    const_iterator end() const
    {
        return mTokens.end();
    }

    const_iterator cbegin() const
    {
        return mTokens.begin();
    }

    const_iterator cend() const
    {
        return mTokens.end();
    }
private:
    // This function is very c-like, because there are times when we can compute the length at compile-time (e.g. a
    // string literal). This ensures we can do so, instead of, say, passing in a template type and computing the length
    // here.
    static bool isToken(char c, const char* const tokens, std::size_t tokensLength)
    {
        for (std::size_t i = 0; i < tokensLength; ++i) {
            if (c == tokens[i]) {
                return true;
            }
        }
        return false;
    }

    template <typename TokenRecognizer>
    static std::vector<std::string> tokenize(const std::string& s, TokenRecognizer tokenRecognizer)
    {
        // Speaking in terms of whitespace...
        // first should point to the next non-whitespace character
        // last should point to the following first whitespace character

        std::vector<std::string> ret;
        using const_iterator = std::string::const_iterator;
        const_iterator first = std::find_if_not(s.cbegin(), s.cend(), tokenRecognizer);
        while (first != s.cend()) {
            const_iterator last = std::find_if(first, s.cend(), tokenRecognizer);
            ret.emplace_back(first, last);
            first = std::find_if_not(last, s.cend(), tokenRecognizer);
        }
        return ret;
    }

    static constexpr const char* const sWhiteSpace = " \t\n\r";
    container mTokens;
};

} // namespace util
} // namespace moonray

