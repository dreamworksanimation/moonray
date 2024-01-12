// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <forward_list>
#include <stdexcept>
#include <string>
#include <sstream>

#include <iostream>

class ArgumentException : public std::runtime_error
{
public:
    ArgumentException(const std::string& e) : std::runtime_error(e) {}
};

template <typename To, typename From>
To lexicalCast(const From& f)
{
    To ret;

    std::stringstream ss;
    ss.exceptions(std::stringstream::failbit);
    ss << f;
    ss >> ret;
    return ret;
}

class ArgumentParser
{
    typedef std::forward_list<std::string> Container;
    typedef Container::iterator iterator;
    typedef Container::const_iterator const_iterator;

public:
    ArgumentParser(int argc, const char* argv[])
    {
        for (int i = argc - 1; i > 0; --i) {
            mArgs.emplace_front(argv[i]);
        }
    }

    void print() const
    {
        for (const auto& t : mArgs) {
            std::cout << t << '\n';
        }
    }

    bool has(const std::string& key) const
    {
        return value(key) != mArgs.end();
    }

    template <typename T>
    void set(const std::string& key, const T& value)
    {
        auto v = value(key);
        *v = lexicalCast<std::string>(value);
    }

    template <typename T>
    T getModifier(const std::string& key, std::size_t mod) const
    {
        auto it = value(key);
        for (std::size_t i = 0; i <= mod; ++i, ++it) {
            if (it == mArgs.end()) {
                throw ArgumentException("Positional modifier to " + key + " not found");
            } 
        }
        if (it == mArgs.end()) {
            throw ArgumentException("Positional modifier to " + key + " not found");
        } 
        try {
            return lexicalCast<T>(*it);
        } catch (...) {
            throw ArgumentException("Unable to convert modifier to specified type.\n");
        }
    }

private:
    iterator value(const std::string& key)
    {
        for (auto it = mArgs.begin(); it != mArgs.end(); ++it) {
            if (*it == key) {
                return it;
            }
        }
        return mArgs.end();
    }

    const_iterator value(const std::string& key) const
    {
        for (auto it = mArgs.cbegin(); it != mArgs.cend(); ++it) {
            if (*it == key) {
                return it;
            }
        }
        return mArgs.cend();
    }

    Container mArgs;
    //Container mDefaults;
};

