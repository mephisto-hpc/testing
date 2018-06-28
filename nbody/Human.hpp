#pragma once

#include <iostream>
#include <sstream>

struct human_readable
{
    uint64_t value;
public:
    human_readable() : value(0) {}
    human_readable(uint64_t s) : value(s) {}
    human_readable(const human_readable& o) : value(o.value) {}
    friend std::ostream& operator<<(std::ostream& out, const human_readable& m);
};

std::ostream& operator<<(std::ostream& out, const human_readable& m)
{
    std::stringstream res;

    if (m.value < 1024) {
        res << m.value;
    } else {
        const char* suffix = "KMGTPE";

        uint64_t v = m.value;
        uint64_t r = 0;
        while (v / 1024 / 1000 && suffix[1]) {
            v = v / 1024;
            suffix++;
        }
        res << std::setw(3) << v / 1000 << "." << std::setw(3) << std::setfill('0') << v % 1000 << std::setfill(' ') << " " << suffix[0] << "iB";
    }

    out << res.str();

    return out;
}
