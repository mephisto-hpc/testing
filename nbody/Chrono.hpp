#pragma once

#include <chrono>
#include <string>

struct Chrono
{
    Chrono()
    : last(std::chrono::high_resolution_clock::now())
    {}

    void printAndReset(const std::string& eventName)
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = end-last;
        std::cout << eventName << " " << (std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count() / 1000000.) << " s\n";
        last = end;
    }

    std::chrono::time_point< std::chrono::high_resolution_clock > last;
};
