#ifndef PPM_HPP_INCLUDED
#define PPM_HPP_INCLUDED

#include <exception>
#include <fstream>
#include <sstream>
#include <string>

namespace ppm {

class Reader
{
    std::ifstream ifs;
    size_t width, height;

    void open()
    {
      using namespace std;

      // Read header.
      auto const expected_magic_number = string("P6");
      auto const expected_max_value = string("255");
      auto magic_number = string();
      auto max_value = string();
      ifs >> magic_number;

      if (!ifs || magic_number != expected_magic_number) {
        auto ss = stringstream();
        ss << "magic number must be '" << expected_magic_number << "'";
        throw runtime_error(ss.str());
      }

      int components = 0;
      while (components < 3 && ifs) {
        char c;
        ifs >> c;
        if (c == '#') {
          while (c != '\n' && ifs)
            ifs.get(c);
          continue;
        }
        ifs.unget();

        switch (components) {
          case 0: ifs >> width; break;
          case 1: ifs >> height; break;
          case 2: ifs >> max_value; break;
        }
        components++;
      }

      if (!ifs || width == 0 || height == 0) {
        auto ss = stringstream();
        ss << "invalid size: " << width << "×" << height;
        throw runtime_error(ss.str());
      }

      if (!ifs || max_value != expected_max_value) {
        auto ss = stringstream();
        ss << "max value must be " << expected_max_value;
        throw runtime_error(ss.str());
      }

      // Skip ahead (an arbitrary number!) to the pixel data.
      ifs.ignore(256, '\n');
    }

public:
    Reader(const std::string& filename)
    {
      using namespace std;

      ifs.open(filename, std::ios::binary);
      if (ifs.fail()) {
        auto ss = stringstream();
        ss << "cannot open file '" << filename << "' for reading";
        throw runtime_error(ss.str());
      }
      open();
    }

    ~Reader()
    {
      ifs.close();
    }

    template<
        typename TBuffer>
    void
    read(TBuffer& pixel_data, size_t max_pixels)
    {
      // Read pixel data.
      pixel_data.resize(max_pixels * 3);
      ifs.read(reinterpret_cast<char*>(pixel_data.data()), pixel_data.size());
      pixel_data.resize(ifs.gcount());
    }

    size_t getWidth() const
    {
        return width;
    }

    size_t getHeight() const
    {
        return height;
    }
};

class Writer
{
    std::ofstream ofs;
    size_t width, height;

public:
    Writer(const std::string& filename, size_t width, size_t height)
    {
      using namespace std;

      ofs.open(filename, std::ios::binary);
      if (ofs.fail()) {
        auto ss = stringstream();
        ss << "cannot open file '" << filename << "' for writing";
        throw runtime_error(ss.str());
      }
      ofs << "P6\n" << width << " " << height << "\n255\n";
    }

    ~Writer()
    {
      ofs.close();
    }

    template<
        typename TBuffer>
    void
    write(TBuffer& pixel_data)
    {
      // Read pixel data.
      ofs.write(reinterpret_cast<char*>(pixel_data.data()), pixel_data.size());
    }
};

} // namespace ppm

#endif // PPM_HPP_INCLUDED
