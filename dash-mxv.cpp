#include <unistd.h>
#include <iostream>
#include <cstddef>
#include <iomanip>

#include <libdash.h>

using std::cout;
using std::endl;
using std::setw;

int main(int argc, char* argv[])
{
  dash::init(&argc, &argv);

  dart_unit_t myid   = dash::myid();
  size_t num_units   = dash::Team::All().size();


  dash::finalize();
}
