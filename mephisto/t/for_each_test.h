#ifndef MEPHISTO_FOR_EACH_TEST_H
#define MEPHISTO_FOR_EACH_TEST_H

#include "gtest/gtest.h"
#include <libdash.h>
#include <mephisto/array>
#include <mephisto/algorithm/for_each>

int meph_argc;
char** meph_argv;

struct Pos {
    unsigned long x, y, z;
};

using Data = Pos;

class ForEachTest : public ::testing::Test {

public:
  static const int Dim = 3;
  using Data       = Pos;
  using PatternT   = dash::BlockPattern<Dim>;
  using MetaT      = typename mephisto::Metadata<PatternT>;
  using ViewT      = typename dash::Array<Data>::local_type;
  using AlpakaDim  = alpaka::dim::DimInt<1>;
  using ArrayT     = dash::Array<Data, dash::default_index_t, PatternT>;
  using SizeT      = ArrayT::size_type;

// Setup accelerator and host
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  using Acc       = alpaka::acc::AccGpuCudaRt<AlpakaDim, SizeT>;
  using QueueAcc  = alpaka::queue::QueueCudaRtSync;
#else
  using Acc       = alpaka::acc::AccCpuSerial<AlpakaDim, SizeT>;
  using QueueAcc  = alpaka::queue::QueueCpuSync;
#endif
  using Host = alpaka::acc::AccCpuSerial<AlpakaDim, SizeT>;

  using DevAcc   = alpaka::dev::Dev<Acc>;
  using DevHost  = alpaka::dev::Dev<Host>;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using PltfAcc  = alpaka::pltf::Pltf<DevAcc>;

  // The mephisto kernel to use in the executor
  using Kernel = mephisto::ForEachWithIndexKernel;

protected:
  virtual void SetUp()
  {
    dash::init(&meph_argc, &meph_argv);
  }

  virtual void TearDown()
  {
    dash::finalize();
  }
};

// This needs to be defined outside of TEST_F for nvcc.
struct ForEachClb {
  void operator()(Data &data, const mephisto::array<size_t, 3> coords)
  {
    data.x = coords[0];
    data.y = coords[1];
    data.z = coords[2];
  }
};

#endif
