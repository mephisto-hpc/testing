#include <libdash.h>
#include <mephisto/algorithm/for_each>
#include <mephisto/execution>
#include <mephisto/array>
#include <alpaka/alpaka.hpp>


struct Pos {
    unsigned long x, y, z;
};

int main(int argc, char *argv[]) {
  auto const Dim  = 3;
  using Data      = Pos;
  using PatternT  = dash::BlockPattern<Dim>;
  using MetaT     = typename mephisto::Metadata<PatternT>;
  using ViewT     = typename dash::Array<Data>::local_type;
  using AlpakaDim = alpaka::dim::DimInt<1>;
  using ArrayT    = dash::Array<Data, dash::default_index_t, PatternT>;
  using SizeT     = ArrayT::size_type;

  dash::init(&argc, &argv);

  PatternT pattern {10, 10, 10};
  ArrayT arr {pattern};
  dash::fill(arr.begin(), arr.end(), {42, 42, 42});


    // Setup accelerator and host
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc       = alpaka::acc::AccGpuCudaRt<AlpakaDim, SizeT>;
    using StreamAcc = alpaka::stream::StreamCudaRtSync;
#else
    using Acc       = alpaka::acc::AccCpuSerial<AlpakaDim, SizeT>;
    using StreamAcc = alpaka::stream::StreamCpuSync;
#endif
    using Host      = alpaka::acc::AccCpuSerial<AlpakaDim, SizeT>;

    using DevAcc    = alpaka::dev::Dev<Acc>;
    using DevHost   = alpaka::dev::Dev<Host>;
    using PltfHost  = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc   = alpaka::pltf::Pltf<DevAcc>;

    // The mephisto kernel to use in the executor
    using Kernel = mephisto::ForEachWithIndexKernel;

    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    StreamAcc stream(devAcc);

    // Setup of the executor:
    //
    // Context consists of the host, the accelerator and the stream
    auto ctx = mephisto::execution::make_context<Acc>(devHost, devAcc, stream);

    // The executor is the one actually doing the computation
    mephisto::execution::AlpakaTwoWayExecutor<Kernel, decltype(ctx)> executor(ctx);

    // The policy is used to relax guarantees.
    auto policy = mephisto::execution::make_parallel_policy(executor);

    // set the coordinates using an Alpaka policy
    dash::for_each_with_index(
        policy,
        arr.begin(),
        arr.end(),
        [] ALPAKA_FN_ACC (
            Data &data, const mephisto::array<SizeT, Dim> coords) {
          data.x = coords[0];
          data.y = coords[1];
          data.z = coords[2];
        });

    // Check the written coordinates using the standard for_each_with_index
    dash::for_each_with_index(arr.begin(), arr.end(), [&pattern](const Data &d, PatternT::index_type i) {
        auto const coords = pattern.coords(i);
        printf("mephisto: [%lu,%lu,%lu] -  dash: [%lu,%lu,%lu]\n", d.x, d.y, d.z, coords[0], coords[1], coords[2]);
    });
    dash::finalize();
}



