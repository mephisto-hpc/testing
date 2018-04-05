#include <libdash.h>
#include <alpaka/alpaka.hpp>
#include <mephisto/algorithm/copy>
#include <mephisto/algorithm/for_each>
#include <mephisto/buffer>
#include <experimental/execution>
#include <experimental/thread_pool>
#include <mephisto/executor>
using std::experimental::static_thread_pool;

int main(int argc, char *argv[]) {
    using Data = float;
    using ArrT = dash::Array<Data>;
    using PatternT = typename dash::Array<Data>::pattern_type;
    using MetaT = typename mephisto::Metadata<PatternT>;
    using ViewT = typename dash::Array<Data>::local_type;
    using Dim = alpaka::dim::DimInt<1>;

    dash::init(&argc, &argv);

    // Setup a dash array and fill it
    auto arr = dash::Array<Data>(1000);
    dash::fill(arr.begin(), arr.end(), 5.0);

    arr.barrier();

    using Size = decltype(arr.size());

    // Setup accelerator and host
    using Acc       = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Host      = alpaka::acc::AccCpuSerial<Dim, Size>;
    using StreamAcc = alpaka::stream::StreamCpuSync;

    using DevAcc    = alpaka::dev::Dev<Acc>;
    using DevHost   = alpaka::dev::Dev<Host>;
    using PltfHost  = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc   = alpaka::pltf::Pltf<DevAcc>;

    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    StreamAcc stream(devAcc);

    // Context is a pair of host and accelerator used by the mpehisto::buffer
    auto ctx = mephisto::execution::make_context(devHost, devAcc, stream);

    // Get a local view to the array
    auto local_arr = arr.local;

    // Create the host buffer. It can be used to copy data to the accelerator
    // and returns a buffer on the device via getDeviceDataBuffer().
    /* auto buf = mephisto::HostDataBuffer< */
    /*   Data, */
    /*   decltype(ctx), */
    /*   PatternT, */
    /*   decltype(local_arr)>(ctx, local_arr); */

    /* // Get the device buffer that was copied in the line before. It contains */
    /* // the actual data plus metadata. */
    /* auto deviceBuf = buf.getDeviceDataBuffer(); */

    /* // Copy buf from the host to the device */
    /* mephisto::put(stream, buf); */

    // foreach
    /* mephisto::for_each(deviceBuf.begin(), deviceBuf.end(), [](const Data &data){ printf("Kernel [%f]\n", data); }); */


    auto executor = mephisto::execution::make_executor(ctx);
    auto policy   = mephisto::execution::make_parallel_policy(executor);
    dash::for_each(policy, arr.begin(), arr.end(), [](const Data &data) {
        printf("for_each: %f\n", data);
    });

    // copy result back to host
    /* mephisto::get(stream, buf); */

    /* size_t i = 0; */
    /* for(auto val : arr.local) { */
    /*     printf("[%d]: %f:%f\n", i, val, deviceBuf.begin()[i++]); */
    /* } */

    dash::finalize();
}


