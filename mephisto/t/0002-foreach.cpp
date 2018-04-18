#include <libdash.h>
#include <mephisto/algorithm/for_each>
#include <mephisto/execution>
#include <alpaka/alpaka.hpp>
#include <mephisto/algorithm/for_each>

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


    // Setup of the executor:
    //
    // Context is a pair of host and accelerator used by the mephisto::buffer
    auto ctx = mephisto::execution::make_context<Acc>(devHost, devAcc, stream);

    // The executor is the one actually doing the computation
    auto executor = mephisto::execution::make_executor(ctx, arr.pattern());
    // The policy is used to relax guarantees
    auto policy   = mephisto::execution::make_parallel_policy(executor);

    dash::for_each(policy, arr.begin(), arr.end(), [](const Data &data) {
    //             ^^^^^^ The policy is the only additional
    //                    parameter compared to a usual for_each call.
        printf("for_each: %f\n", data);
    });

    dash::finalize();
}


