#include <mephisto/array>
#include <mephisto/algorithm/for_each>
// TODO: move from detail namespace
#include <mephisto/detail/meta>
#include <libdash.h>

int main(int argc, char *argv[]) {
    using Data = float; 
    using ArrT = dash::Array<Data>;
    using PatternT = typename dash::Array<Data>::pattern_type;
    using MetaT = typename mephisto::detail::Metadata<PatternT>;
    using ViewT = typename dash::Array<Data>::local_type;
    using Dim = alpaka::dim::DimInt<1>;

    dash::init(&argc, &argv);
 
    auto arr = dash::Array<Data>(1000);

    using Size = decltype(arr.size());

    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using StreamAcc = alpaka::stream::StreamCpuSync;

    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    arr | dash::local();

    /* auto buf = mephisto::detail::DataBuffer<Data, DevAcc, PatternT>(devAcc, arr | dash::local()...); */
}
