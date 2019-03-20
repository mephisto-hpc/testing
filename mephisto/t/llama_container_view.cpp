#include <iostream>
#include <vector>

#include <libdash.h>

#include <mephisto/container>
#include <mephisto/views>

namespace st
{
    struct R {};
    struct G {};
    struct B {};
    struct V {};
}

int
main(int ac, char* av[])
{
    using RGB = llama::DS<
        llama::DE< st::R, double >,
        llama::DE< st::G, double >,
        llama::DE< st::B, double >
    >;

    auto v0 = mephisto::container::llama_factory<
        mephisto::container::factory::std_vector,
        RGB,
        llama::mapping::AoS
    >::create(100);

    auto v0_view = mephisto::view::llama_view::create_host_view(v0);

    auto a0 = mephisto::container::llama_factory<
        mephisto::container::factory::std_array<5>,
        RGB,
        llama::mapping::SoA
    >::create();

    auto a0_view = mephisto::view::llama_view::create_host_view(a0);

    dash::init(&ac, &av);

    auto a1 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_Array,
        RGB,
        llama::mapping::SoA
    >::create(dash::size() * 100);

    auto a1_view = mephisto::view::llama_view::create_host_view(a1);

    auto m0 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_Matrix<2>,
        RGB,
        llama::mapping::AoS
    >::create(dash::size() * 10, dash::size() * 10);

    dash::finalize();

    return 0;
}
