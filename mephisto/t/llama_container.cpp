#include <iostream>
#include <vector>

#include <llama/llama.hpp>

#include <mephisto/container>

namespace st
{
    struct R {};
    struct G {};
    struct B {};
    struct V {};
}

int
main()
{
    using RGB = llama::DS<
        llama::DE< st::R, double >,
        llama::DE< st::G, double >,
        llama::DE< st::B, double >
    >;

    using V = llama::DS<
        llama::DE< st::V, int >
    >;

    auto v0 = mephisto::container::llama_factory<
        mephisto::container::factory::std_vector,
        RGB,
        llama::mapping::AoS
    >::create();

    auto v1 = mephisto::container::llama_factory<
        mephisto::container::factory::std_vector,
        RGB,
        llama::mapping::SoA
    >::create(16 * 16);

    auto v2 = mephisto::container::llama_factory<
        mephisto::container::factory::std_array<5>,
        V,
        llama::mapping::AoS
    >::create();

    return 0;
}
