#include <iostream>
#include <vector>

#include <llama/llama.hpp>
#include <libdash.h>

#include <mephisto/container>

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

    auto a0 = mephisto::container::llama_factory<
        mephisto::container::factory::std_array<5>,
        V,
        llama::mapping::AoS
    >::create();

    dash::init(&ac, &av);

    auto a1 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_Array,
        RGB,
        llama::mapping::SoA
    >::create();

    a1.allocate(100);

    auto a2 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_Array,
        RGB,
        llama::mapping::AoS
    >::create(100);

    auto m0 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_Matrix<2>,
        RGB,
        llama::mapping::AoS
    >::create();

    auto m1 = mephisto::container::llama_factory<
        mephisto::container::factory::dash_NArray<3>,
        RGB,
        llama::mapping::SoA
    >::create();

    using block_pattern = dash::BlockPattern<2>;
    using block_matrix_factory =
        mephisto::container::factory::dash_Matrix<
            block_pattern::ndim(),
            typename block_pattern::index_type,
            block_pattern>;

    auto m2 = mephisto::container::llama_factory<
        block_matrix_factory,
        RGB,
        llama::mapping::SoA
    >::create();

    dash::finalize();

    return 0;
}
