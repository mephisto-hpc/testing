#include <iostream>
#include <chrono>
#include <algorithm>

#include <libdash.h>

#include <alpaka/alpaka.hpp>

#ifdef __CUDACC__
# define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#else
# define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
#endif
#include <llama/llama.hpp>

#include "ppm.hpp"

struct image_size
{
    size_t width, height;
};

struct rgb8
{
    uint8_t r, g, b;
};

struct rgb
{
    double r, g, b;
};

namespace st
{
    struct R {};
    struct G {};
    struct B {};
}

using RGB = llama::DS<
    llama::DE< st::R, double >,
    llama::DE< st::G, double >,
    llama::DE< st::B, double >
>;

using R = llama::DS<
    llama::DE< st::R, double >
>;

struct PassThrough
{
    using PrimType = unsigned char;
    using BlobType = PrimType*;
    using Parameter = BlobType;

    LLAMA_NO_HOST_ACC_WARNING
    static inline
    auto
    allocate(
        std::size_t count,
        Parameter const pointer
    )
    -> BlobType
    {
        return pointer;
    }
};


template<
    std::size_t blockSize,
    std::size_t threadCount,
    std::size_t elemCount
>
struct UpdateKernel
{
    template<
        typename T_Acc,
        typename T_View,
        typename T_Size
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        size_t problemSizeX,
        size_t problemSizeY,
        T_View view,
        T_Size coordX,
        T_Size coordY,
        T_Size blockLocalIdx,
        double factor
    ) const
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const startX = globalThreadIdx[0u] * elemCount;
        auto const   endX = alpaka::math::min(
            acc,
            startX + elemCount,
            problemSizeX
        );
        auto const startY = globalThreadIdx[1u] * elemCount;
        auto const   endY = alpaka::math::min(
            acc,
            startY + elemCount,
            problemSizeY
        );
#if 0
        std::cout << "#threads:     " << threadCount << "\n"
                  << "globalThread: (" << globalThreadIdx[0u] << "/" << globalThreadExtent[0u] << ", " << globalThreadIdx[1u] << "/" << globalThreadExtent[1u] << ")\n"
                  << "extent        [" << startX << ", " << endX << ")×[" << startY << ", " << endY << ") "
                  << std::endl;
#endif

        LLAMA_INDEPENDENT_DATA
        for (auto x = startX; x < endX; ++x) {

            LLAMA_INDEPENDENT_DATA
            for (auto y = startY; y < endY; ++y) {
                const size_t local_index = x * problemSizeY + y;
#if 0
                std::cout << "globalThread: (" << globalThreadIdx[0u] << "/" << globalThreadExtent[0u] << ", " << globalThreadIdx[1u] << "/" << globalThreadExtent[1u] << ")\n"
                          << "coord         (" << x << ", " << y << ") => " << local_index << " "
                          << std::endl;
#endif
                view(blockLocalIdx + local_index) *= (coordX + x) * factor;
            }
        }
    }
};

int
main(int ac, char* av[])
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt< 2 >;
    using Size = std::size_t;
    using Extents = Size;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

#if 1
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
#else
    using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
#endif
    using Queue = alpaka::queue::QueueCpuSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    // LLAMA
    using UD = llama::UserDomain<1>;

    dash::init(&ac, &av);

    dart_unit_t myid = dash::myid();

    dash::Shared<image_size> size_shared;

    /* Load image header by unit 0 and broadcast image dimension to all units */

    ppm::Reader* image = 0;
    if (myid == 0) {
        image = new ppm::Reader(av[1]);
        size_shared.set({image->getWidth(), image->getHeight()});
    }
    size_shared.barrier();

    image_size size = size_shared.get();

    /* Allocate shared matrix for raw 3 byte image data */

    auto distspec = dash::DistributionSpec<2>(dash::BLOCKED, dash::NONE);
    dash::NArray<rgb8, 2> matrix8(
        dash::SizeSpec<2>(size.width, size.height),
        distspec,
        dash::Team::All());

    /* Load image by unit 0 and scatter lines to the remote memories */

    if (myid == 0) {
        size_t lines_per_chunk = (1024 * 1024) / size.width;
        if (lines_per_chunk == 0)
            lines_per_chunk = 1;
        size_t nr_chunks = (size.height + lines_per_chunk - 1) / lines_per_chunk;
        auto iter= matrix8.begin();
        std::vector<uint8_t> buf;
        for (size_t i = 0; i < nr_chunks; i++) {
            image->read(buf, lines_per_chunk * size.width);
            rgb8* rgb8_begin = (rgb8*)buf.data();
            rgb8* rgb8_end = (rgb8*)(buf.data() + buf.size());
            iter = dash::copy(rgb8_begin, rgb8_end, iter);
        }
        buf.resize(0);
        delete image;
    }
    matrix8.barrier();

    /* Allocate shared matrix for double image data */

    dash::NArray<rgb, 2> matrix(
        dash::SizeSpec<2>(size.width, size.height),
        distspec,
        dash::Team::All());

    /* Locally transform image data from byte to double pixels */

    matrix8.barrier();
    auto tpStart(std::chrono::high_resolution_clock::now());

    std::transform(matrix8.lbegin(), matrix8.lend(), matrix.lbegin(), [](const rgb8& pixel) -> rgb {
        return { pixel.r / 255., pixel.g / 255., pixel.b / 255. };
    });
    auto durElapsed(std::chrono::high_resolution_clock::now() - tpStart);
    std::cout << myid << ": RGB(char) -> RGB(double): " << std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() / 1000000. << std::endl;
    matrix8.barrier();

    matrix8.deallocate();

    /* Allocate space for a SoA representation of the image data */

    dash::NArray<double, 2> matrix_soa(
        dash::SizeSpec<2>(size.width, size.height),
        distspec,
        dash::Team::All());

    /* The LLAMA user domain has the size of the local matrix part */

    UD local_matrix_dim{ matrix_soa.local.size() };

    using MappingAoS = llama::mapping::AoS<
        UD,
        RGB,
        llama::LinearizeUserDomainAdress<UD::count>
    >;

    using MappingSoA = llama::mapping::SoA<
        UD,
        R,
        llama::LinearizeUserDomainAdress<UD::count>
    >;

    /* View on the local AoS memory for the local image part */

    MappingAoS mapping_aos(local_matrix_dim);
    using FactoryAoS = llama::Factory<
        MappingAoS,
        PassThrough
    >;
    auto view_aos = FactoryAoS::allocView(mapping_aos, reinterpret_cast<PassThrough::BlobType>(matrix.lbegin()));

    /* View on the local SoA memory for the local image part */

    MappingSoA mapping_soa(local_matrix_dim);
    using FactorySoA = llama::Factory<
        MappingSoA,
        PassThrough
    >;
    auto view_soa = FactorySoA::allocView(mapping_soa, reinterpret_cast<PassThrough::BlobType>(matrix_soa.lbegin()));

    /* Pointwise transformation of the local part from AoS to SoA */

    matrix_soa.barrier();
    tpStart = std::chrono::high_resolution_clock::now();

    LLAMA_INDEPENDENT_DATA
    for (size_t i = 0; i < local_matrix_dim[0]; ++i) {
        view_soa(i) = view_aos(i);
    }

    durElapsed = std::chrono::high_resolution_clock::now() - tpStart;
    std::cout << myid << ": AoS(R) -> SoA(R): " << std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() / 1000000. << std::endl;
    matrix_soa.barrier();

    // Adjust R based on the global coordinate

    //durElapsed = 0;

    // copy local buffer to device

    auto lblocks = matrix.pattern().local_blockspec().size();
    //std::cout << myid << ": " << lblocks << std::endl;
    for (size_t lblock_idx = 0; lblock_idx < lblocks; lblock_idx++) {
        auto lblock_view = matrix.pattern().local_block_local(lblock_idx);
        auto local_index = matrix.pattern().local_at({0,0}, lblock_view);
        auto global_index = matrix.pattern().global(local_index);
        auto global_coords = matrix.pattern().coords(global_index);

        constexpr std::size_t blockSize = 3;
#if 1
        constexpr Size elemCount   = blockSize;
        constexpr Size threadCount = 1;
#elif 1 // Omp2Threads
        constexpr Size elemCount   = (blockSize + 1) / 2;
        constexpr Size threadCount = 2;
#else // CUDA
        constexpr Size elemCount   = 1;
        constexpr Size threadCount = blockSize;
#endif

        const alpaka::vec::Vec< Dim, Size > elems{elemCount, elemCount};
        const alpaka::vec::Vec< Dim, Size > threads{threadCount,  threadCount};
        constexpr auto innerCount = elemCount * threadCount;
        const alpaka::vec::Vec< Dim, Size > blocks{
            Size(lblock_view.extent(0) + innerCount - 1) / innerCount,
            Size(lblock_view.extent(1) + innerCount - 1) / innerCount
        };

        auto const workdiv = alpaka::workdiv::WorkDivMembers<
            Dim,
            Size
        > {
            blocks,
            threads,
            elems
        };

#if 1

        std::cout << myid << ": problem:  " << lblock_view.extent(0) << "×" << lblock_view.extent(1) << std::endl;
        std::cout << myid << ": blocks:   " << blocks[0u] << "×" << blocks[1u] << std::endl;

        tpStart = std::chrono::high_resolution_clock::now();
        // start 2d kernel, pass: global_coords, local_index, size.width, block.extents, view_soa
        UpdateKernel<
            blockSize,
            threadCount,
            elemCount
        > kernel;
        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            kernel,
            lblock_view.extent(0),
            lblock_view.extent(1),
            view_soa,
            global_coords[0],
            global_coords[1],
            local_index,
            1. / size.width
        );

        durElapsed = std::chrono::high_resolution_clock::now() - tpStart;

#else

        LLAMA_INDEPENDENT_DATA
        for (size_t x = 0; x < lblock_view.extent(0); ++x) {

            LLAMA_INDEPENDENT_DATA
            for (size_t y = 0; y < lblock_view.extent(1); ++y) {
                size_t my_global_x = global_coords[0] + x;
                size_t my_global_y = global_coords[1] + y;

                size_t linear_thread = x * lblock_view.extent(1) + y;

                //std::cout << myid << ": " << local_index << ": " << x << ", " << y << " - " << my_global_x << ", " << my_global_y << std::endl;
                view_soa.accessor<st::R>({ local_index + linear_thread }) *= my_global_x / (double)size.width;
            }
        }

#endif
    }

    // copy local buffer from device

    std::cout << myid << ": adjust SoA(R): " << std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() / 1000000. << std::endl;
    matrix_soa.barrier();

    /* Pointwise transformation back of the local part from SoA to AoS */

    matrix_soa.barrier();
    tpStart = std::chrono::high_resolution_clock::now();

    LLAMA_INDEPENDENT_DATA
    for (size_t i = 0; i < local_matrix_dim[0]; ++i) {
        view_aos(i) = view_soa(i);
    }

    durElapsed = std::chrono::high_resolution_clock::now() - tpStart;
    std::cout << myid << ": SoA(R) -> AoS(R): " << std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() / 1000000. << std::endl;
    matrix_soa.barrier();

    matrix_soa.deallocate();

    matrix8.allocate(
        dash::SizeSpec<2>(size.width, size.height),
        distspec,
        dash::Team::All());

    matrix.barrier();

    tpStart = std::chrono::high_resolution_clock::now();

    std::transform(matrix.lbegin(), matrix.lend(), matrix8.lbegin(), [](const rgb& pixel) -> rgb8 {
        return { (uint8_t)(pixel.r * 255.), (uint8_t)(pixel.g * 255.), (uint8_t)(pixel.b * 255.) };
    });
    durElapsed = std::chrono::high_resolution_clock::now() - tpStart;
    std::cout << myid << ": RGB(double) -> RGB(char): " << std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() / 1000000. << std::endl;

    matrix8.barrier();

    matrix.deallocate();

    if (myid == 0) {
        std::string out_name(av[1]);
        std::string suffix("_out");
        auto ext = out_name.find_last_of(".");
        if (ext == std::string::npos) {
            ext = out_name.length();
        }
        out_name.insert(ext, suffix);
        ppm::Writer image(out_name, size.width, size.height);
        std::cout << out_name << std::endl;

        size_t lines_per_chunk = (1024 * 1024) / size.width;
        if (lines_per_chunk == 0)
            lines_per_chunk = 1;
        std::vector<uint8_t> buf;
        buf.resize(lines_per_chunk * size.width * 3);
        size_t nlines = size.height;
        auto iter = matrix8.begin();
        while (nlines > 0) {
            size_t chunk_lines = lines_per_chunk;
            if (nlines < lines_per_chunk) {
                chunk_lines = nlines;
                buf.resize(chunk_lines * size.width * 3);
            }
            nlines -= chunk_lines;
            chunk_lines *= size.width;

            rgb8* rgb8_begin = (rgb8*)buf.data();
            dash::copy(iter, iter + chunk_lines, rgb8_begin);
            iter += chunk_lines;
            image.write(buf);
        }
    }
    matrix8.barrier();

    dash::finalize();

    return 0;
}
