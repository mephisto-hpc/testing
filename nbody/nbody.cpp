/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file nbody.cpp
 *  \brief Realistic nbody example for using LLAMA and ALPAKA together.
 */

#include <iostream>
#include <utility>

#include <libdash.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define NBODY_CUDA 1
#else
/// defines whether cuda shall be used
#define NBODY_CUDA 0
#endif

/// total number of particles
#define NBODY_PROBLEM_SIZE 16*1024
/// number of elements per block
#define NBODY_BLOCK_SIZE 256
/// number of steps to calculate
#define NBODY_STEPS 5


#if BOOST_VERSION < 106700 && (__CUDACC__ || __IBMCPP__)
    #ifdef BOOST_PP_VARIADICS
        #undef BOOST_PP_VARIADICS
    #endif
    #define BOOST_PP_VARIADICS 1
#endif

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#endif
#include <llama/llama.hpp>
#include <random>

#include <AlpakaAllocator.hpp>
#include <AlpakaMemCopy.hpp>
#include <AlpakaThreadElemsDistribution.hpp>

#include "HRChrono.hpp"
#include "Dummy.hpp"
#include "Human.hpp"

namespace nbody
{

using Element = float;
constexpr Element EPS2 = 0.01;

namespace dd
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Mass {};
}

struct particle
{
    struct
    {
        Element x, y, z;
    } pos;
    struct
    {
        Element x, y, z;
    } vel;
    Element mass;
};

using Particle = llama::DS<
    llama::DE< dd::Pos, llama::DS<
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::Vel,llama::DS<
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::Mass, Element >
>;

/** Helper function for particle particle interaction. Gets two virtual datums
 *  like they are real particle objects
 */
template<
    typename T_VirtualDatum1,
    typename T_VirtualDatum2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_VirtualDatum1 && p1,
    T_VirtualDatum2 && p2,
    Element const & ts
)
-> void
{
    // Creating tempory virtual datum object for distance on stack:
    auto distance = p1( dd::Pos( ) ) + p2( dd::Pos( ) );
    distance *= distance; //square for each element
    Element distSqr = EPS2 +
        distance( dd::X( ) ) +
        distance( dd::Y( ) ) +
        distance( dd::Z( ) );
    Element distSixth = distSqr * distSqr * distSqr;
    Element invDistCube = 1.0f / sqrtf( distSixth );
    Element s = p2( dd::Mass( ) ) * invDistCube;
    distance *= s * ts;
    p1( dd::Vel( ) ) += distance;
}

/** Implements an allocator for LLAMA using the ALPAKA shared memory or just
 *  local stack memory depending on the number of threads per block. If only one
 *  thread exists per block, the expensive share memory allocation can be
 *  avoided
 */
template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter,
    std::size_t threads
>
struct BlockSharedMemoryAllocator
{
    using type = common::allocator::AlpakaShared<
        T_Acc,
        T_size,
        T_counter
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping, acc ) )
    {
        return T_Factory::allocView( mapping, acc );
    }
};

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter
>
struct BlockSharedMemoryAllocator<
    T_Acc,
    T_size,
    T_counter,
    1
>
{
    using type = llama::allocator::Stack<
        T_size
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping ) )
    {
        return T_Factory::allocView( mapping );
    }
};

/** Alpaka kernel for updating the speed of every particle based on the distance
 *  and mass to each other particle. Has complexity O(N²).
 */
template<
    std::size_t problemSize,
    std::size_t elems,
    std::size_t blockSize
>
struct UpdateKernel
{
    template<
        typename T_Acc,
        typename T_ViewLocal,
        typename T_ViewRemote
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const & acc,
        T_ViewLocal localParticles,
        T_ViewRemote remoteParticles,
        Element ts
    ) const
    {

        constexpr std::size_t threads = blockSize / elems;
        using SharedAllocator = BlockSharedMemoryAllocator<
            T_Acc,
            llama::SizeOf< typename decltype( remoteParticles )::Mapping::DatumDomain >::value
            * blockSize,
            __COUNTER__,
            threads
        >;

        auto threadBlockIndex = alpaka::idx::getIdx<
            alpaka::Block,
            alpaka::Threads
        >( acc )[ 0u ];

        using SharedMapping = llama::mapping::SoA<
            typename decltype( remoteParticles )::Mapping::UserDomain,
            typename decltype( remoteParticles )::Mapping::DatumDomain
        >;
        SharedMapping const sharedMapping( { blockSize } );

        using SharedFactory = llama::Factory<
            SharedMapping,
            typename SharedAllocator::type
        >;

        auto temp = SharedAllocator::template allocView<
            SharedFactory,
            SharedMapping
        >( sharedMapping, acc );

        auto threadIndex = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            start + elems,
            problemSize
        );
        LLAMA_INDEPENDENT_DATA
        for ( std::size_t b = 0; b < ( problemSize + blockSize - 1u ) / blockSize; ++b )
        {
            auto const start2 = b * blockSize;
            auto const   end2 = alpaka::math::min(
                acc,
                start2 + blockSize,
                problemSize
            ) - start2;

            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype( end2 )( 0 );
                pos2 + threadIndex < end2;
                pos2 += threads
            )
                temp( pos2 + threadBlockIndex ) = remoteParticles( start2 + pos2 + threadBlockIndex );

            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype( end2 )( 0 ); pos2 < end2; ++pos2 )
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos )
                    pPInteraction(
                        localParticles( pos ),
                        temp( pos2 ),
                        ts
                    );
        };
    }
};

/// Alpaka kernel for moving each particle with its speed. Has complexity O(N).
template<
    std::size_t problemSize,
    std::size_t elems
>
struct MoveKernel
{
    template<
        typename T_Acc,
        typename T_View
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const & acc,
        T_View particles,
        Element ts
    ) const
    {
        auto threadIndex = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            ( threadIndex + 1 ) * elems,
            problemSize
        );

        LLAMA_INDEPENDENT_DATA
        for ( auto pos = start; pos < end; ++pos )
            particles( pos )( dd::Pos( ) ) += particles( pos )( dd::Vel( ) ) * ts;
    }
};

template<
    typename T_Parameter
>
struct PassThroughAllocator
{
    using PrimType = unsigned char;
    using BlobType = PrimType*;
    using Parameter = T_Parameter*;

    LLAMA_NO_HOST_ACC_WARNING
    static inline
    auto
    allocate(
        std::size_t count,
        Parameter const pointer
    )
    -> BlobType
    {
        return reinterpret_cast<BlobType>(pointer);
    }
};

int main( int argc, char * * argv )
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt< 1 >;
    using Size = std::size_t;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf< DevHost >;

#if NBODY_CUDA == 1
    using Acc = alpaka::acc::AccGpuCudaRt< Dim, Size >;
#else
    //~ using Acc = alpaka::acc::AccCpuSerial< Dim, Size >;
    using Acc = alpaka::acc::AccCpuOmp2Blocks< Dim, Size >;
    //~ using Acc = alpaka::acc::AccCpuOmp2Threads< Dim, Size >;
    //~ using Acc = alpaka::acc::AccCpuOmp4< Dim, Size >;
#endif // NBODY_CUDA
    using DevAcc = alpaka::dev::Dev< Acc >;
    using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
#if NBODY_CUDA == 1
    using Queue = alpaka::queue::QueueCudaRtSync;
#else
    using Queue = alpaka::queue::QueueCpuSync;
#endif // NBODY_CUDA
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Queue queue( devAcc );

    dash::init(&argc, &argv);

    dart_unit_t myid = dash::myid( );
    dart_unit_t size = dash::size( );

    // NBODY
    constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
    constexpr std::size_t blockSize = NBODY_BLOCK_SIZE;
    constexpr std::size_t hardwareThreads = 2; //relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<
        Acc,
        blockSize,
        hardwareThreads
    >;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr Element ts = 0.0001;
    constexpr std::size_t steps = NBODY_STEPS;

    //DASH
    dash::Array<particle> particles;

    // LLAMA
    using UserDomain = llama::UserDomain< 1 >;
    const UserDomain userDomainSize{ problemSize };

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Particle
    >;
    Mapping const mapping( userDomainSize );

    using DevFactory = llama::Factory<
        Mapping,
        common::allocator::Alpaka<
            DevAcc,
            Size
        >
    >;
    using MirrorFactory = llama::Factory<
        Mapping,
        common::allocator::AlpakaMirror<
            DevAcc,
            Size,
            Mapping
        >
    >;
    using HostFactory = llama::Factory<
        Mapping,
        common::allocator::Alpaka<
            DevHost,
            Size
        >
    >;
    using LocalFactory = llama::Factory<
        Mapping,
        PassThroughAllocator<
            particle
        >
    >;

    if (myid == 0) {
        std::cout << ( size * problemSize ) / 1000 << " thousand particles (";
        std::cout << human_readable( size * problemSize * llama::SizeOf< Particle >::value ) << ")\n";
    }

    HRChrono chrono;

    particles.allocate( size * problemSize );
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin( ) );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrowView = MirrorFactory::allocView( mapping, devView );

    // will be used as double buffer for remote->host and host->device copying
    auto   remoteHostView =   HostFactory::allocView( mapping, devHost );
    auto    remoteDevView =    DevFactory::allocView( mapping,  devAcc );
    auto remoteMirrowView = MirrorFactory::allocView( mapping, devView );

    chrono.printAndReset( "Alloc" );

    /// Random initialization of the particles
    std::mt19937_64 generator;
    std::normal_distribution< Element > distribution(
        Element( 0 ), // mean
        Element( 1 )  // stddev
    );
    LLAMA_INDEPENDENT_DATA
    for ( std::size_t i = 0; i < problemSize; ++i )
    {
        auto temp = llama::stackVirtualDatumAlloc< Particle >( );
        temp( dd::Pos( ), dd::X( ) ) = distribution( generator );
        temp( dd::Pos( ), dd::Y( ) ) = distribution( generator );
        temp( dd::Pos( ), dd::Z( ) ) = distribution( generator );
        temp( dd::Vel( ), dd::X( ) ) = distribution( generator ) / Element( 10 );
        temp( dd::Vel( ), dd::Y( ) ) = distribution( generator ) / Element( 10 );
        temp( dd::Vel( ), dd::Z( ) ) = distribution( generator ) / Element( 10 );
        temp( dd::Mass( ) ) = distribution( generator ) / Element( 100 );
        hostView( i ) = temp;
    }

    chrono.printAndReset( "Init" );

    alpaka::mem::view::ViewPlainPtr< DevHost, unsigned char, Dim, Size > hostPlain(
        reinterpret_cast< unsigned char* >( particles.lbegin( ) ), devHost, problemSize * llama::SizeOf< Particle >::value );
    // was: alpakaMemCopy( devView, hostPlain, userDomainSize, queue );
    alpaka::mem::view::copy( queue,
        devView.blob[ 0 ].buffer,
        hostPlain,
        problemSize * llama::SizeOf< Particle >::value );

    chrono.printAndReset( "Copy H->D" );

    const alpaka::vec::Vec< Dim, Size > elems(
        static_cast< Size >( elemCount )
    );
    const alpaka::vec::Vec< Dim, Size > threads(
        static_cast< Size >( threadCount )
    );
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec< Dim, Size > blocks(
        static_cast< Size >( ( problemSize + innerCount - 1u ) / innerCount )
    );

    auto const workdiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Size
    >{
        blocks,
        threads,
        elems
    };

    UpdateKernel<
        problemSize,
        elemCount,
        blockSize
    > updateKernel;
    MoveKernel<
        problemSize,
        elemCount
    > moveKernel;
    for ( std::size_t s = 0; s < steps; ++s )
    {
        /* pair-wise with local particles */
        alpaka::kernel::exec< Acc >(
            queue,
            workdiv,
            updateKernel,
            mirrowView,
            mirrowView,
            ts
        );

        chrono.printAndReset( "Update kernel:       " );

        /* pair-wise with remote particles */
        for ( dart_unit_t unit_it = 1; unit_it < size; ++unit_it )
        {
            dart_unit_t remote = ( myid + unit_it ) % size;

            // get remote local block into remoteHostView
            auto remote_begin = particles.begin( ) + ( remote * problemSize );
            auto remote_end   = remote_begin + problemSize;
            auto target_begin = reinterpret_cast< particle* >( alpaka::mem::view::getPtrNative( remoteHostView.blob[ 0 ].buffer ) );
            dash::copy( remote_begin, remote_end, target_begin );

            chrono.printAndReset( "Copy from remote:    " );

            alpakaMemCopy( remoteDevView, remoteHostView, userDomainSize, queue );

            alpaka::kernel::exec< Acc >(
                queue,
                workdiv,
                updateKernel,
                mirrowView,
                remoteMirrowView,
                ts
            );

            chrono.printAndReset("Update remote kernel:");
        }

        alpaka::kernel::exec< Acc >(
            queue,
            workdiv,
            moveKernel,
            mirrowView,
            ts
        );
        chrono.printAndReset( "Move kernel:         " );
        dummy( static_cast< void * >( mirrowView.blob[ 0 ] ) );

        // was: alpakaMemCopy( hostPlain, devView, userDomainSize, queue );
        alpaka::mem::view::copy( queue,
            hostPlain,
            devView.blob[ 0 ].buffer,
            problemSize * llama::SizeOf< Particle >::value );

        particles.barrier( );
    }

    chrono.printAndReset( "Copy D->H" );

    dash::finalize( );

    return 0;
}

} // namespace nbody

int main( int argc, char ** argv )
{
    return nbody::main( argc, argv );
}
