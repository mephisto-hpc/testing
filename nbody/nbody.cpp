#include <iostream>
#include <utility>

#include <libdash.h>

#define NBODY_CUDA 0
#define NBODY_USE_TREE 1
#define NBODY_USE_SHARED 1
#define NBODY_USE_SHARED_TREE 1

#define NBODY_PROBLEM_SIZE 16*1024
#define NBODY_BLOCK_SIZE 256
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
#else
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
#endif
#include <llama/llama.hpp>
#include <random>

#include "AlpakaAllocator.hpp"
#include "Chrono.hpp"
#include "Dummy.hpp"
#include "Human.hpp"

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

template<
    typename T_VirtualDatum1,
    typename T_VirtualDatum2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_VirtualDatum1&& localP,
    T_VirtualDatum2&& remoteP,
    Element const & ts
)
-> void
{
    Element const d[3] = {
        localP( dd::Pos(), dd::X() ) -
        remoteP( dd::Pos(), dd::X() ),
        localP( dd::Pos(), dd::Y() ) -
        remoteP( dd::Pos(), dd::Y() ),
        localP( dd::Pos(), dd::Z() ) -
        remoteP( dd::Pos(), dd::Z() )
    };
    Element distSqr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + EPS2;
    Element distSixth = distSqr * distSqr * distSqr;
    Element invDistCube = 1.0f / sqrtf(distSixth);
    Element s = remoteP( dd::Mass() ) * invDistCube;
    Element const v_d[3] = {
        d[0] * s * ts,
        d[1] * s * ts,
        d[2] * s * ts
    };
    localP( dd::Vel(), dd::X() ) += v_d[0];
    localP( dd::Vel(), dd::Y() ) += v_d[1];
    localP( dd::Vel(), dd::Z() ) += v_d[2];
}

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter,
    std::size_t threads
>
struct BlockSharedMemoryAllocator
{
    using type = nbody::allocator::AlpakaShared<
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
        T_Acc const &acc,
        T_ViewLocal localParticles,
        T_ViewRemote remoteParticles,
        Element ts
    ) const
    {

#if NBODY_USE_SHARED == 1
        constexpr std::size_t threads = blockSize / elems;
        using SharedAllocator = BlockSharedMemoryAllocator<
            T_Acc,
            llama::SizeOf< typename decltype(remoteParticles)::Mapping::DatumDomain >::value
            * blockSize,
            __COUNTER__,
            threads
        >;


#if NBODY_USE_SHARED_TREE == 1
        auto treeOperationList = llama::makeTuple(
            llama::mapping::tree::functor::LeaveOnlyRT( )
        );
        using SharedMapping = llama::mapping::tree::Mapping<
            typename decltype(remoteParticles)::Mapping::UserDomain,
            typename decltype(remoteParticles)::Mapping::DatumDomain,
            decltype( treeOperationList )
        >;
        SharedMapping const sharedMapping(
            { blockSize },
            treeOperationList
        );
#else
        using SharedMapping = llama::mapping::SoA<
            typename decltype(remoteParticles)::Mapping::UserDomain,
            typename decltype(remoteParticles)::Mapping::DatumDomain
        >;
        SharedMapping const sharedMapping( { blockSize } );
#endif // NBODY_USE_SHARED_TREE

        using SharedFactory = llama::Factory<
            SharedMapping,
            typename SharedAllocator::type
        >;

        auto temp = SharedAllocator::template allocView<
            SharedFactory,
            SharedMapping
        >( sharedMapping, acc );
#endif // NBODY_USE_SHARED

        auto threadIndex  = alpaka::idx::getIdx<
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
        for ( std::size_t b = 0; b < problemSize / blockSize; ++b )
        {
            auto const start2 = b * blockSize;
            auto const   end2 = alpaka::math::min(
                acc,
                start2 + blockSize,
                problemSize
            ) - start2;
#if NBODY_USE_SHARED == 1
            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype(end2)(0);
                pos2 + threadIndex < end2;
                pos2 += threads
            )
                temp(pos2 + threadIndex) = remoteParticles( start2 + pos2 + threadIndex );
#endif // NBODY_USE_SHARED
            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2 )
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos )
                    pPInteraction(
                        localParticles( pos ),
#if NBODY_USE_SHARED == 1
                        temp( pos2 ),
#else
                        remoteParticles( start2 + pos2 ),
#endif // NBODY_USE_SHARED
                        ts
                    );
        };
    }
};


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
        T_Acc const &acc,
        T_View particles,
        Element ts
    ) const
    {
        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            (threadIndex + 1) * elems,
            problemSize
        );

        LLAMA_INDEPENDENT_DATA
        for ( auto pos = start; pos < end; ++pos )
        {
            particles( pos )( dd::Pos(), dd::X() ) +=
                particles( pos )( dd::Vel(), dd::X() ) * ts;
            particles( pos )( dd::Pos(), dd::Y() ) +=
                particles( pos )( dd::Vel(), dd::Y() ) * ts;
            particles( pos )( dd::Pos(), dd::Z() ) +=
                particles( pos )( dd::Vel(), dd::Z() ) * ts;
        };
    }
};

template<
    typename T_Acc,
    std::size_t blockSize,
    std::size_t hardwareThreads
>
struct ThreadsElemsDistribution
{
    static constexpr std::size_t elemCount = blockSize;
    static constexpr std::size_t threadCount = 1u;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccGpuCudaRt<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount = 1u;
        static constexpr std::size_t threadCount = blockSize;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccCpuOmp2Threads<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount =
            ( blockSize + hardwareThreads - 1u ) / hardwareThreads;
        static constexpr std::size_t threadCount = hardwareThreads;
    };
#endif

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


int main(int argc,char * * argv)
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt< 1 >;
    using Size = std::size_t;
    using Extents = Size;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;

#if NBODY_CUDA == 1
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
#else
    //~ using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp4<Dim, Size>;
#endif // NBODY_CUDA
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
#if NBODY_CUDA == 1
    using Queue = alpaka::queue::QueueCudaRtSync;
#else
    using Queue = alpaka::queue::QueueCpuSync;
#endif // NBODY_CUDA
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Queue queue( devAcc ) ;

    dash::init(&argc, &argv);

    dart_unit_t myid = dash::myid();
    dart_unit_t size = dash::size();

    // NBODY
    constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
    constexpr std::size_t blockSize = NBODY_BLOCK_SIZE;
    constexpr std::size_t hardwareThreads = 2; //relevant for OpenMP2Threads
    using Distribution = ThreadsElemsDistribution<
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

#if NBODY_USE_TREE == 1
    auto treeOperationList = llama::makeTuple(
        llama::mapping::tree::functor::LeaveOnlyRT( )
    );
    using Mapping = llama::mapping::tree::Mapping<
        UserDomain,
        Particle,
        decltype( treeOperationList )
    >;
    const Mapping mapping(
        userDomainSize,
        treeOperationList
    );
#else
    using Mapping = llama::mapping::SoA<
        UserDomain,
        Particle
    >;
    Mapping const mapping( userDomainSize );
#endif // NBODY_USE_TREE

    using DevFactory = llama::Factory<
        Mapping,
        nbody::allocator::Alpaka<
            DevAcc,
            Dim,
            Size
        >
    >;
    using MirrorFactory = llama::Factory<
        Mapping,
        nbody::allocator::AlpakaMirror<
            DevAcc,
            Dim,
            Size,
            Mapping
        >
    >;
    using LocalFactory = llama::Factory<
        Mapping,
        PassThroughAllocator<
            particle
        >
    >;

    if (myid == 0) {
        std::cout << (size * problemSize) / 1000 << " thousand particles (";
        std::cout << human_readable(size * problemSize * llama::SizeOf<Particle>::value) << ")\n";
    }

    Chrono chrono;

    particles.allocate(size * problemSize);
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin() );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrowView = MirrorFactory::allocView( mapping, devView );

    auto    remoteDevView =    DevFactory::allocView( mapping,  devAcc );
    auto remoteMirrowView = MirrorFactory::allocView( mapping, devView );

    chrono.printAndReset("Alloc:");

    std::default_random_engine generator;
    std::normal_distribution< Element > distribution(
        Element( 0 ), // mean
        Element( 1 )  // stddev
    );
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problemSize; ++i)
    {
        //~ auto temp = llama::tempAlloc< 1, Particle >();
        //~ temp(dd::Pos(), dd::X()) = distribution(generator);
        //~ temp(dd::Pos(), dd::Y()) = distribution(generator);
        //~ temp(dd::Pos(), dd::Z()) = distribution(generator);
        //~ temp(dd::Vel(), dd::X()) = distribution(generator)/Element(10);
        //~ temp(dd::Vel(), dd::Y()) = distribution(generator)/Element(10);
        //~ temp(dd::Vel(), dd::Z()) = distribution(generator)/Element(10);
        hostView(i) = seed;
        //~ hostView(dd::Pos(), dd::X()) = seed;
        //~ hostView(dd::Pos(), dd::Y()) = seed;
        //~ hostView(dd::Pos(), dd::Z()) = seed;
        //~ hostView(dd::Vel(), dd::X()) = seed;
        //~ hostView(dd::Vel(), dd::Y()) = seed;
        //~ hostView(dd::Vel(), dd::Z()) = seed;
    }

    chrono.printAndReset("Init:");

    const alpaka::vec::Vec< Dim, Size > elems (
        static_cast< Size >( elemCount )
    );
    const alpaka::vec::Vec< Dim, Size > threads (
        static_cast< Size >( threadCount )
    );
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec< Dim, Size > blocks (
        static_cast< Size >( ( problemSize + innerCount - 1 ) / innerCount )
    );

    auto const workdiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Size
    > {
        blocks,
        threads,
        elems
    };

    // copy hostView to devView

    UpdateKernel<
        problemSize,
        elemCount,
        blockSize
    > updateKernel;
    MoveKernel<
        problemSize,
        elemCount
    > moveKernel;
    for ( std::size_t s = 0; s < steps; ++s)
    {
        /* pair-wise with local particles */
        alpaka::kernel::exec< Acc > (
            queue,
            workdiv,
            updateKernel,
            mirrowView,
            mirrowView,
            ts
        );

        chrono.printAndReset("Update kernel:       ");

        /* pair-wise with remote particles */
        for (dart_unit_t unit_it = 1; unit_it < size; ++unit_it)
        {
            dart_unit_t remote = (myid + unit_it) % size;

            // get remote local block into remoteMirrorView
            auto remote_begin = particles.begin() + (remote * problemSize);
            auto remote_end   = remote_begin + problemSize;
            auto target_begin = reinterpret_cast<particle*>(remoteMirrowView.blob[0]);
            dash::copy(remote_begin, remote_end, target_begin);

            chrono.printAndReset("Copy from remote:    ");

            alpaka::kernel::exec< Acc > (
                queue,
                workdiv,
                updateKernel,
                mirrowView,
                remoteMirrowView,
                ts
            );

            chrono.printAndReset("Update remote kernel:");
        }

        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            moveKernel,
            mirrowView,
            ts
        );
        chrono.printAndReset("Move kernel:         ");
        dummy( static_cast<void*>( mirrowView.blob[0] ) );

        particles.barrier();
    }

    dash::finalize();

    return 0;
}
