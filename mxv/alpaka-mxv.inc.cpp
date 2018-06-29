#include <iostream>
#include <sstream>
#include <chrono>
#include <cassert>

//#include <libdash.h>
#include <alpaka/alpaka.hpp>

/**
 */
struct HostInitBlockMatrix
{
    template<
        typename TAcc,
        typename TData,
        typename TExtent,
        typename TBlockGrid>
    ALPAKA_FN_ACC_NO_CUDA auto operator()(
        TAcc const & acc,
        TData * block,
        TExtent const & extents,
        TBlockGrid const & blockCoord ) const
    -> void
    {
        using Size = alpaka::idx::Idx<TAcc>;

        /**
         * In the most cases the parallel work distibution depends
         * on the current index of a thread and how many threads
         * exist overall. These information can be obtained by
         * getIdx() and getWorkDiv(). In this example these
         * values are obtained for a global scope.
         */
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        /**
         * Map the three dimensional thread index into a
         * one dimensional thread index space. We call it
         * linearize the thread index.
         */
        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        Size N = extents[0u];
        Size NBS = extents[1u];
        Size BS = extents[2u];
        auto const block_linear = blockCoord[0u] * NBS + blockCoord[1u];
        for (Size local_x = 0; local_x < BS; ++local_x) {
            Size global_y = blockCoord[0u] * BS + linearizedGlobalThreadIdx[0u];
            Size global_x = blockCoord[1u] * BS + local_x;
            TData value = (global_y < N && global_x < N) ? (global_y * N + global_x) : 0.0;

#if 0
            printf(
                "*[block y:%2zu x:%2zu linear:%3zu] [local y:%2zu x:%2zu linear:%3zu] %f\n",
                blockCoord[0u], blockCoord[1u], block_linear,
                globalThreadIdx[0u], local_x,
                linearizedGlobalThreadIdx[0u] * BS + local_x,
                value);
#endif
            block[linearizedGlobalThreadIdx[0u] * BS + local_x] = value;
        }
    }
};

/**
 */
struct HostInitBlockVector
{
    template<
        typename TAcc,
        typename TData,
        typename TSize,
        typename TExtent>
    ALPAKA_FN_ACC_NO_CUDA auto operator()(
        TAcc const & acc,
        TData * block,
        TData initValue,
        TData invalidValue,
        TSize blockIdx,
        TExtent const & extents
         ) const
    -> void
    {
        /**
         * In the most cases the parallel work distibution depends
         * on the current index of a thread and how many threads
         * exist overall. These information can be obtained by
         * getIdx() and getWorkDiv(). In this example these
         * values are obtained for a global scope.
         */
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        /**
         * Map the three dimensional thread index into a
         * one dimensional thread index space. We call it
         * linearize the thread index.
         */
        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        TSize global_y = blockIdx * extents[2u] + globalThreadIdx[0u];

        block[linearizedGlobalThreadIdx[0u]] = global_y < extents[0u] ? initValue : invalidValue;
    }
};

/**
 */
template<
    int TThreads>
struct BlockMultMatrixVector
{
    template<
        typename TType,
        size_t TSize>
    struct Array {
        template<
            typename TIdx>
        ALPAKA_FN_HOST_ACC const TType &operator[](
            const TIdx idx) const
        {
            return m_data[idx];
        }

        template<
            typename TIdx>
        ALPAKA_FN_HOST_ACC TType & operator[](
            const TIdx idx)
        {
            return m_data[idx];
        }
    private:
        TType m_data[TSize];
    };

    template<
        typename TAcc,
        typename TData,
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TData * const y,
        TData const * const A,
        TData const * const x,
        TSize BS ) const
    -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const thread_idx = globalThreadIdx[0u];
        auto threads = globalThreadExtent[0u];
        assert(TThreads == threads);

        auto && prod = alpaka::block::shared::st::allocVar<Array<TData, TThreads>, 0>(acc);

        prod[thread_idx] = 0.0;
        for (auto i = thread_idx; i < BS; i += TThreads) {
            prod[thread_idx] += A[i] * x[i];
        }
        alpaka::block::sync::syncBlockThreads(acc);

        while (threads > 1) {
            threads /= 2;
            if (thread_idx < threads) {
                prod[thread_idx] += prod[thread_idx + threads];
            }
            alpaka::block::sync::syncBlockThreads(acc);
        }
        if (thread_idx == 0) {
            *y += prod[0];
        }
    }
};

auto
main(
    int ac,
    char* av[])
-> int
{
    /**
     * Define accelerator types
     *
     * It is possible to choose from a set of accelerators
     * that are defined in the alpaka::acc namespace e.g.:
     * - AccGpuCudaRt
     * - AccCpuThreads
     * - AccCpuFibers
     * - AccCpuOmp2Threads
     * - AccCpuOmp2Blocks
     * - AccCpuOmp4
     * - AccCpuSerial
     *
     * Each accelerator has strengths and weaknesses. Therefore,
     * they need to be choosen carefully depending on the actual
     * use case. Furthermore, some accelerators only support a
     * particular workdiv, but workdiv can also be generated
     * automatically.
     */
    using Data = double;

    using Dim = alpaka::dim::DimInt<1>;
    using Size = std::size_t;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    using Host = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    using QueueHost = alpaka::queue::QueueCpuSync;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

#ifdef USE_GPU
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    using QueueAcc = alpaka::queue::QueueCudaRtSync;
#else
//#define USE_OMP_2_THREADS
#ifdef USE_OMP_2_THREADS
    using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
#else
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
#endif
    using QueueAcc = alpaka::queue::QueueCpuSync;
#endif
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    Size N  = 1024; /* Matrix dimension */
    if (ac > 1) {
        std::istringstream in(av[1]);
        in >> N;
    }
    Size BS = 128;  /* Block size */
    if (ac > 2) {
        std::istringstream in(av[2]);
        in >> BS;
    }

    Size NBS = (N + (BS - 1)) / BS;
    Size NS = NBS * BS;

    std::cout << "N   = " << N << "\n"
              << "BS  = " << BS << "\n"
              << "NBS = " << NBS << "\n"
              << "NS  = " << NS << "\n";

    /**
     * Get the first devices
     *
     * The accelerator only defines how something should be
     * parallized, but a device is the real entity which will
     * run the parallel programm. The device can be choosen
     * by id (0 to the number of devices minus 1) or you
     * can also retrieve all devices in a vector (getDevs()).
     * In this example the first devices is choosen.
     */
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    WorkDiv const workDivHost(
        alpaka::workdiv::getValidWorkDiv<Host>(
            devHost,
            BS,
            Size(1u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

#ifdef USE_GPU
    constexpr Size CS(32);
#else
#ifdef USE_OMP_2_THREADS
    constexpr Size CS(4);
#else
    constexpr Size CS(1);
#endif
#endif

    WorkDiv const workDivAcc(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            CS,
            Size(1u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "Host: " << alpaka::acc::getAccName<Host>() << " " << workDivHost << "\n"
              << "Acc:  " << alpaka::acc::getAccName<Acc>()  << " " << workDivAcc  << "\n";

    /**
     * Create a queue to the accelerator device
     *
     * A queue can be interpreted as the work queue
     * of a particular device. Queues are filled with
     * executors and alpaka takes care that these
     * executors will be executed. Queues are provided in
     * async and sync variants.
     * The example queue is a sync queue to a cpu device,
     * but it also exists an async queue for this
     * device (QueueCpuAsync).
     */
    QueueHost queueHost(devHost);
    QueueAcc queueAcc(devAcc);

    /**
     * Run kernel
     *
     * Kernels need to be provided as classes or structs
     * which provide a public operator(). This operator is
     * the actual method that should be accelerated. An
     * object of the kernel is used to create an execution
     * unit and this unit is finally enqueued into an
     * accelerator queue. The enqueuing can be done
     * synchronously or asynchronously depending on the choosen
     * queue (see type definitions above).
     */
    HostInitBlockMatrix initMatrixKernel;
    HostInitBlockVector initVectorKernel;

    /* Create the matrix and vectors */
    Data *A = new Data[NS * NS];
    Data *x = new Data[NS];
    Data *y = new Data[NS];

    using Dim2 = alpaka::dim::DimInt<2>;
    using Dim3 = alpaka::dim::DimInt<3>;
    const alpaka::vec::Vec<Dim3, Size> block_grid_extent(N, NBS, BS);

    for (Size block_y = 0; block_y < NBS; block_y++) {
        auto y_block = &y[block_y * BS];
        auto x_block = &x[block_y * BS];

        alpaka::kernel::exec<Host>(queueHost,
            workDivHost,
            initVectorKernel,
            x_block,
            1.0, 0.0,
            block_y,
            block_grid_extent);
        alpaka::kernel::exec<Host>(queueHost,
            workDivHost,
            initVectorKernel,
            y_block,
            0.0, 0.0,
            block_y,
            block_grid_extent);

        for (Size block_x = 0; block_x < NBS; block_x++) {
            Size block_linear = block_y * NBS + block_x;

            /* @todo make block available in device memory */
            auto A_block = &A[block_linear * BS * BS];
            const alpaka::vec::Vec<Dim2, Size> block_grid_coord(block_y, block_x);
            alpaka::kernel::exec<Host>(queueHost,
                workDivHost,
                initMatrixKernel,
                A_block,
                block_grid_extent,
                block_grid_coord);

            for (Size local_y = 0; local_y < BS; local_y++) {
                Size global_y = block_y * BS + local_y;
                Data x_value = (global_y < N) ? 1.0 : 0.0;
                Data y_value = (global_y < N) ? 0.0 : 0.0;

                if (x_block[local_y] != x_value)
                    printf("x[%zu]: %f != %f\n", global_y, x_block[local_y], x_value);
                if (y_block[local_y] != y_value)
                    printf("y[%zu]: %f != %f\n", global_y, y_block[local_y], y_value);

#if 0
                printf("x[local x:%zu] [global x:%zu] %f\n",
                       local_y, global_y, x_block[local_y]);

                printf("y[local x:%zu] [global x:%zu] %f\n",
                       local_y, global_y, y_block[local_y]);
#endif
                for (Size local_x = 0; local_x < BS; local_x++) {
                    Size global_x = block_x * BS + local_x;
                    Size local_linear = local_y * BS + local_x;
                    Size global_linear = block_linear * BS * BS + local_linear;
                    Data A_value = (global_y < N && global_x < N) ? (global_y * N + global_x) : 0.;
#if 0
                    printf("A[block y:%2zu x:%2zu linear:%3zu] [local y:%2zu x:%2zu linear:%3zu] [global y:%2zu x:%2zu linear:%3zu]: %f\n",
                            block_y, block_x, block_linear,
                            local_y, local_x,
                            local_linear, /* local linear */
                            global_y,
                            global_x,
                            global_linear,
                            A_value);
#endif
                    if (A_block[local_linear] != A_value)
                        printf("A[%zu,%zu]: %f != %f\n", global_y, global_x, A_block[local_linear], A_value);
                }
            }
        }
    }

#if 1
    BlockMultMatrixVector<CS> multMatricVectorKernel;

    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceYBlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceXBlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceABlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS * BS));

    // Take the time prior to the execution.
    auto const tpStart(std::chrono::high_resolution_clock::now());

    for (Size block_y = 0; block_y < NBS; block_y++) {
        alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostYBlockPlain(&y[block_y * BS], devHost, BS);

        /* copy y from host memory to device */
        alpaka::mem::view::copy(queueAcc, deviceYBlock, hostYBlockPlain, BS);

        for (Size block_x = 0; block_x < NBS; block_x++) {
            Size block_linear = block_y * NBS + block_x;

            alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostABlockPlain(&A[block_linear * BS * BS], devHost, BS * BS);
            alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostXBlockPlain(&x[block_x * BS], devHost, BS);

            /* copy A from host memory to device */
            alpaka::mem::view::copy(queueAcc, deviceABlock, hostABlockPlain, BS * BS);
            /* copy x from host memory to device */
            alpaka::mem::view::copy(queueAcc, deviceXBlock, hostXBlockPlain, BS);

            for (Size block_r = 0; block_r < BS; block_r++) {

                alpaka::kernel::exec<Acc>(queueAcc,
                    workDivAcc,
                    multMatricVectorKernel,
                    alpaka::mem::view::getPtrNative(deviceYBlock) + block_r,
                    alpaka::mem::view::getPtrNative(deviceABlock) + block_r * BS,
                    alpaka::mem::view::getPtrNative(deviceXBlock),
                    BS);
            }
        }

        /* copy y from device back into host memory */
        alpaka::mem::view::copy(queueAcc, hostYBlockPlain, deviceYBlock, BS);

#if 0
        /* validate result */
        for (Size local_y = 0; local_y < BS; local_y++) {
            Size global_y = block_y * BS + local_y;
            Data y_value = (global_y < N) ? (global_y * N * N + ((N * (N - 1)) / 2)) : 0.0;

            if (alpaka::mem::view::getPtrNative(hostYBlockPlain)[local_y] != y_value)
                printf("Y[%zu]: %f != %f\n", global_y, alpaka::mem::view::getPtrNative(hostYBlockPlain)[local_y], y_value);
        }
#endif
    }

    // Take the time after the execution.
    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);
    std::cout << ((double)N * N)/(double)std::chrono::duration_cast<std::chrono::microseconds>(durElapsed).count() << std::endl;
#endif
    delete[] A;
    delete[] x;
    delete[] y;

    /**
     * Everything is fine, so lets return :)
     */
    return EXIT_SUCCESS;
}
