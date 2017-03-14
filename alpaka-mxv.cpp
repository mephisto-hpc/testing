#include <iostream>
#include <sstream>

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
        using Size = alpaka::size::Size<TAcc>;

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

        auto const block_linear = blockCoord[0u] * extents[1u] + blockCoord[1u];
        for (Size local_x = 0; local_x < extents[2u]; ++local_x) {
            Size global_y = blockCoord[0u] * extents[2u] + globalThreadIdx[0u];
            Size global_x = blockCoord[1u] * extents[2u] + local_x;
            TData value = (global_y < extents[0u] && global_x < extents[0u]) ? (global_y * extents[0u] + global_x) : 0.0;

#if 0
            printf(
                "*[block y:%2zu x:%2zu linear:%3zu] [local y:%2zu x:%2zu linear:%3zu] %f\n",
                blockCoord[0u], blockCoord[1u], block_linear,
                globalThreadIdx[0u], local_x,
                linearizedGlobalThreadIdx[0u] * extents[2u] + local_x,
                value);
#endif
            block[linearizedGlobalThreadIdx[0u] * extents[2u] + local_x] = value;
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
    ALPAKA_FN_ACC auto operator()(
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
struct BlockMultMatrixVector
{
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

        TData prod = 0.0;
        for (TSize local_x = 0; local_x < BS; ++local_x) {
            prod += A[linearizedGlobalThreadIdx[0u] * BS + local_x] * x[local_x];
        }
        y[linearizedGlobalThreadIdx[0u]] += prod;
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
    using Dim = alpaka::dim::DimInt<1>;
    using Size = std::size_t;
    using Data = double;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

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
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    /**
     * Create a stream to the accelerator device
     *
     * A stream can be interpreted as the work queue
     * of a particular device. Streams are filled with
     * executors and alpaka takes care that these
     * executors will be executed. Streams are provided in
     * async and sync variants.
     * The example stream is a sync stream to a cpu device,
     * but it also exists an async stream for this
     * device (StreamCpuAsync).
     */
    Stream stream(devAcc);


    /**
     * Init workdiv
     *
     * A kernel is executed for each element of a
     * n-dimensional grid distinguished by the element indices.
     * The work division defines the number of kernel instantiations as
     * well as the type of parallelism used by the executor.
     * Different accelerators have different requirements on the work
     * division. For example, the sequential accelerator can not
     * provide any thread level parallelism (synchronizable as well as non synchronizable),
     * whereas the CUDA accelerator can spawn hundreds of synchronizing
     * and non synchronizing threads at the same time.
     *
     * The workdiv is divided in three levels of parallelization:
     * - grid-blocks:      The number of blocks in the grid (parallel, not synchronizable)
     * - block-threads:    The number of threads per block (parallel, synchronizable).
     *                     Each thread executes one kernel invocation.
     * - thread-elements:  The number of elements per thread (sequential, not synchronizable).
     *                     Each kernel has to execute its elements sequentially.
     *
     * - Grid   : consists of blocks
     * - Block  : consists of threads
     * - Thread : consists of elements
     *
     * Threads in the same grid can access the same global memory,
     * while threads in the same block can access the same shared
     * memory. Elements are supposed to be used for vectorization.
     * Thus, a thread can process data element size wise with its
     * vector processing unit.
     */
    alpaka::vec::Vec<Dim, Size> const elementsPerThread(
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
        static_cast<Size>(BS));

    WorkDiv const workdiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    /**
     * Run kernel
     *
     * Kernels need to be provided as classes or structs
     * which provide a public operator(). This operator is
     * the actual method that should be accelerated. An
     * object of the kernel is used to create an execution
     * unit and this unit is finally enqueued into an
     * accelerator stream. The enqueuing can be done
     * synchronously or asynchronously depending on the choosen
     * stream (see type definitions above).
     */
    HostInitBlockMatrix initMatrixKernel;
    HostInitBlockVector initVectorKernel;

    Size nElementsPerDim = NBS;

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

        auto const initVectorX(alpaka::exec::create<Host>(
            workdiv,
            initVectorKernel,
            x_block,
            1.0, 0.0,
            block_y,
            block_grid_extent));
        auto const initVectorY(alpaka::exec::create<Host>(
            workdiv,
            initVectorKernel,
            y_block,
            0.0, 0.0,
            block_y,
            block_grid_extent));

        alpaka::stream::enqueue(stream, initVectorX);
        alpaka::stream::enqueue(stream, initVectorY);

        for (Size block_x = 0; block_x < NBS; block_x++) {
            Size block_linear = block_y * NBS + block_x;

            /* @todo make block available in device memory */
            auto A_block = &A[block_linear * BS * BS];
            const alpaka::vec::Vec<Dim2, Size> block_grid_coord(block_y, block_x);
            auto const initMatrix(alpaka::exec::create<Host>(
                workdiv,
                initMatrixKernel,
                A_block,
                block_grid_extent,
                block_grid_coord));
            alpaka::stream::enqueue(stream, initMatrix);

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

    BlockMultMatrixVector multMatricVectorKernel;

    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceYBlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceXBlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS));
    alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size> deviceABlock(alpaka::mem::buf::alloc<Data, Size>(devAcc, BS * BS));

    for (Size block_y = 0; block_y < NBS; block_y++) {
        alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostYBlockPlain(&y[block_y * BS], devHost, BS);

        /* copy y from host memory to device */
        alpaka::mem::view::copy(stream, deviceYBlock, hostYBlockPlain, BS);

        for (Size block_x = 0; block_x < NBS; block_x++) {
            Size block_linear = block_y * NBS + block_x;

            alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostABlockPlain(&A[block_linear * BS * BS], devHost, BS * BS);
            alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> hostXBlockPlain(&x[block_x * BS], devHost, BS);

            /* copy A from host memory to device */
            alpaka::mem::view::copy(stream, deviceABlock, hostABlockPlain, BS * BS);
            /* copy x from host memory to device */
            alpaka::mem::view::copy(stream, deviceXBlock, hostXBlockPlain, BS);

            auto const multMatrix(alpaka::exec::create<Acc>(
                workdiv,
                multMatricVectorKernel,
                alpaka::mem::view::getPtrNative(deviceYBlock),
                alpaka::mem::view::getPtrNative(deviceABlock),
                alpaka::mem::view::getPtrNative(deviceXBlock),
                BS));
            alpaka::stream::enqueue(stream, multMatrix);
        }

        /* copy y from device back into host memory */
        alpaka::mem::view::copy(stream, hostYBlockPlain, deviceYBlock, BS);

        /* validate result */
        for (Size local_y = 0; local_y < BS; local_y++) {
            Size global_y = block_y * BS + local_y;
            Data y_value = (global_y < N) ? (global_y * N * N + ((N * (N - 1)) / 2)) : 0.0;

            if (alpaka::mem::view::getPtrNative(hostYBlockPlain)[local_y] != y_value)
                printf("Y[%zu]: %f != %f\n", global_y, alpaka::mem::view::getPtrNative(hostYBlockPlain)[local_y], y_value);
        }
    }

    delete[] A;
    delete[] x;
    delete[] y;

    /**
     * Everything is fine, so lets return :)
     */
    return EXIT_SUCCESS;
}
