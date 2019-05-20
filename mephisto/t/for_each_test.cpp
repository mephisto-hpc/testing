#include "gtest/gtest.h"
#include "for_each_test.h"
#include <libdash.h>
#include <mephisto/algorithm/for_each>
#include <mephisto/execution>
#include <mephisto/array>
#include <mephisto/entity>
#include <patterns/local_pattern.h>
#include <alpaka/alpaka.hpp>

TEST_F(ForEachTest, itWorks) {
  auto const Dim = 3;
  using Data     = Pos;
  using MetaT    = typename mephisto::Metadata<PatternT>;
  using ViewT    = typename dash::Array<Data>::local_type;
  using SizeT    = ArrayT::size_type;
  using EntityT =
      mephisto::Entity<Dim, std::size_t, alpaka::acc::AccCpuSerial>;
  using Queue   = alpaka::queue::QueueCpuSync;
  using Context = mephisto::execution::AlpakaExecutionContext<EntityT, Queue>;
  using BasePattern = dash::BlockPattern<Dim>;
  using PatternT    = patterns::BalancedLocalPattern<BasePattern, EntityT>;
  using ArrayT      = dash::Array<Data, dash::default_index_t, PatternT>;

  BasePattern base{10, 10, 10};
  PatternT pattern{base};
  ArrayT   arr{pattern};
  dash::fill(arr.begin(), arr.end(), {42, 42, 42});

  // Setup of the executor:

  // Context consists of the host, the accelerator and the stream
  Context ctx;

  // The executor is the one actually doing the computation
  mephisto::execution::AlpakaExecutor<Context> executor{ctx};

  // The policy is used to relax guarantees.
  auto policy = mephisto::execution::make_parallel_policy(executor);

  // set the coordinates using an Alpaka policy
  ForEachClb clb;
  dash::transform(policy, arr.begin(), arr.end(), arr.begin(), clb);

  // Check the written coordinates using the standard for_each_with_index
  dash::for_each_with_index(
      arr.begin(),
      arr.end(),
      [&pattern](const Data &d, PatternT::index_type i) {
        auto const coords = pattern.coords(i);
        printf("mephisto: [%lu,%lu,%lu] -  dash: [%lu,%lu,%lu]\n", d.x, d.y, d.z, coords[0], coords[1], coords[2]);
        ASSERT_EQ(d.x, coords[0]);
        ASSERT_EQ(d.y, coords[1]);
        ASSERT_EQ(d.x, coords[2]);
      });
}


int main(int argc, char **argv) {
  meph_argc = argc;
  meph_argv = argv;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

