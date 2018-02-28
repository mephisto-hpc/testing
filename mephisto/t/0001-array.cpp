#include <mephisto/array>

#include <cassert>

int
main()
{
    mephisto::array<int, 3> arr3 = { 0, 1, 2 };
    arr3[0] = 0;
    arr3[1] = 1;
    arr3[2] = 2;

    static_assert(arr3.size() == 3, "Size needs to be 3");
    static_assert(!arr3.empty(), "Array shall not be empty" );
    assert(arr3[0] == 0);
    assert(arr3[1] == 1);
    assert(arr3[2] == 2);

    std::array<int,3> from_dash{1,2,3};
    mephisto::array<int, 3> to_mephisto( from_dash );

    mephisto::array<int, 4> arr4({3, 2, 1, 0});

    return 0;
}
