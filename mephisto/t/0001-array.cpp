#include <mephisto/array>

#include <cassert>

int
main()
{
    mephisto::array<int, 3> arr3;
    arr3[0] = 0;
    arr3[1] = 1;
    arr3[2] = 2;

    static_assert(arr3.size() == 3);
    static_assert(!arr3.empty());
    assert(arr3[0] == 0);
    assert(arr3[1] == 1);
    assert(arr3[2] == 2);

    mephisto::array<int, 4> arr4({3, 2, 1, 0});

    return 0;
}
