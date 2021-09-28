from libc.stdint cimport *
from libcpp.string cimport string


include "htab.pyd"


cdef extern from "hasher.h":
    ctypedef int hasher "&hasher"


ctypedef uint64_t K
ctypedef string V
ctypedef htab[K, V, hasher] HMap


cdef main():
    cdef HMap m
    assert m.size() == 0
    m[1] = b"asdf"
    assert m.size() == 1
    assert m[1] == b"asdf"

    for x in range(100000):
        m[10 + x] = str(x).encode()
        assert m.size() == 2 + x
        # if x % 100 == 0:
        #     print(m.load_factor())

    for x in range(100000):
        assert m[10 + x] == <string>str(x).encode()
        assert m.find(100010 + x) == m.end(), f'x:{x} a:{<uintptr_t>m.find(100000 + x)}'


main()
