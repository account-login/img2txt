cimport cython
from libc.stdint cimport *
from libc.string cimport memcmp
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map


include "htab.pyd"

cdef extern from "hasher.h":
    ctypedef int hasher "&hasher"

ctypedef htab[uint64_t, uint64_t, hasher] MapType


@cython.boundscheck(False)
cdef uint64_t bits_hash(uint8_t *bits, uint32_t w, uint32_t h, uint32_t const1, uint32_t const2):
    cdef uint64_t hcode = 0, col
    cdef uint32_t x, y
    cdef uint32_t remap[2]
    cdef uint8_t zero
    for x in range(w):
        col = 0
        zero = bits[w * (h - 1) + x]
        remap[zero], remap[1 - zero] = const1, const2
        for y in range(h):
            col <<= 1
            col += remap[bits[w * y + x]]
        hcode <<= 1
        hcode ^= col
    return hcode


# indistinguishable from background
cdef bool is_vertical_line_only(uint8_t *bits, uint32_t w, uint32_t h):
    cdef uint32_t y
    for y in range(1, h):
        if memcmp(&bits[w * y], bits, w):
            return False
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--font', required=True, help='font.json')
    ap.add_argument('-o', '--output', required=True, help='output file')
    args = ap.parse_args()

    # load and process font.json
    import json
    with open(args.font, 'rt', encoding='utf-8') as fp:
        code2obj = json.load(fp)

    cdef uint32_t w, h, font_h, font_min_w, font_max_w
    _, font_h, _ = code2obj[ord('A')]
    font_min_w, font_max_w = 0xffffffff, 0

    cdef unordered_set[string] dedup
    cdef string cbits[0x10000]
    cdef string cbits_all[0x10000]
    cdef vector[uint8_t] flags
    flags.resize(0x10000)
    cdef uint32_t code
    cdef string bits
    cdef size_t i
    for code, obj in enumerate(code2obj):
        if obj is None:
            continue

        w, h, py_bits = obj
        bits = py_bits.encode()
        for i in range(bits.size()):
            bits[i] -= 48
        cbits_all[code] = bits

        if is_vertical_line_only(<uint8_t *>bits.data(), w, h):
            flags[code] = 1
            # print('vertical_line_only', code, repr(chr(code)))
            # print(bits)
            continue

        assert h == font_h
        font_max_w = max(font_max_w, w)
        font_min_w = min(font_min_w, w)

        if dedup.count(bits):
            continue
        dedup.insert(bits)
        cbits[code] = bits

    print('font dim', font_h, font_min_w, font_max_w)

    cdef uint32_t const1, const2
    const1, const2 = test_loop(&cbits[0], font_h, font_min_w, font_max_w)
    assert const1 and const2
    print('const', const1, const2)

    # 8b    magic
    # 8b    pad

    # 4b    font_h
    # 4b    blob_bytes
    # 4b    const1
    # 4b    const2

    # 4x64k codes
    # ...   blob
    cdef unordered_map[string, uint32_t] bits2off
    cdef vector[uint32_t] codes
    codes.resize(0x10000)
    cdef vector[uint32_t] blob
    cdef size_t bitsz
    for code in range(0x10000):
        if cbits_all[code].empty():
            continue

        if not bits2off.count(cbits_all[code]):
            bitsz = cbits_all[code].size()
            w = bitsz // font_h

            off_w = (blob.size() << 10) | (flags[code] << 8) | w
            assert off_w < ((<uint64_t>1) << 32)
            bits2off[cbits_all[code]] = <uint32_t>off_w

            for i in range(bitsz):
                if i % 32 == 0:
                    blob.push_back(0)
                blob[blob.size() - 1] |= (<uint32_t>cbits_all[code][i]) << (i % 32)

        codes[code] = bits2off[cbits_all[code]]

    import struct
    with open(args.output, 'wb') as fp:
        fp.truncate()
        fp.write(b'IMG2TXT!')
        fp.write(b'\x00' * 8)
        fp.write(struct.pack('<IIII', font_h, blob.size(), const1, const2))
        fp.write((<uint8_t *>codes.data())[0:0x40000])
        fp.write((<uint8_t *>blob.data())[:4*blob.size()])


cdef (uint32_t, uint32_t) test_loop(string *cbits, uint32_t font_h, uint32_t font_min_w, uint32_t font_max_w):
    cdef size_t i, j, z
    cdef uint32_t const1, const2
    for i in range(100):
        for j in range(i):
            for z in range(2):
                const1, const2 = i, j
                if z:
                    const2, const1 = const1, const2
                if const1 == 0 or const2 == 0:
                    continue
                if test_const(&cbits[0], font_h, font_min_w, font_max_w, const1, const2):
                    return (const1, const2)
    return (0, 0)


@cython.cdivision(True)
cdef bool test_const(
    string *cbits, uint32_t font_h, uint32_t font_min_w, uint32_t font_max_w,
    uint32_t const1, uint32_t const2,
) except *:
    cdef MapType hash_set
    cdef string zero
    zero.resize(font_h * font_max_w)
    cdef uint32_t w
    for code in range(0x10000):
        if cbits[code].empty():
            continue
        w = cbits[code].size() // font_h
        hcode = bits_hash(<uint8_t *>cbits[code].data(), w, font_h, const1, const2)
        assert hcode > 0
        hash_set[hcode] = 1

    for w in range(font_min_w, font_max_w + 1):
        hcode = bits_hash(<uint8_t *>zero.data(), w, font_h, const1, const2)
        assert hcode > 0, f'w:{w}'
        if hash_set.count(hcode):
            return False

    return True


if __name__ == '__main__':
    main()
