cimport cython
from cython.operator cimport dereference
from libc.stdint cimport *
from libc.string cimport memcmp
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map


# indistinguishable from background
cdef bool is_vertical_line_only(uint8_t *bits, uint32_t w, uint32_t h):
    cdef uint32_t y
    for y in range(1, h):
        if memcmp(&bits[w * y], bits, w):
            return False
    return True


cdef struct Node:
    uint64_t levels
    vector[uint32_t] codes


include "htab.pyd"

cdef extern from "hasher.h":
    ctypedef int hasher "&hasher"

ctypedef htab[uint64_t, Node, hasher] MapType
# ctypedef unordered_map[uint64_t, Node] MapType


cdef void add_bits(
    MapType &mapping, uint32_t const1, uint32_t const2,
    uint32_t code, uint8_t *bits, uint32_t w, uint32_t h,
):
    cdef uint64_t col, loc
    cdef uint64_t hval = 0
    cdef uint32_t x, y
    for x in range(w):
        col = 0
        for y in range(h):
            col <<= 1
            col |= bits[w * y + x]

        loc = (~col) & ((1 << h) - 1)
        if 1 & col:
            col, loc = loc, col

        hval <<= 1
        hval ^= col * const2 + loc * const1

        is_new = not mapping.count(hval)
        node = &mapping[hval]
        if is_new:
            node.levels = 0

        if x + 1 == w:
            # if code == 65:
            #     assert hval == 1073122565, f'bits:{bits[:w*h]}'
            node.codes.push_back(code)
        else:
            node.levels |= (<uint64_t>1) << x


cdef uint64_t space_hash(uint32_t w, uint32_t h, uint32_t const1, uint32_t const2):
    cdef uint64_t hval = 0
    for _ in range(w):
        hval <<= 1
        hval ^= (((<uint64_t>1) << h) - 1) * const1
    return hval


cdef struct CharPos:
    uint32_t x, y
    uint32_t code
    void *font


cdef class CharResult:
    cdef uint64_t stats_time_vhash_us
    cdef uint64_t stats_time_loop_us
    cdef uint64_t stats_lookup
    cdef uint64_t stats_node
    cdef uint64_t stats_check
    cdef uint64_t stats_hit
    cdef uint64_t stats_pos_total

    cdef vector[CharPos] chars


cdef class Font:
    cdef bool loaded
    cdef uint32_t font_h
    cdef uint32_t font_min_w, font_max_w
    cdef MapType mapping
    cdef uint32_t const1, const2
    cdef vector[string] code2bits

    def __init__(self):
        pass

    def load_json_file(self, filename, uint32_t const1, uint32_t const2):
        assert not self.loaded
        self.loaded = True

        self.const1, self.const2 = const1, const2

        import json
        with open(filename, 'rt', encoding='utf-8') as fp:
            code2obj = json.load(fp)

        _, self.font_h, _ = code2obj[ord('A')]
        self.font_min_w, self.font_max_w = 0xffffffff, 0

        self.code2bits.resize(0x10000)

        cdef uint32_t w, h
        cdef string bits
        cdef unordered_set[string] dedup
        cdef uint32_t code
        for code, obj in enumerate(code2obj):
            if obj is None:
                continue

            w, h, py_bits = obj
            assert h == self.font_h
            bits = py_bits.encode('utf-8')
            for i in range(bits.size()):
                bits[i] -= 48

            if dedup.count(bits):
                continue
            dedup.insert(bits)

            if w <= 1:
                continue    # too narrow
            if is_vertical_line_only(<uint8_t *>bits.data(), w, h):
                continue    # FIXME: handle this later

            self.font_max_w = max(self.font_max_w, w)
            self.font_min_w = min(self.font_min_w, w)

            add_bits(self.mapping, const1, const2, code, <uint8_t *>bits.data(), w, h)
            self.code2bits[code] = bits
            # self.code2bits[code].swap(bits)

        # check empty space not collide with other chars
        for w in range(self.font_max_w + 1):
            hval = space_hash(w, self.font_h, const1, const2)
            if not self.mapping.count(hval):
                continue
            assert self.mapping[hval].codes.empty(), f'w:{w} hval:{hval} node:{self.mapping[hval]} bits:{self.code2bits[self.mapping[hval].codes[0]]}'

        # assert not self.mapping.count(0)

    @cython.cdivision(True)
    cdef CharResult run(self, uint32_t *pixels, uint32_t w, uint32_t h):
        assert self.loaded

        cdef CharResult out = CharResult()

        cdef uint32_t font_h = self.font_h
        cdef uint32_t mask = (1 << font_h) - 1

        cdef uint32_t const1, const2
        const1, const2 = self.const1, self.const2
        cdef uint32_t font_min_w, font_max_w
        font_min_w, font_max_w = self.font_min_w, self.font_max_w

        import time
        t0 = time.monotonic()

        # vhash
        cdef vector[uint64_t] vhash
        vhash.resize(w * h)

        cdef uint32_t x, y
        cdef uint32_t pc
        cdef uint64_t col, hval
        for x in range(w):
            col = 0
            pc = pixels[x]
            for y in range(h):
                if pixels[w * y + x] != pc:
                    pc = pixels[w * y + x]
                    col = ~col
                col <<= 1

                if y + 1 < font_h:
                    continue

                hval = (col & mask) * const2 + ((~col) & mask) * const1
                vhash[w * (y + 1 - font_h) + x] = hval
        t1 = time.monotonic()

        # match
        cdef uint8_t *cbits
        cdef uint32_t fw
        cdef uint32_t ix
        for y in range(h + 1 - font_h):
            for x in range(w):
                hval = 0
                for ix in range(x, min(x + font_max_w, w)):
                    hval <<= 1
                    hval ^= vhash[w * y + ix]

                    if ix - x + 1 < font_min_w:
                        continue

                    out.stats_lookup += 1
                    it = self.mapping.find(hval)
                    if it == self.mapping.end():
                        break

                    node = &dereference(it).second
                    for code in node.codes:
                        # assert not self.code2bits[code].empty()
                        # if self.code2bits[code].size() != (ix - x + 1) * font_h:
                        #     continue
                        fw = self.code2bits[code].size() // font_h
                        if fw != ix - x + 1:
                            continue

                        out.stats_check += 1
                        cbits = <uint8_t *>self.code2bits[code].data()
                        if not is_match(pixels, w, h, x, y, cbits, fw, font_h):
                            continue

                        out.stats_hit += 1
                        out.chars.push_back(CharPos(x=x, y=y, code=code, font=<void *>self))

                    if not (node.levels & ((<uint64_t>1) << (ix - x))):
                        # XXX: handle ix - x >= 64?
                        # XXX: 32bit enough?
                        break
                    out.stats_node += 1

        t2 = time.monotonic()

        out.stats_time_vhash_us = (t1 - t0) * 1e6
        out.stats_time_loop_us = (t2 - t1) * 1e6
        out.stats_pos_total = (w - font_min_w + 1) * (h - font_h + 1)
        return out


cdef bool is_match(
    uint32_t *pixels, uint32_t iw, uint32_t ih,
    uint32_t ix, uint32_t iy,
    uint8_t *bits, uint32_t fw, uint32_t fh,
):
    cdef uint32_t[2] c_ref = [-1, -1]
    cdef uint8_t b
    cdef uint32_t c
    cdef uint32_t fx, fy
    for fy in range(fh):
        for fx in range(fw):
            c = pixels[iw * (iy + fy) + ix + fx]
            b = bits[fw * fy + fx]
            if c != c_ref[b]:
                if c_ref[b] == <uint32_t>-1 and c_ref[not b] != c:
                    c_ref[b] = c
                    continue
                return False
    return True


cdef main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='image file')
    ap.add_argument('--font', required=True, help='font.json')
    ap.add_argument('--const1', required=True, type=int, help='const1')
    ap.add_argument('--const2', required=True, type=int, help='const1')
    args = ap.parse_args()

    # load image
    import PIL.Image
    im = PIL.Image.open(args.image)
    im = im.convert('RGB')

    cdef uint32_t w, h
    w, h = im.width, im.height
    cdef vector[uint32_t] pixels
    pixels.resize(w * h)

    cdef uint32_t x, y, r, g, b
    for y in range(h):
        for x in range(w):
            r, g, b = im.getpixel((x, y))
            pixels[w * y + x] = (b << 16) | (g << 8) | r

    # load font
    font = Font()
    font.load_json_file(args.font, args.const1, args.const2)
    print(
        'maping_size', font.mapping.size(),
        'load_factor', font.mapping.load_factor(), 'max_load_factor', font.mapping.max_load_factor(),
    )

    result = font.run(pixels.data(), w, h)

    print('stats_lookup', result.stats_lookup)
    print('stats_node', result.stats_node)
    print('stats_check', result.stats_check)
    print('stats_hit', result.stats_hit)
    print('stats_pos_total', result.stats_pos_total)
    print()
    print('stats_time_vhash_us', result.stats_time_vhash_us)
    print('stats_time_loop_us', result.stats_time_loop_us)

    # for cp in result.chars:
    #     print(cp.x, cp.y, cp.code, chr(cp.code))


# cdef struct CharPos:
#     uint32_t x, y
#     uint32_t code
#     void *font


if __name__ == '__main__':
    main()