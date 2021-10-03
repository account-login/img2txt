cimport cython
from cython.operator cimport dereference
from libc.stdint cimport *
from libc.string cimport memcmp, memcpy
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from posix.unistd cimport read as posix_read


include "htab.pyd"

cdef extern from "hasher.h":
    ctypedef int hasher "&hasher"

ctypedef htab[uint64_t, uint64_t, hasher] MapType
# ctypedef unordered_map[uint64_t, Node] MapType


cdef void add_bits(_CPPFont *self, uint32_t code, uint8_t *bits, uint32_t w, uint32_t h):
    cdef uint64_t col, loc
    cdef uint64_t hval = 0
    cdef uint32_t x, y
    cdef uint32_t sz, off
    cdef uint64_t *val_ref
    cdef size_t i
    for x in range(w):
        col = 0
        for y in range(h):
            col <<= 1
            col |= bits[w * y + x]

        loc = (~col) & ((1 << h) - 1)
        if 1 & col:
            col, loc = loc, col

        hval <<= 1
        hval ^= col * self.const2 + loc * self.const1

        val_ref = &self.mapping[x if x < 64 else 63][hval]
        if x + 1 == w:
            sz = val_ref[0] >> 32
            off = <uint32_t>val_ref[0]

            for i in range(sz):
                self.codes.push_back(self.codes[off + i])
            self.codes.push_back(code)

            sz += 1
            off = self.codes.size() - sz
            val_ref[0] = ((<uint64_t>sz) << 32) | off


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


cdef cppclass _CPPFont:
    bool loaded
    uint32_t font_h
    uint32_t font_min_w, font_max_w
    uint32_t const1, const2
    vector[string] code2bits
    vector[uint32_t] codes
    MapType mapping[64]


cdef void load_bin_file(_CPPFont *self, filename):
    assert not self.loaded
    self.loaded = True

    cdef vector[uint8_t] file_data
    cdef size_t got = 0
    import os
    cdef int fd = os.open(filename, os.O_RDONLY)
    try:
        file_data.resize(<size_t>os.fstat(fd).st_size)
        while True:
            rv = posix_read(fd, file_data.data() + got, file_data.size() - got)
            assert rv > 0
            got += rv
            if got == file_data.size():
                break
    finally:
        os.close(fd)

    # fp.write(b'IMG2TXT!')
    # fp.write(b'\x00' * 8)
    # fp.write(struct.pack('<IIII', font_h, blob.size(), const1, const2))
    # fp.write((<uint8_t *>codes.data())[0:0x40000])
    # fp.write((<uint8_t *>blob.data())[:4*blob.size()])
    import struct
    assert file_data.data()[:8] == b'IMG2TXT!'
    assert file_data.data()[8:16] == b'\0\0\0\0\0\0\0\0'
    cdef size_t blob_sz
    self.font_h, blob_sz, self.const1, self.const2 = struct.unpack('<IIII', file_data.data()[16:32])
    cdef uint32_t *codes = <uint32_t *>(file_data.data() + 32)
    cdef uint32_t *blob = &codes[0x10000]
    assert 32 + 0x40000 + blob_sz <= file_data.size()

    self.font_min_w, self.font_max_w = 0xffffffff, 0
    self.code2bits.resize(0x10000)

    cdef uint32_t code
    cdef unordered_set[uint32_t] dedup
    cdef uint32_t w
    cdef uint8_t flag
    cdef size_t off, i
    cdef string bits
    for code in range(0x10000):
        if codes[code] == 0:
            continue

        if dedup.count(codes[code]):
            continue
        dedup.insert(codes[code])

        w = codes[code] & 0xff
        flag = (codes[code] >> 8) & 0b11
        off = codes[code] >> 10

        if w <= 1:
            continue    # too narrow
        if flag:
            continue    # FIXME: handle vertical lines later

        self.font_max_w = max(self.font_max_w, w)
        self.font_min_w = min(self.font_min_w, w)

        bits.resize(w * self.font_h)
        for i in range(w * self.font_h):
            bits[i] = <bool>(blob[off + i // 32] & (1 << (i % 32)))

        add_bits(self, code, <uint8_t *>bits.data(), w, self.font_h)
        self.code2bits[code] = bits
        # self.code2bits[code].swap(bits)

    # check empty space not collide with other chars
    cdef size_t mapidx
    for w in range(1, self.font_max_w + 1):
        hval = space_hash(w, self.font_h, self.const1, self.const2)
        mapidx = w - 1 if w - 1 < 64 else 63
        if not self.mapping[mapidx].count(hval):
            continue
        assert self.mapping[mapidx][hval] == 0

    # assert not self.mapping.count(0)


@cython.cdivision(True)
cdef CharResult run(_CPPFont *self, uint32_t *pixels, uint32_t w, uint32_t h):
    assert self.loaded

    cdef CharResult out = CharResult()

    cdef uint32_t font_h = self.font_h
    cdef uint32_t const1, const2
    const1, const2 = self.const1, self.const2
    cdef uint32_t font_min_w, font_max_w
    font_min_w, font_max_w = self.font_min_w, self.font_max_w

    import time
    t0 = time.monotonic()

    # vhash
    cdef vector[uint32_t] vhash
    vhash.resize(w * h)
    get_vhash(vhash.data(), font_h, const1, const2, pixels, w, h)

    t1 = time.monotonic()

    # match
    cdef uint32_t x, y
    cdef uint64_t hval
    cdef uint8_t *cbits
    cdef uint32_t fw

    cdef vector[uint64_t] hhash_data
    hhash_data.resize(w * 2)
    cdef uint64_t *hhash_in = hhash_data.data()
    cdef uint64_t *hhash_out = hhash_data.data() + w

    cdef vector[uint32_t] hpos_data
    hpos_data.resize(w * 2)
    cdef uint32_t *hpos_in = hpos_data.data()
    cdef uint32_t *hpos_out = hpos_data.data() + w

    cdef size_t in_size, out_size
    cdef size_t i
    cdef MapType *m
    cdef uint64_t sz_off
    cdef uint32_t sz, off

    for y in range(h + 1 - font_h):
        in_size, out_size = w, 0
        for i in range(w):
            hpos_in[i] = i
        for i in range(w):
            hhash_in[i] = vhash[w * y + i]
        for fw in range(1, font_min_w):
            for i in range(w - fw):
                hhash_in[i] <<= 1
                hhash_in[i] ^= vhash[w * y + i + fw]

        for fw in range(font_min_w, font_max_w + 1):
            m = &self.mapping[fw - 1 if fw - 1 < 64 else 63]
            for i in range(in_size):
                x = hpos_in[i]
                hval = hhash_in[i]
                it = m.find(hval)
                if it == m.end():
                    continue

                sz_off = dereference(it).second
                sz = sz_off >> 32
                off = <uint32_t>sz_off
                for i in range(sz):
                    code = self.codes[off + i]
                    # assert not self.code2bits[code].empty()
                    if self.code2bits[code].size() != fw * font_h:
                        continue

                    out.stats_check += 1
                    cbits = <uint8_t *>self.code2bits[code].data()
                    if not is_match(pixels, w, h, x, y, cbits, fw, font_h):
                        continue

                    out.stats_hit += 1
                    out.chars.push_back(CharPos(x=x, y=y, code=code, font=<void *>self))

                if x + fw < w:
                    hval <<= 1
                    hval ^= vhash[w * y + x + fw]
                    hhash_out[out_size] = hval
                    hpos_out[out_size] = x
                    out_size += 1

            out.stats_lookup += in_size
            out.stats_node += out_size

            hhash_in, hhash_out = hhash_out, hhash_in
            hpos_in, hpos_out = hpos_out, hpos_in
            in_size, out_size = out_size, 0

    t2 = time.monotonic()

    out.stats_time_vhash_us = (t1 - t0) * 1e6
    out.stats_time_loop_us = (t2 - t1) * 1e6
    out.stats_pos_total = (w - font_min_w + 1) * (h - font_h + 1)
    return out


cdef void get_vhash(
    uint32_t *vhash, uint32_t font_h, uint32_t const1, uint32_t const2,
    uint32_t *pixels, uint32_t w, uint32_t h,
):
    assert font_h < 32
    if h < font_h:
        return

    cdef uint32_t mask = (1 << font_h) - 1
    cdef uint32_t hval      # NOTE: font_h < 32
    cdef uint32_t x, y
    cdef uint32_t c, pc
    cdef vector[uint32_t] col
    col.resize(w)
    for y in range(1, h):
        for x in range(w):
            pc = pixels[w * (y - 1) + x]
            c = pixels[w * y + x]
            if pc != c:
                col[x] = ~col[x]
            col[x] <<= 1

            if y + 1 < font_h:
                continue

            hval = (col[x] & mask) * const2 + ((~col[x]) & mask) * const1
            vhash[w * (y + 1 - font_h) + x] = hval


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


cdef class Font:
    cdef _CPPFont *wrapped

    def __cinit__(self):
        self.wrapped = new _CPPFont()

    def __dealloc__(self):
        del self.wrapped

    cdef void load_bin_file(self, filename):
        load_bin_file(self.wrapped, filename)

    cdef CharResult run(self, uint32_t *pixels, uint32_t w, uint32_t h):
        return run(self.wrapped, pixels, w, h)


cdef main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='image file')
    ap.add_argument('--bin', required=True, help='trained font file')
    args = ap.parse_args()

    # load image
    import PIL.Image
    im = PIL.Image.open(args.image)
    im = im.convert('RGB')
    im.load()

    cdef uint32_t w, h
    w, h = im.width, im.height
    cdef vector[uint32_t] pixels
    pixels.resize(w * h)
    cdef uint32_t **rgb = <uint32_t **><uintptr_t>(dict(im.im.unsafe_ptrs)['image32'])

    cdef uint32_t x, y
    for y in range(h):
        for x in range(w):
            pixels[w * y + x] = rgb[y][x] & 0xffffff

    # load font
    font = Font()
    font.load_bin_file(args.bin)

    # for i in range(64):
    #     if font.wrapped.mapping[i].empty():
    #         continue
    #     print(
    #         'w', i,
    #         'maping_size', font.wrapped.mapping[i].size(),
    #         'load_factor', font.wrapped.mapping[i].load_factor()
    #     )

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
