cimport cython
from cython.operator cimport dereference
from libc.stdint cimport *
from libc.string cimport memcmp, memcpy
from libcpp cimport bool
from libcpp.algorithm cimport sort as std_sort
from libcpp.string cimport string, npos
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from posix.unistd cimport read as posix_read


cdef extern from *:
    enum: PyUnicode_4BYTE_KIND
    object PyUnicode_FromKindAndData(int kind, const void *buffer, Py_ssize_t size)


include "htab.pyd"

cdef extern from "hasher.h":
    ctypedef int hasher "&hasher"

ctypedef htab[uint64_t, uint64_t, hasher] MapType
# ctypedef unordered_map[uint64_t, Node] MapType


cdef extern from "Il_fix.h":
    cdef unordered_set[string] g_Il_fix


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
    uint32_t x, y, w
    uint32_t code
    void *font


cdef class CharResult:
    cdef public uint64_t stats_time_vhash_us
    cdef public uint64_t stats_time_loop_us
    cdef public uint64_t stats_lookup
    cdef public uint64_t stats_node
    cdef public uint64_t stats_check
    cdef public uint64_t stats_hit
    cdef public uint64_t stats_pos_total

    cdef vector[CharPos] chars

    def num_chars(self):
        return self.chars.size()

    def get_char(self, i):
        if not (0 <= i < self.chars.size()):
            raise KeyError(i)
        return (self.chars[i].x, self.chars[i].y, self.chars[i].code)


cdef struct CharGroup:
    uint32_t w, h
    vector[CharPos] chars


cdef class LineResult:
    cdef vector[CharGroup] lines

    def tolist(self):
        l = []
        cdef vector[uint32_t] pycodes
        cdef size_t i
        for i in range(self.lines.size()):
            line = &self.lines[i]
            pycodes.clear()
            for ch in line.chars:
                pycodes.push_back(ch.code)
            py_str = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, pycodes.data(), pycodes.size())
            l.append((line.chars[0].x, line.chars[0].y, line.w, line.h, py_str))
        return l


cdef cppclass _CPPFont:
    bool loaded
    uint32_t font_h
    uint32_t font_min_w, font_max_w
    uint32_t space_w
    bool fix_Il
    uint32_t const1, const2
    vector[string] code2bits
    vector[uint32_t] codes
    MapType mapping[64]


cdef int32_t read_all(filename, vector[uint8_t] &out) except *:
    cdef size_t got = 0
    import os
    cdef int fd = os.open(filename, os.O_RDONLY)
    cdef ssize_t rv
    try:
        out.resize(<size_t>os.fstat(fd).st_size)
        while True:
            rv = posix_read(fd, out.data() + got, out.size() - got)
            if rv < 0:
                return rv
            assert rv > 0
            got += rv
            if got == out.size():
                break
    finally:
        os.close(fd)
    return 0


cdef void load_bin_file(_CPPFont *self, filename) except *:
    assert not self.loaded
    self.loaded = True

    cdef vector[uint8_t] file_data
    rv = read_all(filename, file_data)
    assert rv == 0

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
    self.space_w = 0
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
        if flag and code == 32:
            self.space_w = w
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

    self.fix_Il = (codes[<size_t>b'I'] == codes[<size_t>b'l'])

    # check empty space not collide with other chars
    cdef size_t mapidx
    for w in range(1, self.font_max_w + 1):
        hval = space_hash(w, self.font_h, self.const1, self.const2)
        mapidx = w - 1 if w - 1 < 64 else 63
        if not self.mapping[mapidx].count(hval):
            continue
        assert self.mapping[mapidx][hval] == 0

    # assert not self.mapping.count(0)


# TODO: add stride param to support cropping
# TODO: add dist param to restrict the y coords
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
                    out.chars.push_back(CharPos(x=x, y=y, w=fw, code=code, font=<void *>self))

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


cdef bool less_by_pos(const CharPos &lhs, const CharPos &rhs):
    cdef uint64_t k1 = ((<uint64_t>lhs.y) << 32) | lhs.x
    cdef uint64_t k2 = ((<uint64_t>rhs.y) << 32) | rhs.x
    if k1 == k2:
        return lhs.w > rhs.w    # prioritize longer chars
    else:
        return k1 < k2


cdef void fix_Il(CharGroup &line):
    cdef string word
    cdef size_t i
    cdef bool need_fix, is_upper
    for i in range(line.chars.size()):
        code = line.chars[i].code
        if b'a' <= code <= b'z' or b'A' <= code <= b'Z':
            word.push_back(code)
        elif word.size() > 0:
            need_fix = word.size() > 1
            need_fix = need_fix and word.find(73) != npos
            is_upper = True
            for ch in word:
                is_upper = is_upper and b'A' <= ch <= b'Z'
            need_fix = need_fix and not is_upper
            need_fix = need_fix and g_Il_fix.count(word)
            if need_fix:
                j = i - word.size()
                for idx in range(j, i):
                    if line.chars[idx].code == b'I':
                        line.chars[idx].code = b'l'
            word.clear()


cdef LineResult make_lines(_CPPFont *self, uint32_t *pixels, uint32_t w, uint32_t h, CharResult cr):
    cdef LineResult out = LineResult()

    # group adjacent chars
    std_sort(cr.chars.begin(), cr.chars.end(), &less_by_pos)

    cdef CharGroup *line
    cdef vector[uint8_t] used
    cdef size_t i = 0, j, cur, used_cnt
    cdef uint32_t x = 0
    while i < cr.chars.size():
        # group by y
        j = i + 1
        while j < cr.chars.size() and cr.chars[i].y == cr.chars[j].y:
            j += 1

        used_cnt = 0
        used.clear()
        used.resize(j - i)
        while used_cnt < j - i:     # possibly overlapping lines
            out.lines.push_back(CharGroup(w=0, h=self.font_h))
            line = &out.lines.back()
            for cur in range(i, j):
                if used[cur - i]:
                    continue
                if line.chars.empty():
                    x = cr.chars[cur].x
                if x == cr.chars[cur].x:
                    x += cr.chars[cur].w
                    used[cur - i] = 1
                    used_cnt += 1
                    line.w += cr.chars[cur].w
                    line.chars.push_back(cr.chars[cur])
                elif x < cr.chars[cur].x:
                    break
                # else overlapping

        i = j

    if out.lines.empty():
        return out

    # TODO: remove all overlapping lines
    # remove overlapping lines with the same y
    cdef CharGroup *cur_line
    cdef CharGroup *next_line
    cdef uint32_t cur_line_end
    cdef uint32_t next_line_begin
    cdef uint32_t next_line_end
    cdef vector[CharGroup] buf

    buf.push_back(CharGroup(w=out.lines[0].w, h=self.font_h))
    buf.back().chars.swap(out.lines[0].chars)
    for i in range(1, out.lines.size()):
        cur_line = &buf[buf.size() - 1]
        next_line = &out.lines[i]
        cur_line_end = cur_line.chars[0].x + cur_line.w
        next_line_begin = next_line.chars[0].x
        next_line_end = next_line_begin + next_line.w

        if cur_line.chars[0].y != next_line.chars[0].y:
            pass
        elif next_line_begin > cur_line_end:
            pass
        elif next_line_end <= cur_line_end:
            # cur_line contains next_line, drop next_line
            continue
        elif cur_line.chars.size() == 1:
            kill_cur = is_killable_right(
                self, cur_line.chars[0].code, cur_line.w,
                cur_line_end - next_line_begin,
            )
            if kill_cur:
                buf.pop_back()
        elif next_line.chars.size() == 1:
            kill_next = is_killable_left(
                self, next_line.chars[0].code, next_line.w,
                cur_line_end - next_line_begin,
            )
            if kill_next:
                continue

        buf.push_back(CharGroup(w=out.lines[i].w, h=self.font_h))
        buf.back().chars.swap(out.lines[i].chars)
    out.lines.swap(buf)

    # TODO: add vertical lines and other spaces
    # add spaces
    buf.clear()
    cdef vector[uint32_t] is_sp     # the number of following spaces
    is_sp.resize(out.lines.size())
    i = 0
    while i + 1 < out.lines.size():
        x = out.lines[i].chars[0].x + out.lines[i].w
        y = out.lines[i].chars[0].y
        is_sp[i] = out.lines[i + 1].chars[0].y == y and out.lines[i + 1].chars[0].x > x
        is_sp[i] = is_sp[i] and (out.lines[i + 1].chars[0].x - x) % self.space_w == 0
        is_sp[i] = is_sp[i] and is_space(pixels, w, h, x, y, out.lines[i + 1].chars[0].x - x, self.font_h)
        if is_sp[i]:
            is_sp[i] = (out.lines[i + 1].chars[0].x - x) // self.space_w
        i += 1

    i = 0
    while i < out.lines.size():
        j = i
        while j < out.lines.size() and is_sp[j]:
            j += 1
        if j < out.lines.size():
            j += 1

        buf.push_back(CharGroup(w=out.lines[i].w, h=self.font_h))
        cur_line = &buf[buf.size() - 1]
        cur_line.chars.swap(out.lines[i].chars)
        for cur in range(i + 1, j):
            # pad spaces
            for _ in range(is_sp[cur - 1]):
                cur_line.chars.push_back(CharPos(
                    x=cur_line.chars[0].x + cur_line.w, y=cur_line.chars[0].y, w=self.space_w, code=32,
                ))
                cur_line.w += self.space_w
            # the next piece
            next_line = &out.lines[cur]
            cur_line.chars.insert(cur_line.chars.end(), next_line.chars.begin(), next_line.chars.end())

        i = j
    out.lines.swap(buf)

    if self.fix_Il:
        for i in range(out.lines.size()):
            fix_Il(out.lines[i])

    return out


cdef bool is_killable_left(_CPPFont *self, uint32_t code, uint32_t fw, uint32_t overlap_len):
    # assert overlap_len < fw
    cdef uint8_t *bits = <uint8_t *>self.code2bits[code].data()
    cdef uint32_t x, y
    for y in range(self.font_h):
        for x in range(overlap_len, fw):
            if bits[fw * y + x]:
                return False
    return True


cdef bool is_killable_right(_CPPFont *self, uint32_t code, uint32_t fw, uint32_t overlap_len):
    # assert overlap_len < fw
    cdef uint8_t *bits = <uint8_t *>self.code2bits[code].data()
    cdef uint32_t x, y
    for y in range(self.font_h):
        for x in range(fw - overlap_len):
            if bits[fw * y + x]:
                return False
    return True


cdef bool is_space(
    uint32_t *pixels, uint32_t w, uint32_t h,
    uint32_t x, uint32_t y, uint32_t space_w, uint32_t space_h,
):
    cdef uint32_t i
    for i in range(x + 1, x + space_w):
        if pixels[w * y + x] != pixels[w * y + i]:
            return False
    for i in range(y + 1, y + space_h):
        if 0 != memcmp(&pixels[w * y + x], &pixels[w * i + x], 4 * space_w):
            return False
    return True


cdef class Font:
    cdef _CPPFont *wrapped

    def __cinit__(self):
        self.wrapped = new _CPPFont()

    def __dealloc__(self):
        del self.wrapped

    cpdef void load_bin_file(self, filename):
        load_bin_file(self.wrapped, filename)

    cpdef CharResult run(self, uintptr_t pixels, uint32_t w, uint32_t h):
        return run(self.wrapped, <uint32_t *>pixels, w, h)

    cpdef LineResult make_lines(self, uintptr_t pixels, uint32_t w, uint32_t h, CharResult cr):
        return make_lines(self.wrapped, <uint32_t *>pixels, w, h, cr)


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

    result = font.run(<uintptr_t>pixels.data(), w, h)

    print('stats_lookup', result.stats_lookup)
    print('stats_node', result.stats_node)
    print('stats_check', result.stats_check)
    print('stats_hit', result.stats_hit)
    print('stats_pos_total', result.stats_pos_total)
    print()
    print('stats_time_vhash_us', result.stats_time_vhash_us)
    print('stats_time_loop_us', result.stats_time_loop_us)

    line_result = font.make_lines(<uintptr_t>pixels.data(), w, h, result)
    for x, y, w, h, s in line_result.tolist():
        print(y, x, w, s)


if __name__ == '__main__':
    main()
