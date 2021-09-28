import ctypes
import struct
import win32con
import win32gui
from collections import defaultdict


# https://stackoverflow.com/questions/5750887/python-use-windows-api-to-render-text-using-a-ttf-font
def native_bmp_to_pixel(hdc, bitmap_handle, width, height):
    bmpheader = struct.pack("LHHHH", struct.calcsize("LHHHH"), width, height, 1, 24)
    c_bmpheader = ctypes.c_buffer(bmpheader)
    row_size = (width * 3 + 3) & -4     # padded to 4 bytes
    c_bits = ctypes.c_buffer(b" " * (height * row_size))
    res = ctypes.windll.gdi32.GetDIBits(hdc, bitmap_handle, 0, height, c_bits, c_bmpheader, win32con.DIB_RGB_COLORS)
    assert res
    mat = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            off = y * row_size + 3 * x
            b, g, r = c_bits[off:off+3]
            mat[height - y - 1][x] = (b << 16) | (g << 8) | r
    return mat


class Win32Font:
    def __init__(self, name, height, weight=win32con.FW_NORMAL, italic=False, underline=False):
        font = win32gui.LOGFONT()
        font.lfHeight = height
        font.lfWeight = weight
        font.lfItalic = italic
        font.lfUnderline = underline
        font.lfQuality = win32con.NONANTIALIASED_QUALITY
        font.lfFaceName = name
        self.hf = win32gui.CreateFontIndirect(font)

        self.dc = win32gui.CreateCompatibleDC(None)
        win32gui.SelectObject(self.dc, self.hf)
        win32gui.SetTextColor(self.dc, 0)
        win32gui.SetBkMode(self.dc, win32con.OPAQUE)

    def render(self, text):
        _, (_, _, w, h) = win32gui.DrawText(
            self.dc, text, -1, (0, 0, 0, 0), win32con.DT_LEFT | win32con.DT_NOPREFIX | win32con.DT_CALCRECT,
        )
        if w == 0:
            return None

        bitmap = win32gui.CreateCompatibleBitmap(self.dc, w, h)
        try:
            win32gui.SelectObject(self.dc, bitmap)
            win32gui.DrawText(self.dc, text, -1, (0, 0, w, h), win32con.DT_LEFT | win32con.DT_NOPREFIX)
            return native_bmp_to_pixel(self.dc, bitmap.handle, w, h)
        finally:
            win32gui.DeleteObject(bitmap.handle)

    def __del__(self):
        win32gui.DeleteDC(self.dc)
        win32gui.DeleteObject(self.hf)


def matrix_simplify(mat):
    w, h = len(mat[0]), len(mat)
    bits = tuple(0 if x else 1 for row in mat for x in row)
    return w, h, bits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, help='font family')
    ap.add_argument('--height', required=True, type=int, help='font height')
    args = ap.parse_args()

    f = Win32Font(args.family, -args.height)

    code2mat = [None] * 0x10000
    for code in range(0x10000):
        mat = f.render(chr(code))
        if mat is None:
            continue
        code2mat[code] = matrix_simplify(mat)

    _, common_height, _ = code2mat[ord('a')]
    for code, mat in enumerate(code2mat):
        if mat is None:
            continue
        w, h, bits = mat
        assert h == common_height, 'same height'
        # ones = sum(bits)
        # if ones == 0 or ones == len(bits):
        #     # print('remove full block', code)
        #     code2mat[code] = None

    code2obj = [None] * 0x10000
    for code, mat in enumerate(code2mat):
        if mat is None:
            continue
        w, h, bits = mat
        bits_str = ''.join('1' if x else '0' for x in bits)
        code2obj[code] = (w, h, bits_str)

    import json
    import sys
    json.dump(code2obj, sys.stdout, indent=2)


if __name__ == '__main__':
    main()


# f = Win32Font("Microsoft Yahei", -14)
# f = Win32Font("SimSun", -14)
# # mat = f.render("窗1 谢ƀƀƀ")

# # mat = f.render('ĨĨaĨaa&aa')
# # for row in mat:
# #     for v in row:
# #         print('口' if v else '  ', end='')
# #     print()
# # print(len(row))


# def matrix_simplify(mat):
#     w, h = len(mat[0]), len(mat)
#     bits = tuple(0 if x else 1 for row in mat for x in row)
#     return w, h, bits


# code2mat = [None] * 0x10000
# for code in range(0x10000):
#     mat = f.render(chr(code))
#     if mat is None:
#         continue
#     code2mat[code] = matrix_simplify(mat)


# _, common_height, _ = code2mat[ord('a')]
# for code, mat in enumerate(code2mat):
#     if mat is None:
#         continue
#     w, h, bits = mat
#     assert h == common_height, 'same height'
#     ones = sum(bits)
#     if ones == 0 or ones == len(bits):
#         # print('remove full block', code)
#         code2mat[code] = None


# def zero_line(bits, w, y):
#     for x in range(w):
#         if bits[y * w + x]:
#             return False
#     return True


# def bits_spacing(bits, w, h):
#     is_zero = [zero_line(bits, w, y) for y in range(h)]
#     for top in range(h):
#         if not is_zero[top]:
#             break
#     for bot in reversed(range(h)):
#         if not is_zero[bot]:
#             break
#     return top, h - bot - 1


# def print_bits(bits, w, h):
#     for y in range(h):
#         print('|', end='')
#         for x in range(w):
#             print('  ' if bits[w * y + x] else '口', end='')
#         print('|')


# # # chars that uses full height
# # for code, mat in enumerate(code2mat):
# #     if mat is None:
# #         continue
# #     w, h, bits = mat
# #     top, bot = bits_spacing(bits, w, h)
# #     if top + bot == 0:
# #         print(code, w, h, top, bot)
# #         print_bits(bits, w, h)


# # # cut extra spacing on top and bottom
# # s_top, s_bot = 0, 0
# # for code, mat in enumerate(code2mat):
# #     if mat is None:
# #         continue
# #     w, h, bits = mat
# #     assert h == code2mat[0][1], 'same height'

# #     for y in range(h):
# #         if not zero_line(bits, w, y):
# #             break
# #     s_top = min(s_top, y)

# #     for y in reversed(range(h)):
# #         if not zero_line(bits, w, y):
# #             break
# #     s_bot = max(s_bot, y)

# # for code, mat in enumerate(code2mat):
# #     if mat is None:
# #         continue
# #     w, h, bits = mat
# #     bits = tuple(bits[y * w + x] for y in range(s_top, s_bot + 1) for x in range(w))
# #     h = s_bot - s_top + 1
# #     assert w * h == len(bits)
# #     code2mat[code] = (w, h, bits)

# # print('h', h)


# # map duplicated bits
# bits2code = dict()

# for code, mat in enumerate(code2mat):
#     if mat is None:
#         continue

#     w, h, bits = mat
#     if bits not in bits2code:
#         bits2code[bits] = []
#     bits2code[bits].append(code)


# # width distribution
# dim_stats = dict()
# for bits in bits2code.keys():
#     w = len(bits) // common_height
#     if w not in dim_stats:
#         dim_stats[w] = 0
#     dim_stats[w] += 1

# print(dim_stats)


# # hash
# def bits_hash(bits, w, h):
#     hcode = 0
#     for y in range(h):
#         row = 0
#         for x in range(w):
#             row <<= 1
#             if bits[w * y + x]:
#                 row ^= 0b01
#             else:
#                 row ^= 0b10
#             # row |= bits[w * y + x]
#         hcode <<= 1
#         hcode ^= row
#     # hcode &= (1 << (w + h - 1)) - 1
#     return hcode


# def bits_hash(bits, w, h):
#     hcode = 0
#     for x in range(w):
#         col = 0
#         for y in range(h):
#             col <<= 1
#             if bits[w * y + x]:
#                 col ^= 0b01
#             else:
#                 col ^= 0b10
#             # col |= bits[w * y + x]
#         hcode <<= 1
#         hcode ^= col
#     # hcode &= (1 << (w + h - 1)) - 1
#     return hcode


# hash2bits = dict()
# for bits in bits2code.keys():
#     w = len(bits) // common_height
#     hcode = bits_hash(bits, w, h)
#     if hcode not in hash2bits:
#         hash2bits[hcode] = []
#     hash2bits[hcode].append(bits)
# # for hcode in sorted(hash2bits.keys()):
# #     print(hcode, len(hash2bits[hcode]))

# col_cnt = defaultdict(int)
# for collision in map(len, hash2bits.values()):
#     col_cnt[collision] += 1
# print(col_cnt)
