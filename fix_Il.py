import sys
import json


w2f = dict()

for line in sys.stdin:
    _, word, freq = line.split()
    freq = float(freq)
    if word == word.capitalize():
        continue
    if len(word) <= 1:
        continue

    word = word.lower()
    assert word not in w2f
    w2f[word] = freq


to_fix = []
for word, freq in w2f.items():
    if 'l' not in word:
        continue

    before_fix = word.replace('l', 'i')
    if word[0] == 'l':
        if w2f.get(before_fix, 0) >= freq:
            print('// no fix', word, freq, before_fix, w2f[before_fix])
            continue

    before_fix = word.replace('l', 'I')
    to_fix.append(before_fix)
    title = before_fix[0].upper() + before_fix[1:]
    if before_fix != title:
        to_fix.append(title)

print()
print('#include <string>')
print('#include <unordered_set>')
print()

# NOTE: this segfaults g++ and clang++ on cygwin
# print('static const std::unordered_set<std::string> g_Il_fix = {')
# for word in to_fix:
#     print('    ' + json.dumps(word) + ',')
# print('};')

print('static const char *g_Il_fix_list[] = {')
for word in to_fix:
    print('    ' + json.dumps(word) + ',')
print('};')

print(r'''
std::unordered_set<std::string> __init_g_Il_fix() {
    std::unordered_set<std::string> m;
    size_t n = sizeof(g_Il_fix_list) / sizeof(g_Il_fix_list[0]);
    for (size_t i = 0; i < n; ++i) {
        m.insert(g_Il_fix_list[i]);
    }
    return m;
}

static const std::unordered_set<std::string> g_Il_fix = __init_g_Il_fix();
''')
