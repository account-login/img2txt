#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <utility>


typedef uint64_t (*__Hasher)(uint64_t);


// NOTE: the default constructed key is treated as invalid key
template <class K, class V, __Hasher H>
struct htab {
    typedef std::pair<K, V> value_type;

    value_type *_slots = NULL;
    size_t _mask = 0;
    size_t _size = 0;
    uint32_t _bound = 0;
    float _max_factor = 0.7;

    ~htab() {
        clear();
    }

    size_t bucket_count() const {
        return _mask ? _mask + 1 : 0;
    }

    size_t size() const {
        return _size;
    }

    bool empty() const {
        return 0 == size();
    }

    float max_load_factor() const {
        return _max_factor;
    }
    void max_load_factor(float factor) {
        assert(factor < 0.99);
        _max_factor = factor;
    }

    float load_factor() const {
        return (float)_size / (_mask + 1);
    }

    size_t count(const K &key) const {
        return !!get(key);
    }

    void clear() {
        for (size_t i = 0; i < _mask + 1 && _mask > 0; ++i) {
            _slots[i].~value_type();
        }
        free(_slots);
        _slots = NULL;
        _mask = _bound = _size = 0;
    }

    V &operator[](const K &key) {
        return get_mut(key);
    }

    // TODO: iterator
    typedef value_type *iterator;

    value_type *end() const {
        return NULL;
    }

    value_type *find(const K &key) const {
        V *value = get(key);
        if (!value) {
            return NULL;
        }
        return (value_type *)(((char *)(void *)value) - offsetof(value_type, second));
    }

    void _rehash() {
        // init new slots
        size_t new_cap = (_mask + 1) * 2;
        size_t new_mask = new_cap - 1;
        value_type *new_slots = (value_type *)malloc(sizeof(value_type) * new_cap);
        for (size_t i = 0; i < new_cap; ++i) {
            new (new_slots + i) value_type();
        }

        // migrate
        uint32_t max_l = 0;
        for (size_t i = 0; i < _mask + 1 && _size > 0; ++i) {
            if (_slots[i].first == K()) {
                continue;
            }

            uint32_t l = 0;
            size_t h = H(_slots[i].first);
            while (true) {
                l++;
                assert(l < new_cap);    // FIXME: infinite loop
                if (new_slots[h & new_mask].first == K()) {
                    std::swap(new_slots[h & new_mask].first, _slots[i].first);
                    std::swap(new_slots[h & new_mask].second, _slots[i].second);
                    break;
                }
                h = H(h + 1);
            }
            if (l > max_l) {
                max_l = l;
            }
        }

        // replace old slots
        for (size_t i = 0; i < _mask + 1 && _mask > 0; ++i) {
            _slots[i].~value_type();
        }
        free(_slots);
        _slots = new_slots;

        // mask and bound
        _mask = new_mask;
        _bound = __builtin_popcountll(_mask);
        if (_bound < max_l) {
            _bound = max_l;
        }
        if (_bound < 4) {
            _bound = 4;
        }
    }

    V *get(const K &key) const {
        if (key == K()) {
            return NULL;
        }

        size_t h = H(key);
        for (uint32_t ntry = 0; ntry < _bound; ++ntry) {
            if (_slots[h & _mask].first == key) {
                return &_slots[h & _mask].second;
            }
            if (_slots[h & _mask].first == K()) {
                return NULL;
            }
            h = H(h + 1);
        }
        return NULL;
    }

    V &get_mut(const K &key) {
        assert(key != K());

        if (_mask * _max_factor <= _size + 1) {
            _rehash();
        }

        size_t h = H(key);
        for (uint32_t ntry = 0; ntry < _bound; ++ntry) {
            if (_slots[h & _mask].first == K()) {
                _slots[h & _mask].first = key;      // new
                _size += 1;
                break;
            }
            if (_slots[h & _mask].first == key) {
                break;
            }
            h = H(h + 1);
        }

        if (_slots[h & _mask].first != key) {
            _rehash();
            return get_mut(key);    // FIXME: infinite loop
        }

        return _slots[h & _mask].second;
    }

};
