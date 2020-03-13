#pragma once

#include <utility>
#include <vector>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** \brief Dictionary that stores the insertion order, similar to Python's OrderedDict.
 
    Optimized for small dictionaries: the storage is flat and uses O(N) lookup.
 
    Note: iterators are NOT permanent! They might be invalided after any
    addition or removal of elements. Apart from that, the interface tries to
    follow std::map interface as much as possible.
 */
template <typename Key, typename T>
class FlatOrderedDict
{
public:
    using key_type    = Key;
    using mapped_type = T;
    using value_type  = std::pair<Key, T>;

private:
    using container_type = std::vector<value_type>;

public:
    using iterator       = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    FlatOrderedDict() noexcept {}
    FlatOrderedDict(const FlatOrderedDict&) = default;
    FlatOrderedDict(FlatOrderedDict&&)      = default;
    FlatOrderedDict& operator=(const FlatOrderedDict&) = default;
    FlatOrderedDict& operator=(FlatOrderedDict&&) = default;

    // Note: std::initializer_list does not support move...
    FlatOrderedDict(std::initializer_list<value_type> list) : c_(list) {}

    iterator       begin() noexcept { return c_.begin(); }
    const_iterator begin() const noexcept { return c_.begin(); }
    iterator       end() noexcept { return c_.end(); }
    const_iterator end() const noexcept { return c_.end(); }

    bool empty() const noexcept { return c_.empty(); }
    size_t size() const noexcept { return c_.size(); }
    void reserve(size_t size) { c_.reserve(size); }

    /// Returns true if the container contains the given key, otherwise false.
    bool contains(const Key& key) const { return find(key) != end(); }

    iterator find(const Key& key)
    {
        iterator it = begin();
        iterator e  = end();
        for (; it != e; ++it)
            if (it->first == key)
                break;
        return it;
    }
    const_iterator find(const Key& key) const
    {
        const_iterator it = begin();
        const_iterator e  = end();
        for (; it != e; ++it)
            if (it->first == key)
                break;
        return it;
    }

    /// Insert an element if it doesn't already exist.
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args)
    {
        c_.emplace_back(std::forward<Args>(args)...);
        iterator it = find(c_.back().first);
        if (it == c_.end() - 1) {
            return {it, true};
        } else {
            c_.pop_back();
            return {it, false};
        }
    }

    template <typename... Args>
    void try_emplace(Key&& key, Args&&... args)
    {
        if (!contains(key))
            c_.emplace_back(std::move(key), T{std::forward<Args>(args)...});
    }

    template <typename... Args>
    void try_emplace(const Key& key, Args&&... args)
    {
        if (!contains(key))
            c_.emplace_back(key, T{std::forward<Args>(args)...});
    }

    void insert(value_type &&value)
    {
        if (!contains(value.first))
            c_.emplace_back(std::move(value));
    }

    void insert_or_assign(Key key, T t)
    {
        iterator it = find(key);
        if (it == end())
            c_.emplace_back(std::move(key), std::move(t));
        else
            it->second = std::move(t);
    }

    void insert_or_assign(std::initializer_list<value_type> list)
    {
        for (const value_type& value : list)
            insert_or_assign(value.first, value.second);
    }

    /// Insert without checking if the key exists or not.
    void unsafe_insert(Key key, T t)
    {
        c_.emplace_back(std::move(key), std::move(t));
    }

    /// Insert without checking if the key exists or not.
    void unsafe_insert(const_iterator it, Key key, T t)
    {
        c_.emplace(it, std::move(key), std::move(t));
    }

    void erase(const_iterator it) { c_.erase(it); }
    void erase(const Key& key)
    {
        auto it = find(key);
        if (it != end())
            c_.erase(it);
    }

private:
    container_type c_;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
