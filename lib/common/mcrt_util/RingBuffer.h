// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include "CPPSupport.h"
#include "Finally.h"
#include "Wait.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

enum class ConsumerTraits
{
    MULTIPLE_CONSUMERS,
    SINGLE_CONSUMER
};

enum class ProducerTraits
{
    MULTIPLE_PRODUCERS,
    SINGLE_PRODUCER
};

enum class LockTraits
{
    AUTO_DETECT,
    SPIN_LOCK,
    LOCK_FREE
};

struct DefaultRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::MULTIPLE_PRODUCERS;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::MULTIPLE_CONSUMERS;
    static constexpr LockTraits     lock_traits     = LockTraits::AUTO_DETECT;
};

template <typename T>
constexpr bool can_use_type_noexcept()
{
    return std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value;
}

template <typename T, LockTraits input_lock_traits>
struct GetLockTraits
{
    static constexpr LockTraits traits = input_lock_traits;
};

template <typename T>
struct GetLockTraits<T, LockTraits::AUTO_DETECT>
{
    // We prefer to use a spin lock when we can, because it has better worse-case performance. We use lock-free when
    // we can through in the move operations.
    static constexpr LockTraits traits = (can_use_type_noexcept<T>()) ? LockTraits::SPIN_LOCK : LockTraits::LOCK_FREE;
};

template <typename T, std::size_t log_n_elements, typename Traits>
class RingBufferImpl
{
    enum class occupied_t : std::int_least8_t
    {
        UNOCCUPIED,
        IN_TRANSITION,
        OCCUPIED,
        EXCEPTION
    };

    using index_t                       = std::uint32_t;
    using atomic_index_t                = std::atomic<index_t>;
    using atomic_occupied_t             = std::atomic<occupied_t>;
    static constexpr index_t k_capacity = 1 << log_n_elements;

    enum class PopImplementation
    {
        MOVE,
        PLACEMENT
    };

    static constexpr LockTraits k_lock_traits = GetLockTraits<T, Traits::lock_traits>::traits;

public:
    using value_type = T;

    RingBufferImpl() noexcept
    : m_size(0)
    , m_read_idx(0)
    , m_write_idx(0)
    , m_nodes(static_cast<Node*>(scene_rdl2::util::alignedMalloc(sizeof(Node) * k_capacity)))
    {
        for (index_t i = 0; i < k_capacity; ++i) {
            new (m_nodes + i) Node{};
        }

        // Check for lock-free types. atomic_occupied_t can easily be changed to an integer type that supports lock-free
        // atomic operations.
#if defined(__cpp_lib_atomic_is_always_lock_free)
        static_assert(atomic_index_t::is_always_lock_free, "Expecting lock-free types");
        static_assert(atomic_occupied_t::is_always_lock_free, "Expecting lock-free types");
#endif
        MNRY_ASSERT(m_size.is_lock_free());
        MNRY_ASSERT(m_read_idx.is_lock_free());
        MNRY_ASSERT(m_write_idx.is_lock_free());
        MNRY_ASSERT(m_nodes[0].m_occupied.is_lock_free());
    }

    ~RingBufferImpl()
    {
        // Instead of just calling pop() until we're empty, we do it manually to save some atomic operations.
        index_t remaining = size();
        index_t idx       = m_read_idx.load(std::memory_order_relaxed);
        while (remaining > 0) {
            const auto occupied_status = m_nodes[idx].m_occupied.load(std::memory_order_relaxed);
            MNRY_ASSERT(occupied_status != occupied_t::IN_TRANSITION);
            if (occupied_status == occupied_t::OCCUPIED) {
                get_pointer(idx)->~T();
                --remaining;
            }
            idx = increment(idx);
        }
        for (index_t i = 0; i < k_capacity; ++i) {
            (m_nodes + i)->~Node();
        }
        scene_rdl2::util::alignedFree(m_nodes);
    }

    static constexpr LockTraits get_lock_traits() noexcept
    {
        return k_lock_traits;
    }

    NO_DISCARD bool empty() const noexcept
    {
        return m_size == 0;
    }

    NO_DISCARD bool full() const noexcept
    {
        return m_size == k_capacity;
    }

    NO_DISCARD index_t size() const noexcept
    {
        return m_size;
    }

    static constexpr index_t capacity() noexcept
    {
        return k_capacity;
    }

    NO_DISCARD bool try_push(const T& t)
    {
        return try_push_impl(t);
    }

    NO_DISCARD bool try_push(T&& t)
    {
        return try_push_impl(std::move(t));
    }

    void push(const T& t)
    {
        push_impl(t);
    }

    void push(T&& t)
    {
        push_impl(std::move(t));
    }

    template <typename Iterator>
    NO_DISCARD bool try_push_batch(Iterator first, Iterator last)
    {
        static_assert(Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER,
                      "Batch mode supported for single producer only");
        index_t throwaway;
        return try_push_batch_impl(first, last, throwaway);
    }

    template <typename Iterator>
    void push_batch(Iterator first, Iterator last)
    {
        push_batch(first, last, typename std::iterator_traits<Iterator>::iterator_category());
    }

    template <typename... Args>
    void emplace(Args&&... args)
    {
        constexpr bool no_except_construct = noexcept(T(std::forward<Args>(args)...));
        // cppcheck-suppress syntaxError
        IF_CONSTEXPR (k_lock_traits == LockTraits::SPIN_LOCK && !no_except_construct) {
            // We don't have a way to check which constructor is called by emplace for our class static exception
            // checks used to determine our lock traits. We only check exception safety in move construction and move
            // assignment when we check our top-level traits. Check here, and if it can throw, create our type before we
            // call push in an effort to use the move operations.
            push_impl(T(std::forward<Args>(args)...));
        } else {
            push_impl(std::forward<Args>(args)...);
        }
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    NO_DISCARD bool try_pop(T& ret)
    {
        IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::SINGLE_CONSUMER &&
                      k_lock_traits == LockTraits::SPIN_LOCK) {
            return try_pop_single_impl<PopImplementation::MOVE>(ret);
        } else IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::SINGLE_CONSUMER &&
                   k_lock_traits == LockTraits::LOCK_FREE) {
            return try_pop_single_lock_free_impl<PopImplementation::MOVE>(ret);
        } else IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::MULTIPLE_CONSUMERS &&
                   k_lock_traits == LockTraits::SPIN_LOCK) {
            return try_pop_impl<PopImplementation::MOVE>(ret);
        } else {
            return try_pop_lock_free_impl<PopImplementation::MOVE>(ret);
        }
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    T pop()
    {
        // We don't want to make a default constructor a requirement of T, so we just allocate stack space for it, and
        // tell our pop function to do placement new for a copy constructor.
        storage_t storage;
        T* const  p = reinterpret_cast<T*>(std::addressof(storage));
        do_pop_dispatch<PopImplementation::PLACEMENT>(*p);
        T ret{ std::move(*p) };
        p->~T();
        return ret;
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    void pop(T& ret)
    {
        do_pop_dispatch<PopImplementation::MOVE>(ret);
    }

private:
    template <typename... Args>
    void do_construct(index_t write_idx, Args&&... args)
    {
        new (get_pointer(write_idx)) T(std::forward<Args>(args)...);
    }

    // Sometimes our forwarded arguments are actually the type itself (as opposed to arguments for creating the type).
    // Do a move if we can, but if we throw, we don't want to lose the data the user is passing in.
    void do_construct(index_t write_idx, T&& t)
    {
        new (get_pointer(write_idx)) T(std::move_if_noexcept(t));
    }

    template <PopImplementation pop_implementation>
    void do_pop_dispatch(T& ret)
    {
        bool popped;
        do {
            IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::SINGLE_CONSUMER &&
                          k_lock_traits == LockTraits::SPIN_LOCK) {
                popped = try_pop_single_impl<pop_implementation>(ret);
            } else IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::SINGLE_CONSUMER &&
                       k_lock_traits == LockTraits::LOCK_FREE) {
                popped = try_pop_single_lock_free_impl<pop_implementation>(ret);
            } else IF_CONSTEXPR (Traits::consumer_traits == ConsumerTraits::MULTIPLE_CONSUMERS &&
                       k_lock_traits == LockTraits::SPIN_LOCK) {
                popped = try_pop_impl<pop_implementation>(ret);
            } else {
                popped = try_pop_lock_free_impl<pop_implementation>(ret);
            }
            if (!popped) {
                wait(m_size, 0u);
            }
        } while (!popped);
    }

    template <typename... Args>
    void push_impl(Args&&... args)
    {
        bool pushed;
        do {
            // Does it make you nervous to see std::forward in a loop? I understand. The contents, however, will not be
            // consumed unless the push succeeds (i.e. there will be no moves until success).
            IF_CONSTEXPR (Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER &&
                          k_lock_traits == LockTraits::SPIN_LOCK) {
                pushed = try_push_single_impl(std::forward<Args>(args)...);
            } else IF_CONSTEXPR (Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER &&
                       k_lock_traits == LockTraits::LOCK_FREE) {
                pushed = try_push_single_lock_free_impl(std::forward<Args>(args)...);
            } else IF_CONSTEXPR (Traits::producer_traits == ProducerTraits::MULTIPLE_PRODUCERS &&
                       k_lock_traits == LockTraits::SPIN_LOCK) {
                pushed = try_push_impl(std::forward<Args>(args)...);
            } else {
                pushed = try_push_lock_free_impl(std::forward<Args>(args)...);
            }
            if (!pushed) {
                wait(m_size, k_capacity);
            }
        } while (!pushed);
    }

    // This version of push assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <typename... Args>
    NO_DISCARD bool try_push_impl(Args&&... t)
    {
        index_t size = m_size;
        if (size == k_capacity) {
            return false;
        }
        // Increment size counter. This allows us to reserve our space before any changes take place.
        // A failed CAS may set size to capacity, so check for a full buffer again in loop.
        while (!m_size.compare_exchange_weak(size, size + 1u)) {
            if (size == k_capacity) {
                return false;
            }
        }

        // We know we have space (and have reserved it) at this point. We just have to grab our write index.

        // Get and increment write pos.
        index_t write_idx = m_write_idx;
        while (!m_write_idx.compare_exchange_weak(write_idx, increment(write_idx)))
            ;

        MNRY_ASSERT(write_idx < k_capacity);

        // Wait until our slot isn't occupied, and then mark it as in transition so that:
        // A) other writers know they can't use this index.
        // B) readers know they can't read from this index yet.
        //
        // What guarantees that our slot doesn't get stolen by another writer between obtaining our index and marking it
        // as in transition? Nothing. It's completely possible that we have wrapped around the buffer and that a later
        // thread is writing to our slot, but they will have to mark the slot as in transition or occupied, stopping us
        // from writing to it at the same point, and it will become available once a reader clears it up.
        //
        // What guarantees us that a reader will read from our slot if another writer has stolen it? The read index will
        // have to be incremented. If we have looped so that there are two writes, we will eventually loop so that there
        // are two reads.
        occupied_t occupied_value = occupied_t::UNOCCUPIED;
        while (!m_nodes[write_idx].m_occupied.compare_exchange_weak(occupied_value, occupied_t::IN_TRANSITION)) {
            // We only want to update this when it's unoccupied, so reset our expected value.
            wait(m_nodes[write_idx].m_occupied, occupied_value);
            occupied_value = occupied_t::UNOCCUPIED;
        }

        // Construct
        do_construct(write_idx, std::forward<Args>(t)...);

        // Inform readers that this slot if valid and can be read from.
        m_nodes[write_idx].m_occupied = occupied_t::OCCUPIED;
        // We may notify another writer after we've filled it up again, which does us no good.
        notify_all(m_size);
        notify_all(m_nodes[write_idx].m_occupied);
        return true;
    }

    // This version of push handles the class throwing exceptions while still maintaining lock-free semantics.
    template <typename... Args>
    NO_DISCARD bool try_push_lock_free_impl(Args&&... t)
    {
        index_t size = m_size;
        if (size == k_capacity) {
            return false;
        }
        // Increment size counter. This allows us to reserve our space before any changes take place.
        // A failed CAS may set size to capacity, so check for a full buffer again in loop.
        while (!m_size.compare_exchange_weak(size, size + 1u)) {
            if (size == k_capacity) {
                return false;
            }
        }

        // We know we have space (and have reserved it) at this point. We just have to grab our write index.

        while (true) {
            // Get and increment write pos.
            index_t write_idx = m_write_idx;
            while (!m_write_idx.compare_exchange_weak(write_idx, increment(write_idx)))
                ;

            MNRY_ASSERT(write_idx < k_capacity);

            // Wait until our slot isn't occupied, and then mark it as in transition so that:
            // A) other writers know they can't use this index.
            // B) readers know they can't read from this index yet.
            //
            // What guarantees that our slot doesn't get stolen by another writer between obtaining our index and
            // marking it as in transition? Nothing. It's completely possible that we have wrapped around the buffer and
            // that a later thread is writing to our slot, but they will have to mark the slot as in transition or
            // occupied, stopping us from writing to it at the same point, and it will become available once a reader
            // clears it up.
            //
            // What guarantees us that a reader will read from our slot if another writer has stolen it? The read index
            // will have to be incremented. If we have looped so that there are two writes, we will eventually loop so
            // that there are two reads.
            occupied_t occupied_value = occupied_t::UNOCCUPIED;
            if (m_nodes[write_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::IN_TRANSITION)) {
                auto cleanup = finally([this, write_idx] {
                    // We may notify another writer after we've filled it up again, which does us no good, so we notify
                    // all threads.
                    notify_all(m_size);
                    notify_all(m_nodes[write_idx].m_occupied);
                });

                try {
                    // Construct
                    do_construct(write_idx, std::forward<Args>(t)...);
                } catch (...) {
                    m_nodes[write_idx].m_occupied = occupied_t::EXCEPTION;
                    throw;
                }

                // Inform readers that this slot if valid and can be read from.
                m_nodes[write_idx].m_occupied = occupied_t::OCCUPIED;
                return true;
            }
        }
    }

    // Special case for a single producer.
    // This version of push assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <typename... Args>
    NO_DISCARD bool try_push_single_impl(Args&&... t)
    {
        // We're the only writer. m_size will only decrease.
        if (m_size == k_capacity) {
            return false;
        }

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for size or for the data to be ready.
        ++m_size;

        // Get and increment write pos.
        const index_t write_idx =
            m_write_idx.exchange(increment(m_write_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
        MNRY_ASSERT(write_idx < k_capacity);

        // Wait until it's something other than occupied.
        // Can't atomic::wait here because we need to wait if we're either occupied or in transition
        while (m_nodes[write_idx].m_occupied != occupied_t::UNOCCUPIED) {
            do_pause();
        }

        // Construct
        do_construct(write_idx, std::forward<Args>(t)...);

        // Inform readers that this slot if valid and can be read from.
        m_nodes[write_idx].m_occupied = occupied_t::OCCUPIED;

        // We only need to wake up one thread, but we have two wait variables, and we can't control which thread gets
        // the notification. Inform one thread that the size has changed, and inform all threads (including the one that
        // knows the size has changed) that the status has changed
        notify_one(m_size);
        notify_all(m_nodes[write_idx].m_occupied);
        return true;
    }

    // Special case for a single producer.
    // This version of push handles the class throwing exceptions while still maintaining lock-free semantics.
    template <typename... Args>
    NO_DISCARD bool try_push_single_lock_free_impl(Args&&... t)
    {
        // We're the only writer. m_size will only decrease.
        if (m_size == k_capacity) {
            return false;
        }

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for size or for the data to be ready.
        ++m_size;

        while (true) {
            // Get and increment write pos.
            const index_t write_idx =
                m_write_idx.exchange(increment(m_write_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
            MNRY_ASSERT(write_idx < k_capacity);

            occupied_t occupied_value = occupied_t::UNOCCUPIED;
            if (m_nodes[write_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::IN_TRANSITION)) {
                auto cleanup = finally([this, write_idx] {
                    // There are only other readers, so we don't need to worry about waking another writer.
                    // We only need to wake up one thread, but we have two wait variables, and we can't control which
                    // thread gets the notification. Inform one thread that the size has changed, and inform all threads
                    // (including the one that knows the size has changed) that the status has changed
                    notify_one(m_size);
                    notify_all(m_nodes[write_idx].m_occupied);
                });

                try {
                    // Construct
                    do_construct(write_idx, std::forward<Args>(t)...);
                } catch (...) {
                    m_nodes[write_idx].m_occupied = occupied_t::EXCEPTION;
                    throw;
                }

                // Inform readers that this slot if valid and can be read from.
                m_nodes[write_idx].m_occupied = occupied_t::OCCUPIED;
                return true;
            }
        }
    }

    template <typename Iterator>
    void push_batch(Iterator first, Iterator last, std::input_iterator_tag)
    {
        static_assert(Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER,
                      "Batch mode supported for single producer only");

        // With input iterators, we can't count, and we can't go over the values again. The best we can do is add them
        // one at a time.
        for (; first != last; ++first) {
            push_impl(*first);
        }
    }

    // This function works for forward iterators, but is more efficient with random access iterators.
    template <typename Iterator>
    void push_batch(Iterator first, Iterator last, std::forward_iterator_tag)
    {
        static_assert(Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER,
                      "Batch mode supported for single producer only");
        index_t num_elements_to_process = std::distance(first, last);

        while (first != last) {
            // We're the only writer. m_size will only decrease, meaning this is pessimistic (which is perfect).
            const auto container_size = m_size.load(std::memory_order_acquire);
            if (container_size == capacity()) {
                wait(m_size, container_size);
            }

            const index_t mini_batch_size = std::min(num_elements_to_process, capacity() - container_size);

            MNRY_ASSERT(mini_batch_size <= static_cast<index_t>(std::distance(first, last)));
            const auto           next    = std::next(first, mini_batch_size);
            [[gnu::unused]] bool success = try_push_batch_impl(first, next, mini_batch_size);
            MNRY_ASSERT(success);
            first = next;
            num_elements_to_process -= mini_batch_size;
        }
    }

    // This version uses a(n implicit) spinlock.
    // We pass in num_elements even though it should be the difference between first and last so that we don't have to
    // iterate over [first, last] when using forward_iterators.
    // Precondition: This function should only be called when there is enough room to add num_elements.
    template <typename Iterator>
    NO_DISCARD bool try_push_batch_impl(Iterator first, Iterator last, index_t num_elements)
    {
        // TODO: C++20: This is a good case for concepts.
        static_assert(std::is_base_of<std::forward_iterator_tag,
                                      typename std::iterator_traits<Iterator>::iterator_category>::value,
                      "This only works for forward iterators or better");

        // We pass in num_elements even though it can be derived through the iterators for efficiency in the case of
        // forward iterators.
        MNRY_ASSERT(std::distance(first, last) == num_elements);

        // Since we're the one producer, this should succeed if this function is called properly.
        MNRY_ASSERT(capacity() - m_size.load() >= num_elements);

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for container_size or for the data to be ready. Doing it here means:
        // * We only pay for this atomic operation once.
        // * Consumers can get data as soon as we update the occupied state.
        // The drawback is that we block consumers (even if they do a try_pop) because they think there is data to be
        // read but the data is inaccessible.
        m_size += num_elements;
        notify_all(m_size);

        index_t write_idx = m_write_idx.load(std::memory_order_relaxed);
        for (; first != last; ++first, write_idx = increment(write_idx)) {
            MNRY_ASSERT(write_idx < k_capacity);

            // Wait until it's something other than occupied.
            // Can't atomic::wait here because we need to wait if we're either unoccupied or in transition
            while (m_nodes[write_idx].m_occupied != occupied_t::UNOCCUPIED) {
                do_pause();
            }

            // Construct
            new (get_pointer(write_idx)) T(std::move_if_noexcept(*first));

            // Inform readers that this slot is valid and can be read from.
            m_nodes[write_idx].m_occupied = occupied_t::OCCUPIED;

            // We have already informed all threads about the size change. We won't wake another writer (we're the only
            // one), so just wake one thread on the status change.
            notify_one(m_nodes[write_idx].m_occupied);
        }
        m_write_idx.store(write_idx, std::memory_order_relaxed);

        // If uploading memory, do memcpy and do a m_size.notify_all() here instead of the notify one

        return true;
    }

    // This version of pop assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <PopImplementation pop_implementation>
    NO_DISCARD bool try_pop_impl(T& ret)
    {
        // Check for empty buffer
        // Decrement counter
        index_t size = m_size;
        if (size == 0) {
            return false;
        }
        while (!m_size.compare_exchange_weak(size, size - 1u)) {
            if (size == 0) {
                return false;
            }
        }

        MNRY_ASSERT(size <= k_capacity);

        // Get and increment read pos
        index_t read_idx = m_read_idx;
        while (!m_read_idx.compare_exchange_weak(read_idx, increment(read_idx)))
            ;

        MNRY_ASSERT(read_idx < k_capacity);

        // Wait until we have valid data
        occupied_t occupied_value = occupied_t::OCCUPIED;
        while (!m_nodes[read_idx].m_occupied.compare_exchange_weak(occupied_value, occupied_t::IN_TRANSITION)) {
            wait(m_nodes[read_idx].m_occupied, occupied_value);
            occupied_value = occupied_t::OCCUPIED;
        }

        try {
            IF_CONSTEXPR (pop_implementation == PopImplementation::PLACEMENT) {
                new (std::addressof(ret)) T{ std::move_if_noexcept(*get_pointer(read_idx)) };
            } else IF_CONSTEXPR (pop_implementation == PopImplementation::MOVE) {
                ret = std::move_if_noexcept(*get_pointer(read_idx));
            }

            // Destroy
            get_pointer(read_idx)->~T();

            // Mark as not occupied
            m_nodes[read_idx].m_occupied = occupied_t::UNOCCUPIED;
            // We may notify another reader after we've emptied it up again, which does us no good.
            notify_all(m_size);
            notify_all(m_nodes[read_idx].m_occupied);
        } catch (...) {
            get_pointer(read_idx)->~T();
            m_nodes[read_idx].m_occupied = occupied_t::OCCUPIED;
            // We may notify another reader after we've emptied it up again, which does us no good.
            notify_all(m_size);
            notify_all(m_nodes[read_idx].m_occupied);
            throw;
        }
        return true;
    }

    // This version of pop handles the class throwing exceptions while still maintaining lock-free semantics.
    template <PopImplementation pop_implementation>
    NO_DISCARD bool try_pop_lock_free_impl(T& ret)
    {
        // Check for empty buffer
        // Decrement counter
        index_t size = m_size;
        if (size == 0) {
            return false;
        }
        while (!m_size.compare_exchange_weak(size, size - 1u)) {
            if (size == 0) {
                return false;
            }
        }

        MNRY_ASSERT(size <= k_capacity);

        while (true) {
            // Get and increment read pos
            index_t read_idx = m_read_idx;
            while (!m_read_idx.compare_exchange_weak(read_idx, increment(read_idx)))
                ;

            MNRY_ASSERT(read_idx < k_capacity);

            auto cleanup_function = [this, read_idx] {
                // Mark as not occupied
                m_nodes[read_idx].m_occupied = occupied_t::UNOCCUPIED;

                // We may notify another reader after we've emptied it up again, which does us no good.
                notify_all(m_size);
                notify_all(m_nodes[read_idx].m_occupied);
            };

            // Wait until we have valid data
            occupied_t occupied_value = occupied_t::OCCUPIED;
            if (m_nodes[read_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::IN_TRANSITION)) {
                auto cleanup = finally(cleanup_function);

                try {
                    IF_CONSTEXPR (pop_implementation == PopImplementation::PLACEMENT) {
                        new (std::addressof(ret)) T{ std::move_if_noexcept(*get_pointer(read_idx)) };
                    } else IF_CONSTEXPR (pop_implementation == PopImplementation::MOVE) {
                        ret = std::move_if_noexcept(*get_pointer(read_idx));
                    }

                    // Destroy
                    get_pointer(read_idx)->~T();
                    return true;
                } catch (...) {
                    get_pointer(read_idx)->~T();
                    throw;
                }
            } else if (occupied_value == occupied_t::EXCEPTION) {
                auto cleanup = finally(cleanup_function);
                return false;
            }
        }
    }

    // Special case for a single consumer.
    // This version of pop assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <PopImplementation pop_implementation>
    NO_DISCARD bool try_pop_single_impl(T& ret)
    {
        // Check for empty buffer
        // We're the only reader. m_size will only increase.
        if (m_size == 0) {
            return false;
        }

        // We only have one thread reading. We don't have to decrement this right away, but either way, asynchronous
        // writers will be busy waiting either for size or for the data to be ready.
        --m_size;

        // Get and increment read pos
        const index_t read_idx =
            m_read_idx.exchange(increment(m_read_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
        MNRY_ASSERT(read_idx < k_capacity);

        // Wait until we have valid data
        // Can't atomic::wait here because we need to wait if we're either unoccupied or in transition
        while (m_nodes[read_idx].m_occupied != occupied_t::OCCUPIED) {
            do_pause();
        }

        try {
            IF_CONSTEXPR (pop_implementation == PopImplementation::PLACEMENT) {
                new (std::addressof(ret)) T{ std::move_if_noexcept(*get_pointer(read_idx)) };
            } else IF_CONSTEXPR (pop_implementation == PopImplementation::MOVE) {
                ret = std::move_if_noexcept(*get_pointer(read_idx));
            }

            // Destroy
            get_pointer(read_idx)->~T();

            // Mark as not occupied
            m_nodes[read_idx].m_occupied = occupied_t::UNOCCUPIED;

            // We only need to wake up one thread, but we have two wait variables, and we can't control which
            // thread gets the notification. Inform one thread that the size has changed, and inform all threads
            // (including the one that knows the size has changed) that the status has changed
            notify_one(m_size);
            notify_all(m_nodes[read_idx].m_occupied);
        } catch (...) {
            get_pointer(read_idx)->~T();
            m_nodes[read_idx].m_occupied = occupied_t::UNOCCUPIED;

            // We only need to wake up one thread, but we have two wait variables, and we can't control which
            // thread gets the notification. Inform one thread that the size has changed, and inform all threads
            // (including the one that knows the size has changed) that the status has changed
            notify_one(m_size);
            notify_all(m_nodes[read_idx].m_occupied);
            throw;
        }
        return true;
    }

    // Special case for a single consumer.
    // This version of pop handles the class throwing exceptions while still maintaining lock-free semantics.
    template <PopImplementation pop_implementation>
    NO_DISCARD bool try_pop_single_lock_free_impl(T& ret)
    {
        // Check for empty buffer
        // We're the only reader. m_size will only increase.
        if (m_size == 0) {
            return false;
        }

        // We only have one thread reading. We don't have to decrement this right away, but either way, asynchronous
        // writers will be busy waiting either for size or for the data to be ready.
        --m_size;

        while (true) {
            // Get and increment read pos
            const index_t read_idx =
                m_read_idx.exchange(increment(m_read_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
            MNRY_ASSERT(read_idx < k_capacity);

            auto cleanup_function = [this, read_idx] {
                // Mark as not occupied
                m_nodes[read_idx].m_occupied = occupied_t::UNOCCUPIED;

                // There are only other writers, so we don't need to worry about waking another reader.
                // We only need to wake up one thread, but we have two wait variables, and we can't control which
                // thread gets the notification. Inform one thread that the size has changed, and inform all threads
                // (including the one that knows the size has changed) that the status has changed
                notify_one(m_size);
                notify_all(m_nodes[read_idx].m_occupied);
            };

            // Check to see if we have valid data
            occupied_t occupied_value = occupied_t::OCCUPIED;
            if (m_nodes[read_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::IN_TRANSITION)) {
                auto cleanup = finally(cleanup_function);

                try {
                    IF_CONSTEXPR (pop_implementation == PopImplementation::PLACEMENT) {
                        new (std::addressof(ret)) T{ std::move_if_noexcept(*get_pointer(read_idx)) };
                    } else IF_CONSTEXPR (pop_implementation == PopImplementation::MOVE) {
                        ret = std::move_if_noexcept(*get_pointer(read_idx));
                    }

                    // Destroy
                    get_pointer(read_idx)->~T();
                    return true;
                } catch (...) {
                    get_pointer(read_idx)->~T();
                    throw;
                }
            } else if (occupied_value == occupied_t::EXCEPTION) {
                auto cleanup = finally(cleanup_function);
                return false;
            }
        }
    }

    static index_t mod(index_t v) noexcept
    {
        return v % k_capacity;
    }

    static index_t increment(index_t v) noexcept
    {
        return mod(v + 1u);
    }

    T* get_pointer(index_t i) noexcept
    {
        return reinterpret_cast<T*>(&m_nodes[i].m_storage);
    }

    const T* get_pointer(index_t i) const noexcept
    {
        return reinterpret_cast<T*>(&m_nodes[i].m_storage);
    }

    using storage_t = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    struct alignas(16) Node
    {
        Node()
        : m_storage{}
        , m_occupied{ occupied_t::UNOCCUPIED }
        {
        }

        storage_t         m_storage;
        atomic_occupied_t m_occupied;
    };

    alignas(CACHE_LINE_SIZE) atomic_index_t m_size;
    alignas(CACHE_LINE_SIZE) atomic_index_t m_read_idx;
    alignas(CACHE_LINE_SIZE) atomic_index_t m_write_idx;
    alignas(CACHE_LINE_SIZE) Node* m_nodes;
};

struct SingleProducerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::SINGLE_PRODUCER;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::MULTIPLE_CONSUMERS;
    static constexpr LockTraits     lock_traits     = LockTraits::AUTO_DETECT;
};

struct SingleConsumerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::MULTIPLE_PRODUCERS;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::SINGLE_CONSUMER;
    static constexpr LockTraits     lock_traits     = LockTraits::AUTO_DETECT;
};

struct SingleProducerSingleConsumerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::SINGLE_PRODUCER;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::SINGLE_CONSUMER;
    static constexpr LockTraits     lock_traits     = LockTraits::AUTO_DETECT;
};

struct LockFreeTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::MULTIPLE_PRODUCERS;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::MULTIPLE_CONSUMERS;
    static constexpr LockTraits     lock_traits     = LockTraits::LOCK_FREE;
};

template <typename T, std::size_t log_n_elements>
using RingBuffer = RingBufferImpl<T, log_n_elements, DefaultRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleProducer = RingBufferImpl<T, log_n_elements, SingleProducerRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleConsumer = RingBufferImpl<T, log_n_elements, SingleConsumerRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleProducerSingleConsumer =
    RingBufferImpl<T, log_n_elements, SingleProducerSingleConsumerRingBufferTraits>;

