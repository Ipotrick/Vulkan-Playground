#pragma once

#include "daxa/c/core.h"
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <array>
#include <concepts>
#include <span>
#include <limits>

#include <daxa/core.hpp>

namespace daxa
{
    inline namespace types
    {
        using u8 = std::uint8_t;
        using u16 = std::uint16_t;
        using u32 = std::uint32_t;
        using u64 = std::uint64_t;
        using usize = std::size_t;
        using b32 = u32;

        using i8 = std::int8_t;
        using i16 = std::int16_t;
        using i32 = std::int32_t;
        using i64 = std::int64_t;
        using isize = std::ptrdiff_t;

        using f32 = float;
        using f64 = double;

        using DeviceAddress = u64;
    } // namespace types

    struct ImplHandle;

    template <typename CRTP_CHILD_T, typename HANDLE_T = ImplHandle *>
    struct ManagedPtr
    {
        ManagedPtr() = default;

        ~ManagedPtr()
        {
            cleanup();
        }

        ManagedPtr(ManagedPtr const & other) { *this = other; }

        ManagedPtr(ManagedPtr && other) noexcept { *this = std::move(other); }

        auto operator=(ManagedPtr const & other) -> ManagedPtr &
        {
            cleanup();
            this->object = other.object;
            if (this->object != nullptr)
            {
                CRTP_CHILD_T::inc_refcnt(reinterpret_cast<ImplHandle *>(object));
            }
            return *this;
        }

        auto operator=(ManagedPtr && other) noexcept -> ManagedPtr &
        {
            cleanup();
            std::swap(this->object, other.object);
            return *this;
        }

        auto is_valid() const -> bool
        {
            return this->object != nullptr;
        }

        auto get() -> HANDLE_T { return object; }

        auto get() const -> HANDLE_T { return object; }

      protected:
        HANDLE_T object = {};

        void cleanup()
        {
            if (this->object != nullptr)
            {
                CRTP_CHILD_T::dec_refcnt(reinterpret_cast<ImplHandle *>(object));
                this->object = {};
            }
        }
    };

    struct NoneT
    {
    };
    static inline constexpr NoneT None = NoneT{};

    template <typename T>
    struct Optional
    {
      private:
        T m_value = {};
        bool m_has_value = {};

      public:
        Optional() : m_value{}, m_has_value{false} {}
        Optional(Optional<T> const &) = default;
        Optional(T const & v) : m_value{v}, m_has_value{true} {}
        Optional(NoneT const &) : m_value{}, m_has_value{} {}
        Optional<T> & operator=(Optional<T> const &) = default;
        Optional<T> & operator=(T const & v)
        {
            this->m_value = v;
            this->m_has_value = true;
            return *this;
        }
        Optional<T> & operator=(NoneT const &)
        {
            this->m_value = {};
            this->m_has_value = {};
            return *this;
        }

        [[nodiscard]] auto has_value() const -> bool
        {
            return this->m_has_value;
        }

        [[nodiscard]] auto value() -> T &
        {
            return this->m_value;
        }

        [[nodiscard]] auto value() const -> T const &
        {
            return this->m_value;
        }

        [[nodiscard]] auto value_or(T const & v) const -> T const &
        {
            return has_value() ? value() : v;
        }
    };

    /// NOTE: We never need more then 255 elements in a fixed list.
    using FixedListSizeT = u8;

    template <typename T, FixedListSizeT CAPACITY>
    struct FixedList
    {
        std::array<T, CAPACITY> m_data = {};
        FixedListSizeT m_size = {};

        FixedList() = default;
        FixedList(T const * in_data, usize in_size)
        {
            DAXA_DBG_ASSERT_TRUE_M(static_cast<FixedListSizeT>(in_size) < CAPACITY, "EXCEEDED CAPACITY");
            for (FixedListSizeT i = 0; i < static_cast<FixedListSizeT>(in_size); ++i)
            {
                m_data[i] = in_data[i];
            }
            m_size = static_cast<FixedListSizeT>(in_size);
        }
        template <usize IN_SIZE>
            requires(IN_SIZE <= CAPACITY)
        FixedList(std::array<T, IN_SIZE> const & in)
        {
            for (FixedListSizeT i = 0; i < static_cast<FixedListSizeT>(IN_SIZE); ++i)
            {
                m_data[i] = in[i];
            }
            m_size = IN_SIZE;
        }
        FixedList(std::initializer_list<T> const & in)
        {
            auto in_size = std::min<FixedListSizeT>(CAPACITY, static_cast<FixedListSizeT>(in.size()));
            FixedListSizeT i = 0;
            for (auto const & elem : in)
            {
                if (!(i < in_size))
                {
                    break;
                }
                m_data[i] = elem;
                ++i;
            }
            m_size = in_size;
        }
        [[nodiscard]] auto at(FixedListSizeT i) -> T &
        {
            DAXA_DBG_ASSERT_TRUE_M(i < m_size, "INDEX OUT OF RANGE");
            return this->m_data.at(i);
        }
        [[nodiscard]] auto at(FixedListSizeT i) const -> T const &
        {
            return this->m_data.at(i);
        }
        [[nodiscard]] auto operator[](FixedListSizeT i) -> T &
        {
            return this->m_data[i];
        }
        [[nodiscard]] auto operator[](FixedListSizeT i) const -> T const &
        {
            return this->m_data[i];
        }
        [[nodiscard]] static constexpr auto capacity() -> FixedListSizeT
        {
            return CAPACITY;
        }
        [[nodiscard]] auto size() const -> FixedListSizeT
        {
            return this->m_size;
        }
        [[nodiscard]] auto data() const -> T const *
        {
            return this->m_data.data();
        }
        [[nodiscard]] auto data() -> T *
        {
            return this->m_data.data();
        }
        [[nodiscard]] auto empty() const -> bool
        {
            return this->m_size == 0;
        }
        void push_back(T v)
        {
            DAXA_DBG_ASSERT_TRUE_M(m_size < CAPACITY, "EXCEEDED CAPACITY");
            this->m_data[this->m_size++] = v;
        }
        void pop_back()
        {
            DAXA_DBG_ASSERT_TRUE_M(m_size > 0, "ALREADY EMPTY");
            this->m_data[this->m_size--].~T();
        }
        [[nodiscard]] auto back() -> T &
        {
            DAXA_DBG_ASSERT_TRUE_M(m_size > 0, "EMPTY");
            return this->m_data[this->m_size - 1];
        }
        [[nodiscard]] auto span() -> std::span<T>
        {
            return {this->m_data.data(), static_cast<usize>(this->m_size)};
        }
        [[nodiscard]] auto span() const -> std::span<T const>
        {
            return {this->m_data.data(), static_cast<usize>(this->m_size)};
        }
    };

    template <typename T>
    struct Span
    {
        T * m_data = {};
        usize m_size = {};

        Span() = default;
        Span(T const * in_data, usize in_size) : m_data{in_data}, m_size{in_size} {}
        template <typename T2, usize IN_SIZE>
            requires std::same_as<T2, std::remove_cv_t<T>>
        Span(std::array<T2, IN_SIZE> const & in) : m_data{in.data()}, m_size{in.size()}
        {
        }
        [[nodiscard]] auto at(usize i) -> T &
        {
            DAXA_DBG_ASSERT_TRUE_M(i < m_size, "INDEX OUT OF RANGE");
            return this->m_data[i];
        }
        [[nodiscard]] auto at(usize i) const -> T const &
        {
            return this->m_data.at(i);
        }
        [[nodiscard]] auto operator[](usize i) -> T &
        {
            return this->m_data[i];
        }
        [[nodiscard]] auto operator[](usize i) const -> T const &
        {
            return this->m_data[i];
        }
        [[nodiscard]] auto size() const -> usize
        {
            return this->m_size;
        }
        [[nodiscard]] auto data() const -> T const *
        {
            return this->m_data.data();
        }
        [[nodiscard]] auto data() -> T *
        {
            return this->m_data.data();
        }
        [[nodiscard]] auto empty() const -> bool
        {
            return this->m_size == 0;
        }
        [[nodiscard]] auto back() -> T &
        {
            DAXA_DBG_ASSERT_TRUE_M(m_size > 0, "EMPTY");
            return this->m_data[this->m_size - 1];
        }
        [[nodiscard]] auto span() const -> std::span<T const>
        {
            return {this->m_data.data(), static_cast<usize>(this->m_size)};
        }
    };

    struct SmallString final : public FixedList<char, DAXA_SMALL_STRING_CAPACITY>
    {
        using FixedList::FixedList;
        constexpr SmallString(char const * c_str)
        {
            while (c_str != nullptr && *c_str != 0)
            {
                if (this->m_size >= this->capacity())
                {
                    break;
                }
                this->m_data[this->m_size++] = *(c_str++);
            }
        }
        constexpr SmallString(std::string_view sw)
        {
            this->m_size = static_cast<FixedListSizeT>(std::min(static_cast<FixedListSizeT>(sw.size()), this->capacity()));
            for (FixedListSizeT i = 0; i < this->m_size; ++i)
            {
                this->m_data[i] = sw[i];
            }
        }
        constexpr SmallString(std::string const & stl_str)
        {
            this->m_size = static_cast<FixedListSizeT>(std::min(static_cast<FixedListSizeT>(stl_str.size()), this->capacity()));
            for (FixedListSizeT i = 0; i < this->m_size; ++i)
            {
                this->m_data[i] = stl_str[i];
            }
        }
        SmallString(SmallString const & other) = default;
        SmallString & operator=(SmallString const & other) = default;
        [[nodiscard]] auto view() const -> std::string_view
        {
            return {this->m_data.data(), static_cast<usize>(this->m_size)};
        }
        [[nodiscard]] auto c_str() const -> std::array<char, DAXA_SMALL_STRING_CAPACITY + 1>
        {
            std::array<char, DAXA_SMALL_STRING_CAPACITY + 1> ret;
            for (u8 i = 0; i < this->m_size; ++i)
            {
                ret[i] = this->m_data[i];
            }
            for (u8 i = this->m_size; i < DAXA_SMALL_STRING_CAPACITY + 1; ++i)
            {
                ret[i] = 0;
            }
            return ret;
        }
    };
    static_assert(sizeof(SmallString) == 64);

    // clang-format off

    #define DXV_FWD(x) static_cast<decltype(x) &&>(x)
    #define DXV_MOV(x) static_cast<std::remove_reference_t<decltype(x)> &&>(x)

    namespace Variant_detail {
        struct variant_tag {};
        struct emplacer_tag {};
    } // namespace Variant_detail

    template <class T> struct in_place_type_t : private Variant_detail::emplacer_tag {};
    template <std::size_t Index> struct in_place_index_t : private Variant_detail::emplacer_tag {};
    template <std::size_t Index> inline static constexpr in_place_index_t<Index> in_place_index;
    template <class T> inline static constexpr in_place_type_t<T> in_place_type;

    namespace Variant_detail {
        template <int N>
        constexpr auto find_first_true(bool (&&arr)[N]) -> int {
            for (int k = 0; k < N; ++k) { if (arr[k]) { return k; } }
            return -1;
        }
        template <class T, class... Ts>
        inline constexpr bool appears_exactly_once = (static_cast<unsigned short>(std::is_same_v<T, Ts>) + ...) == 1;

        // ============= type pack element

        template <unsigned char = 1> struct find_type_i;
        template <> struct find_type_i<1> {
            template <std::size_t Idx, class T, class... Ts> using f = typename find_type_i<(Idx != 1)>::template f<Idx - 1, Ts...>;
        };
        template <> struct find_type_i<0> {
            template <std::size_t, class T, class... Ts> using f = T;
        };

        // ============= overload match detector. to be used for Variant generic assignment

        template <class T> using arr1 = T[1];
        template <std::size_t N, class A> struct overload_frag {
            using type = A;
            template <class T> requires requires { arr1<A>{std::declval<T>()}; } auto operator()(A, T &&) -> overload_frag<N, A>;
        };
        template <class Seq, class... Args> struct make_overload;
        template <std::size_t... Idx, class... Args> struct make_overload<std::integer_sequence<std::size_t, Idx...>, Args...> : overload_frag<Idx, Args>... {
            using overload_frag<Idx, Args>::operator()...;
        };
        template <class T, class... Ts> using best_overload_match = typename decltype(make_overload<std::make_index_sequence<sizeof...(Ts)>, Ts...>{}(std::declval<T>(), std::declval<T>()))::type;
        template <class T, class... Ts> concept has_non_ambiguous_match = requires { typename best_overload_match<T, Ts...>; };
        template <class A> struct emplace_no_dtor_from_elem {
            template <class T> constexpr void operator()(T &&elem, auto index_) const {
                a.template emplace_no_dtor<index_>(static_cast<T &&>(elem));
            }
            A &a;
        };
        template <class E, class T>
        constexpr void destruct(T &obj) {
            if constexpr (not std::is_trivially_destructible_v<E>) { obj.~E(); }
        }

        // =============================== Variant union types

        // =================== base Variant storage type
        // this type is used to build a N-ary tree of union.
        struct dummy_type {
            static constexpr int elem_size = 0;
        }; // used to fill the back of union nodes
        using union_index_t = unsigned;

        template <bool IsLeaf> struct node_trait;
        template <> struct node_trait<true> {
            template <class A, class B>
            static constexpr auto elem_size = not(std::is_same_v<B, dummy_type>) ? 2 : 1;
            template <std::size_t, class>
            static constexpr char ctor_branch = 0;
        };
        template <> struct node_trait<false> {
            template <class A, class B>
            static constexpr auto elem_size = A::elem_size + B::elem_size;
            template <std::size_t Index, class A>
            static constexpr char ctor_branch = (Index < A::elem_size) ? 1 : 2;
        };

        #define TRAIT(trait) (std::is_##trait##_v<A> && std::is_##trait##_v<B>)
        #define SFM(signature, trait)                                            \
            signature = default;                                                 \
            signature requires(TRAIT(trait) and not TRAIT(trivially_##trait)) {} \
        // given the two members of type A and B of an union X
        // this create the proper conditionally trivial special members functions
        #define INJECT_UNION_SFM(X)                                                   \
            SFM(constexpr X(const X &), copy_constructible)                           \
            SFM(constexpr X(X &&), move_constructible)                                \
            SFM(constexpr auto operator=(const X &) noexcept -> X &, copy_assignable) \
            SFM(constexpr auto operator=(X &&) noexcept -> X &, move_assignable)      \
            SFM(constexpr ~X(), destructible)

        template <bool IsLeaf, class A, class B> struct union_node {
            union {
                A a;
                B b;
            };
            static constexpr auto elem_size = node_trait<IsLeaf>::template elem_size<A, B>; constexpr union_node() = default;
            template <std::size_t Index, class... Args> requires(node_trait<IsLeaf>::template ctor_branch<Index, A> == 1) constexpr explicit union_node(in_place_index_t<Index> /*unused*/, Args &&...args) : a{in_place_index<Index>, static_cast<Args &&>(args)...} {}
            template <std::size_t Index, class... Args> requires(node_trait<IsLeaf>::template ctor_branch<Index, A> == 2) constexpr explicit union_node(in_place_index_t<Index> /*unused*/, Args &&...args) : b{in_place_index<Index - A::elem_size>, static_cast<Args &&>(args)...} {}
            template <class... Args> requires(IsLeaf) constexpr explicit union_node(in_place_index_t<0> /*unused*/, Args &&...args) : a{static_cast<Args &&>(args)...} {}
            template <class... Args> requires(IsLeaf) constexpr explicit union_node(in_place_index_t<1> /*unused*/, Args &&...args) : b{static_cast<Args &&>(args)...} {}
            constexpr explicit union_node(dummy_type /*unused*/) requires(std::is_same_v<dummy_type, B>) : b{} {}
            template <union_index_t Index>
            constexpr auto get() -> auto & {
                if constexpr (IsLeaf) {
                    if constexpr (Index == 0) { return a; } else { return b; }
                } else {
                    if constexpr (Index < A::elem_size) { return a.template get<Index>(); } else { return b.template get<Index - A::elem_size>(); }
                }
            }
            // NOTE(grundlett) This intentionally does not implement copy/move correctly, since it must
            // be implemented by Variant instead
            INJECT_UNION_SFM(union_node)
        };
        #undef INJECT_UNION_SFM
        #undef SFM
        #undef TRAIT
        // =================== algorithm to build the tree of unions
        // take a sequence of types and perform an order preserving fold until only one type is left
        // the first parameter is the numbers of types remaining for the current pass
        constexpr auto pick_next(unsigned remaining) -> unsigned char {
            return remaining >= 2 ? 2 : static_cast<unsigned char>(remaining);
        }
        template <unsigned char Pick, unsigned char GoOn, bool FirstPass> struct make_tree;
        template <bool IsFirstPass> struct make_tree<2, 1, IsFirstPass> {
            template <unsigned Remaining, class A, class B, class... Ts>
            using f = typename make_tree<
                pick_next(Remaining - 2),
                static_cast<unsigned char>(sizeof...(Ts) != 0),
                IsFirstPass>::template f<Remaining - 2, Ts..., union_node<IsFirstPass, A, B>>;
        };

        // only one type left, stop
        template <bool F> struct make_tree<0, 0, F> {
            template <unsigned, class A> using f = A;
        };

        // end of one pass, restart
        template <bool IsFirstPass> struct make_tree<0, 1, IsFirstPass> {
            template <unsigned Remaining, class... Ts>
            using f = typename make_tree<
                pick_next(sizeof...(Ts)),
                static_cast<unsigned char>(sizeof...(Ts) != 1),
                false // <- both first pass and tail call recurse into a tail call
                >::template f<sizeof...(Ts), Ts...>;
        };

        // one odd type left in the pass, put it at the back to preserve the order
        template <> struct make_tree<1, 1, false> {
            template <unsigned Remaining, class A, class... Ts>
            using f = typename make_tree<0, static_cast<unsigned char>(sizeof...(Ts) != 0), false>::template f<0, Ts..., A>;
        };
        // one odd type left in the first pass, wrap it in an union
        template <> struct make_tree<1, 1, true> {
            template <unsigned, class A, class... Ts> using f = typename make_tree<0, static_cast<unsigned char>(sizeof...(Ts) != 0), false>::template f<0, Ts..., union_node<true, A, dummy_type>>;
        };
        template <class... Ts> using make_tree_union = typename make_tree<pick_next(sizeof...(Ts)), 1, true>::template f<sizeof...(Ts), Ts...>;

        // ============================================================
        // Ts... must be sorted in ascending size
        template <std::size_t Num, class... Ts> using smallest_suitable_integer_type = uint8_t;
        // ========================= visit dispatcher
        template <class Fn, class... Vars> using rtype_visit = decltype((std::declval<Fn>()(std::declval<Vars>().template unsafe_get<0>()...)));
        template <class Fn, class Var> using rtype_index_visit = decltype((std::declval<Fn>()(std::declval<Var>().template unsafe_get<0>(), std::integral_constant<std::size_t, 0>{})));

        inline namespace v1 {
            #if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
            #define DeclareUnreachable() __builtin_unreachable()
            #elif defined(_MSC_VER)
            #define DeclareUnreachable() __assume(false)
            #else
            #error "Compiler not supported, please file an issue."
            #endif
            #define DEC(N) X((N)) X((N) + 1) X((N) + 2) X((N) + 3) X((N) + 4) X((N) + 5) X((N) + 6) X((N) + 7) X((N) + 8) X((N) + 9)
            #define SEQ30(N) DEC((N) + 0) DEC((N) + 10) DEC((N) + 20)
            #define SEQ100(N) SEQ30((N) + 0) SEQ30((N) + 30) SEQ30((N) + 60) DEC((N) + 90)
            #define SEQ200(N) SEQ100((N) + 0) SEQ100((N) + 100)
            #define SEQ400(N) SEQ200((N) + 0) SEQ200((N) + 200)
            #define CAT(M, N) M##N
            #define CAT2(M, N) CAT(M, N)
            #define INJECT_SEQ(N) CAT2(SEQ, N)(0)
            // single-visitation
            template <unsigned Offset, class Rtype, class Fn, class V>
            constexpr auto single_visit_tail(Fn &&fn, V &&v) -> Rtype {
                constexpr auto RemainingIndex = std::decay_t<V>::size - Offset;
                #define X(N)                                                                                         \
                    case ((N) + Offset):                                                                             \
                        if constexpr ((N) < RemainingIndex) {                                                        \
                            return static_cast<Fn &&>(fn)(static_cast<V &&>(v).template unsafe_get<(N) + Offset>()); \
                            break;                                                                                   \
                        } else { DeclareUnreachable(); }
                #define SEQ_SIZE 200
                switch (v.index()) {
                    INJECT_SEQ(SEQ_SIZE)
                default:
                    if constexpr (SEQ_SIZE < RemainingIndex) {
                        return Variant_detail::single_visit_tail<Offset + SEQ_SIZE, Rtype>(static_cast<Fn &&>(fn), static_cast<V &&>(v));
                    } else {
                        DeclareUnreachable();
                    }
                }
                #undef X
                #undef SEQ_SIZE
            }

            template <unsigned Offset, class Rtype, class Fn, class V>
            constexpr auto single_visit_w_index_tail(Fn &&fn, V &&v) -> Rtype {
                constexpr auto RemainingIndex = std::decay_t<V>::size - Offset;
                #define X(N)                                                                                                                                           \
                    case ((N) + Offset):                                                                                                                               \
                        if constexpr ((N) < RemainingIndex) {                                                                                                          \
                            return static_cast<Fn &&>(fn)(static_cast<V &&>(v).template unsafe_get<(N) + Offset>(), std::integral_constant<unsigned, (N) + Offset>{}); \
                            break;                                                                                                                                     \
                        } else { DeclareUnreachable(); }

                #define SEQ_SIZE 200
                switch (v.index()) {
                    INJECT_SEQ(SEQ_SIZE)
                default:
                    if constexpr (SEQ_SIZE < RemainingIndex) {
                        return Variant_detail::single_visit_w_index_tail<Offset + SEQ_SIZE, Rtype>(static_cast<Fn &&>(fn), static_cast<V &&>(v));
                    } else {
                        DeclareUnreachable();
                    }
                }
                #undef X
                #undef SEQ_SIZE
            }
            template <class Fn, class V>
            constexpr auto visit(Fn &&fn, V &&v) -> decltype(auto) {
                return Variant_detail::single_visit_tail<0, rtype_visit<Fn &&, V &&>>(DXV_FWD(fn), DXV_FWD(v));
            }
            // unlike other visit functions, this takes the Variant first!
            // this is confusing, but make the client code easier to read
            template <class Fn, class V>
            constexpr auto visit_with_index(V &&v, Fn &&fn) -> decltype(auto) {
                return Variant_detail::single_visit_w_index_tail<0, rtype_index_visit<Fn &&, V &&>>(DXV_FWD(fn), DXV_FWD(v));
            }
            template <class Fn, class Head, class... Tail>
            constexpr auto multi_visit(Fn &&fn, Head &&head, Tail &&...tail) -> decltype(auto) {
                // visit them one by one, starting with the last
                auto vis = [&fn, &head](auto &&...args) -> decltype(auto) {
                    return Variant_detail::visit([&fn, &args...](auto &&elem) -> decltype(auto) { return DXV_FWD(fn)(DXV_FWD(elem), DXV_FWD(args)...); }, DXV_FWD(head));
                };
                if constexpr (sizeof...(tail) == 0) {
                    return DXV_FWD(vis)();
                } else if constexpr (sizeof...(tail) == 1) {
                    return Variant_detail::visit(DXV_FWD(vis), DXV_FWD(tail)...);
                } else {
                    return Variant_detail::multi_visit(DXV_FWD(vis), DXV_FWD(tail)...);
                }
            }
            #undef DEC
            #undef SEQ30
            #undef SEQ100
            #undef SEQ200
            #undef SEQ400
            #undef DeclareUnreachable
            #undef CAT
            #undef CAT2
            #undef INJECT_SEQ
        } // namespace v1

        struct variant_npos_t {
            template <class T> constexpr auto operator==(T index) const noexcept -> bool { return index == std::numeric_limits<T>::max(); }
        };

        template <class T>
        inline constexpr T* addressof( T& obj ) noexcept {
            #if defined(__GNUC__) || defined(__clang__)
                return __builtin_addressof(obj);
            #else
                // if & is overloaded, use the ugly version
                // if constexpr (requires { obj.operator&(); }) {
                //     return reinterpret_cast<T*> (&const_cast<char&>(reinterpret_cast<const volatile char&>(obj)));
                // } else {
                //     return &obj;
                // }
                return std::addressof(obj);
            #endif
        }
    } // namespace Variant_detail

    template <class T> inline constexpr bool is_variant = std::is_base_of_v<Variant_detail::variant_tag, std::decay_t<T>>;
    inline static constexpr Variant_detail::variant_npos_t variant_npos;

    template <class... Ts> class Variant;

    // ill-formed Variant, an empty specialization prevents some really bad errors messages on gcc
    template <class... Ts> requires((std::is_array_v<Ts> || ...) || (std::is_reference_v<Ts> || ...) || (std::is_void_v<Ts> || ...) || sizeof...(Ts) == 0)
    class Variant<Ts...> {
        static_assert(sizeof...(Ts) > 0, "A Variant cannot be empty.");
        static_assert(not(std::is_reference_v<Ts> || ...), "A Variant cannot contain references, consider using reference wrappers instead.");
        static_assert(not(std::is_void_v<Ts> || ...), "A Variant cannot contains void.");
        static_assert(not(std::is_array_v<Ts> || ...), "A Variant cannot contain a raw array type, consider using std::array instead.");
    };

    template <class... Ts>
    class Variant : private Variant_detail::variant_tag {
        using storage_t = Variant_detail::union_node<false, Variant_detail::make_tree_union<Ts...>, Variant_detail::dummy_type>;
        static constexpr bool is_trivial = std::is_trivial_v<storage_t>;
        static constexpr bool has_copy_ctor = std::is_copy_constructible_v<storage_t>;
        static constexpr bool trivial_copy_ctor = is_trivial || std::is_trivially_copy_constructible_v<storage_t>;
        static constexpr bool has_copy_assign = std::is_copy_constructible_v<storage_t>;
        static constexpr bool trivial_copy_assign = is_trivial || std::is_trivially_copy_assignable_v<storage_t>;
        static constexpr bool has_move_ctor = std::is_move_constructible_v<storage_t>;
        static constexpr bool trivial_move_ctor = is_trivial || std::is_trivially_move_constructible_v<storage_t>;
        static constexpr bool has_move_assign = std::is_move_assignable_v<storage_t>;
        static constexpr bool trivial_move_assign = is_trivial || std::is_trivially_move_assignable_v<storage_t>;
        static constexpr bool trivial_dtor = std::is_trivially_destructible_v<storage_t>;

      public:
        template <std::size_t Idx>
        using alternative = std::remove_reference_t<decltype(std::declval<storage_t &>().template get<Idx>())>;
        static constexpr bool can_be_valueless = not is_trivial;
        static constexpr unsigned size = sizeof...(Ts);
        using index_type = Variant_detail::smallest_suitable_integer_type<sizeof...(Ts) + static_cast<size_t>(can_be_valueless), unsigned char, unsigned short, unsigned>;
        static constexpr index_type npos = std::numeric_limits<index_type>::max();
        template <class T>
        static constexpr int index_of = Variant_detail::find_first_true({std::is_same_v<T, Ts>...});

        // ============================================= constructors (20.7.3.2)

        // default constructor
        constexpr Variant() noexcept(std::is_nothrow_default_constructible_v<alternative<0>>) requires std::is_default_constructible_v<alternative<0>> : storage{in_place_index<0>} {}
        // copy constructor (trivial)
        constexpr Variant(const Variant &) requires trivial_copy_ctor = default;
        // note : both the copy and move constructor cannot be meaningfully constexpr without std::construct_at copy constructor
        constexpr Variant(const Variant &o) requires(has_copy_ctor and not trivial_copy_ctor) : storage{Variant_detail::dummy_type{}} { construct_from(o); }
        // move constructor (trivial)
        constexpr Variant(Variant &&) noexcept requires trivial_move_ctor = default;
        // move constructor
        constexpr Variant(Variant &&o) noexcept((std::is_nothrow_move_constructible_v<Ts> && ...)) requires(has_move_ctor and not trivial_move_ctor) : storage{Variant_detail::dummy_type{}} { construct_from(static_cast<Variant &&>(o)); }
        // generic constructor
        template <class T, class M = Variant_detail::best_overload_match<T &&, Ts...>, class D = std::decay_t<T>>
        constexpr Variant(T &&t) noexcept(std::is_nothrow_constructible_v<M, T &&>)
            requires(not std::is_same_v<D, Variant> and not std::is_base_of_v<Variant_detail::emplacer_tag, D>)
            : Variant{in_place_index<index_of<M>>, static_cast<T &&>(t)} {}
        // construct at index
        template <std::size_t Index, class... Args> requires(Index < size && std::is_constructible_v<alternative<Index>, Args && ...>)
        explicit constexpr Variant(in_place_index_t<Index> tag, Args &&...args) : storage{tag, static_cast<Args &&>(args)...}, current(Index) {}
        // construct a given type
        template <class T, class... Args> requires(Variant_detail::appears_exactly_once<T, Ts...> && std::is_constructible_v<T, Args && ...>)
        explicit constexpr Variant(in_place_type_t<T>, Args &&...args) : Variant{in_place_index<index_of<T>>, static_cast<Args &&>(args)...} {}
        // initializer-list constructors
        template <std::size_t Index, class U, class... Args> requires((Index < size) and std::is_constructible_v<alternative<Index>, std::initializer_list<U> &, Args && ...>)
        explicit constexpr Variant(in_place_index_t<Index> tag, std::initializer_list<U> list, Args &&...args) : storage{tag, list, DXV_FWD(args)...}, current{Index} {}
        template <class T, class U, class... Args> requires(Variant_detail::appears_exactly_once<T, Ts...> && std::is_constructible_v<T, std::initializer_list<U> &, Args && ...>)
        explicit constexpr Variant(in_place_type_t<T>, std::initializer_list<U> list, Args &&...args) : storage{in_place_index<index_of<T>>, list, DXV_FWD(args)...}, current{index_of<T>} {}

        // ================================ destructors (20.7.3.3)

        constexpr ~Variant() requires trivial_dtor = default;
        constexpr ~Variant() requires(not trivial_dtor) { reset(); }

        // ================================ assignment (20.7.3.4)

        // copy assignment (trivial)
        constexpr auto operator=(const Variant &o) -> Variant &
            requires trivial_copy_assign &&trivial_copy_ctor
        = default;
        // copy assignment
        constexpr auto operator=(const Variant &rhs) -> Variant &requires(has_copy_assign and not(trivial_copy_assign && trivial_copy_ctor)) {
            assign_from(rhs, [this](const auto &elem, auto index_cst) {
                if (index() == index_cst) {
                    unsafe_get<index_cst>() = elem;
                } else {
                    using type = alternative<index_cst>;
                    if constexpr (std::is_nothrow_copy_constructible_v<type> or not std::is_nothrow_move_constructible_v<type>) {
                        this->emplace<index_cst>(elem);
                    } else {
                        alternative<index_cst> tmp = elem;
                        this->emplace<index_cst>(DXV_MOV(tmp));
                    }
                }
            });
            return *this;
        }
        // move assignment (trivial)
        constexpr auto operator=(Variant &&o) noexcept -> Variant &requires(trivial_move_assign and trivial_move_ctor and trivial_dtor) = default;
        // move assignment
        constexpr auto operator=(Variant &&o) noexcept((std::is_nothrow_move_constructible_v<Ts> && ...) && (std::is_nothrow_move_assignable_v<Ts> && ...))
             -> Variant &requires(has_move_assign &&has_move_ctor and not(trivial_move_assign and trivial_move_ctor and trivial_dtor)) {
            this->assign_from(DXV_FWD(o), [this](auto &&elem, auto index_cst) {
                if (index() == index_cst) { this->template unsafe_get<index_cst>() = DXV_MOV(elem); } else { this->emplace<index_cst>(DXV_MOV(elem)); }
            });
            return *this;
        }
        // generic assignment
        template <class T>
            requires Variant_detail::has_non_ambiguous_match<T, Ts...>
        constexpr auto operator=(T &&t) noexcept(std::is_nothrow_assignable_v<Variant_detail::best_overload_match<T &&, Ts...>, T &&> &&std::is_nothrow_constructible_v<Variant_detail::best_overload_match<T &&, Ts...>, T &&>) -> Variant & {
            using related_type = Variant_detail::best_overload_match<T &&, Ts...>;
            constexpr auto new_index = index_of<related_type>;
            if (this->current == new_index) {
                this->template unsafe_get<new_index>() = DXV_FWD(t);
            } else {
                constexpr bool do_simple_emplace = std::is_nothrow_constructible_v<related_type, T> or not std::is_nothrow_move_constructible_v<related_type>;
                if constexpr (do_simple_emplace) {
                    this->emplace<new_index>(DXV_FWD(t));
                } else {
                    related_type tmp = t;
                    this->emplace<new_index>(DXV_MOV(tmp));
                }
            }
            return *this;
        }

        // ================================== modifiers (20.7.3.5)
        template <class T, class... Args> requires (std::is_constructible_v<T, Args&&...> && Variant_detail::appears_exactly_once<T, Ts...>)
        constexpr T& emplace(Args&&... args) { return emplace<index_of<T>>(static_cast<Args&&>(args)...); }
        template <std::size_t Idx, class... Args> requires (Idx < size and std::is_constructible_v<alternative<Idx>, Args&&...>  )
        constexpr auto& emplace(Args&&... args) { return emplace_impl<Idx>(DXV_FWD(args)...); }

        // emplace with initializer-lists
        template <std::size_t Idx, class U, class... Args> requires (Idx < size && std::is_constructible_v<alternative<Idx>, std::initializer_list<U>&, Args&&...>)
        constexpr auto& emplace(std::initializer_list<U> list, Args&&... args) { return emplace_impl<Idx>(list, DXV_FWD(args)...); }
        template <class T, class U, class... Args> requires (std::is_constructible_v<T, std::initializer_list<U>&, Args&&...> && Variant_detail::appears_exactly_once<T, Ts...>)
        constexpr T& emplace(std::initializer_list<U> list, Args&&... args) { return emplace_impl<index_of<T>>( list, DXV_FWD(args)... ); }

        // +================================== methods for internal use
        // these methods performs no errors checking at all

        template <Variant_detail::union_index_t Idx>
        constexpr auto unsafe_get() & noexcept -> auto & {
            static_assert(Idx < size); return storage.template get<Idx>();
        }
        template <Variant_detail::union_index_t Idx>
        constexpr auto unsafe_get() && noexcept -> auto && {
            static_assert(Idx < size); return DXV_MOV(storage.template get<Idx>());
        }
        template <Variant_detail::union_index_t Idx>
        [[nodiscard]] constexpr auto unsafe_get() const & noexcept -> const auto & {
            static_assert(Idx < size); return const_cast<Variant &>(*this).unsafe_get<Idx>();
        }
        template <Variant_detail::union_index_t Idx>
        [[nodiscard]] constexpr auto unsafe_get() const && noexcept -> const auto && {
            static_assert(Idx < size); return DXV_MOV(unsafe_get<Idx>());
        }

        // ==================================== value status (20.7.3.6)

        [[nodiscard]] constexpr auto valueless_by_exception() const noexcept -> bool {
            if constexpr (can_be_valueless) { return current == npos; } else { return false; }
        }
        [[nodiscard]] constexpr auto index() const noexcept -> index_type { return current; }

      private:
        template <unsigned Idx, class... Args> constexpr auto &emplace_impl(Args &&...args) {
            reset();
            emplace_no_dtor<Idx>(DXV_FWD(args)...);
            return unsafe_get<Idx>();
        }
        // can be used directly only when the Variant is in a known empty state
        template <unsigned Idx, class... Args> constexpr void emplace_no_dtor(Args &&...args) {
            using T = alternative<Idx>;
            if constexpr (not std::is_nothrow_constructible_v<T, Args &&...>) {
                if constexpr (std::is_nothrow_move_constructible_v<T>) {
                    do_emplace_no_dtor<Idx>(T{DXV_FWD(args)...});
                } else if constexpr (std::is_nothrow_copy_constructible_v<T>) {
                    T tmp{DXV_FWD(args)...};
                    do_emplace_no_dtor<Idx>(tmp);
                } else {
                    static_assert(can_be_valueless && (Idx == Idx), "Internal error : the possibly valueless branch of emplace was taken despite |can_be_valueless| being false");
                    current = npos;
                    do_emplace_no_dtor<Idx>(DXV_FWD(args)...);
                }
            } else {
                do_emplace_no_dtor<Idx>(DXV_FWD(args)...);
            }
        }
        template <unsigned Idx, class... Args> constexpr void do_emplace_no_dtor(Args &&...args) {
            auto *ptr = Variant_detail::addressof(unsafe_get<Idx>());
            using T = alternative<Idx>;
            new (static_cast<void*>(ptr)) T(DXV_FWD(args)...);
            current = static_cast<index_type>(Idx);
        }

        // assign from another Variant
        template <class Other, class Fn>
        constexpr void assign_from(Other &&o, Fn &&fn) {
            if constexpr (can_be_valueless) {
                if (o.index() == npos) {
                    if (current != npos) { reset_no_check(); current = npos; }
                    return;
                }
            }
            Variant_detail::visit_with_index(DXV_FWD(o), DXV_FWD(fn));
        }
        // destroy the current elem IFF not valueless
        constexpr void reset() {
            if constexpr (can_be_valueless) {
                if (valueless_by_exception()) { return; }
            }
            reset_no_check();
        }
        // destroy the current element without checking for valueless
        constexpr void reset_no_check() {
            if constexpr (not trivial_dtor) {
                Variant_detail::visit_with_index(*this, [](auto &elem, auto index_) { Variant_detail::destruct<alternative<index_>>(elem); });
            }
        }
        // construct this from another variant, for constructors only
        template <class Other> constexpr void construct_from(Other &&o) {
            if constexpr (can_be_valueless) if (o.valueless_by_exception()) { current = npos; return; }
            Variant_detail::visit_with_index(DXV_FWD(o), Variant_detail::emplace_no_dtor_from_elem<Variant &>{*this});
        }
        template <class T> friend struct Variant_detail::emplace_no_dtor_from_elem;
        storage_t storage;
        index_type current{};
    };

    template <class T, class... Ts>
    constexpr auto holds_alternative(const Variant<Ts...> &v) noexcept -> bool {
        static_assert((std::is_same_v<T, Ts> || ...), "Requested type is not contained in the Variant");
        constexpr auto Index = Variant<Ts...>::template index_of<T>;
        return v.index() == Index;
    }

    // ========= get by index
    template <std::size_t Idx, class... Ts>
    constexpr auto &get(Variant<Ts...> &v) {
        static_assert(Idx < sizeof...(Ts), "Index exceeds the Variant size. ");
        DAXA_DBG_ASSERT_TRUE_M(v.index() == Idx, "Bad Variant access in get");
        return (v.template unsafe_get<Idx>());
    }
    template <std::size_t Idx, class... Ts> constexpr const auto &get(const Variant<Ts...> &v) { return daxa::get<Idx>(const_cast<Variant<Ts...> &>(v)); }
    template <std::size_t Idx, class... Ts> constexpr auto &&get(Variant<Ts...> &&v) { return SWL_MOV(daxa::get<Idx>(v)); }
    template <std::size_t Idx, class... Ts> constexpr const auto &&get(const Variant<Ts...> &&v) { return SWL_MOV(daxa::get<Idx>(v)); }
    // ========= get by type
    template <class T, class... Ts> constexpr T &get(Variant<Ts...> &v) { return daxa::get<Variant<Ts...>::template index_of<T>>(v); }
    template <class T, class... Ts> constexpr const T &get(const Variant<Ts...> &v) { return daxa::get<Variant<Ts...>::template index_of<T>>(v); }
    template <class T, class... Ts> constexpr T &&get(Variant<Ts...> &&v) { return daxa::get<Variant<Ts...>::template index_of<T>>(DXV_FWD(v)); }
    template <class T, class... Ts> constexpr const T &&get(const Variant<Ts...> &&v) { return daxa::get<Variant<Ts...>::template index_of<T>>(DXV_FWD(v)); }
    // ===== get_if by index
    template <std::size_t Idx, class... Ts> constexpr const auto *get_if(const Variant<Ts...> *v) noexcept {
        using rtype = typename Variant<Ts...>::template alternative<Idx> *;
        if (v == nullptr || v->index() != Idx) { return rtype{nullptr}; }
        else { return Variant_detail::addressof(v->template unsafe_get<Idx>()); }
    }
    template <std::size_t Idx, class... Ts> constexpr auto *get_if(Variant<Ts...> *v) noexcept {
        using rtype = typename Variant<Ts...>::template alternative<Idx>;
        return const_cast<rtype *>(daxa::get_if<Idx>(static_cast<const Variant<Ts...> *>(v)));
    }
    // ====== get_if by type
    template <class T, class... Ts> constexpr T *get_if(Variant<Ts...> *v) noexcept {
        static_assert((std::is_same_v<T, Ts> || ...), "Requested type is not contained in the Variant");
        return daxa::get_if<Variant<Ts...>::template index_of<T>>(v);
    }
    template <class T, class... Ts> constexpr const T *get_if(const Variant<Ts...> *v) noexcept {
        static_assert((std::is_same_v<T, Ts> || ...), "Requested type is not contained in the Variant");
        return daxa::get_if<Variant<Ts...>::template index_of<T>>(v);
    }

    struct Monostate {
    };

    #undef DXV_FWD
    #undef DXV_MOV

    // clang-format on

    enum struct Format
    {
        UNDEFINED = 0,
        R4G4_UNORM_PACK8 = 1,
        R4G4B4A4_UNORM_PACK16 = 2,
        B4G4R4A4_UNORM_PACK16 = 3,
        R5G6B5_UNORM_PACK16 = 4,
        B5G6R5_UNORM_PACK16 = 5,
        R5G5B5A1_UNORM_PACK16 = 6,
        B5G5R5A1_UNORM_PACK16 = 7,
        A1R5G5B5_UNORM_PACK16 = 8,
        R8_UNORM = 9,
        R8_SNORM = 10,
        R8_USCALED = 11,
        R8_SSCALED = 12,
        R8_UINT = 13,
        R8_SINT = 14,
        R8_SRGB = 15,
        R8G8_UNORM = 16,
        R8G8_SNORM = 17,
        R8G8_USCALED = 18,
        R8G8_SSCALED = 19,
        R8G8_UINT = 20,
        R8G8_SINT = 21,
        R8G8_SRGB = 22,
        R8G8B8_UNORM = 23,
        R8G8B8_SNORM = 24,
        R8G8B8_USCALED = 25,
        R8G8B8_SSCALED = 26,
        R8G8B8_UINT = 27,
        R8G8B8_SINT = 28,
        R8G8B8_SRGB = 29,
        B8G8R8_UNORM = 30,
        B8G8R8_SNORM = 31,
        B8G8R8_USCALED = 32,
        B8G8R8_SSCALED = 33,
        B8G8R8_UINT = 34,
        B8G8R8_SINT = 35,
        B8G8R8_SRGB = 36,
        R8G8B8A8_UNORM = 37,
        R8G8B8A8_SNORM = 38,
        R8G8B8A8_USCALED = 39,
        R8G8B8A8_SSCALED = 40,
        R8G8B8A8_UINT = 41,
        R8G8B8A8_SINT = 42,
        R8G8B8A8_SRGB = 43,
        B8G8R8A8_UNORM = 44,
        B8G8R8A8_SNORM = 45,
        B8G8R8A8_USCALED = 46,
        B8G8R8A8_SSCALED = 47,
        B8G8R8A8_UINT = 48,
        B8G8R8A8_SINT = 49,
        B8G8R8A8_SRGB = 50,
        A8B8G8R8_UNORM_PACK32 = 51,
        A8B8G8R8_SNORM_PACK32 = 52,
        A8B8G8R8_USCALED_PACK32 = 53,
        A8B8G8R8_SSCALED_PACK32 = 54,
        A8B8G8R8_UINT_PACK32 = 55,
        A8B8G8R8_SINT_PACK32 = 56,
        A8B8G8R8_SRGB_PACK32 = 57,
        A2R10G10B10_UNORM_PACK32 = 58,
        A2R10G10B10_SNORM_PACK32 = 59,
        A2R10G10B10_USCALED_PACK32 = 60,
        A2R10G10B10_SSCALED_PACK32 = 61,
        A2R10G10B10_UINT_PACK32 = 62,
        A2R10G10B10_SINT_PACK32 = 63,
        A2B10G10R10_UNORM_PACK32 = 64,
        A2B10G10R10_SNORM_PACK32 = 65,
        A2B10G10R10_USCALED_PACK32 = 66,
        A2B10G10R10_SSCALED_PACK32 = 67,
        A2B10G10R10_UINT_PACK32 = 68,
        A2B10G10R10_SINT_PACK32 = 69,
        R16_UNORM = 70,
        R16_SNORM = 71,
        R16_USCALED = 72,
        R16_SSCALED = 73,
        R16_UINT = 74,
        R16_SINT = 75,
        R16_SFLOAT = 76,
        R16G16_UNORM = 77,
        R16G16_SNORM = 78,
        R16G16_USCALED = 79,
        R16G16_SSCALED = 80,
        R16G16_UINT = 81,
        R16G16_SINT = 82,
        R16G16_SFLOAT = 83,
        R16G16B16_UNORM = 84,
        R16G16B16_SNORM = 85,
        R16G16B16_USCALED = 86,
        R16G16B16_SSCALED = 87,
        R16G16B16_UINT = 88,
        R16G16B16_SINT = 89,
        R16G16B16_SFLOAT = 90,
        R16G16B16A16_UNORM = 91,
        R16G16B16A16_SNORM = 92,
        R16G16B16A16_USCALED = 93,
        R16G16B16A16_SSCALED = 94,
        R16G16B16A16_UINT = 95,
        R16G16B16A16_SINT = 96,
        R16G16B16A16_SFLOAT = 97,
        R32_UINT = 98,
        R32_SINT = 99,
        R32_SFLOAT = 100,
        R32G32_UINT = 101,
        R32G32_SINT = 102,
        R32G32_SFLOAT = 103,
        R32G32B32_UINT = 104,
        R32G32B32_SINT = 105,
        R32G32B32_SFLOAT = 106,
        R32G32B32A32_UINT = 107,
        R32G32B32A32_SINT = 108,
        R32G32B32A32_SFLOAT = 109,
        R64_UINT = 110,
        R64_SINT = 111,
        R64_SFLOAT = 112,
        R64G64_UINT = 113,
        R64G64_SINT = 114,
        R64G64_SFLOAT = 115,
        R64G64B64_UINT = 116,
        R64G64B64_SINT = 117,
        R64G64B64_SFLOAT = 118,
        R64G64B64A64_UINT = 119,
        R64G64B64A64_SINT = 120,
        R64G64B64A64_SFLOAT = 121,
        B10G11R11_UFLOAT_PACK32 = 122,
        E5B9G9R9_UFLOAT_PACK32 = 123,
        D16_UNORM = 124,
        X8_D24_UNORM_PACK32 = 125,
        D32_SFLOAT = 126,
        S8_UINT = 127,
        D16_UNORM_S8_UINT = 128,
        D24_UNORM_S8_UINT = 129,
        D32_SFLOAT_S8_UINT = 130,
        BC1_RGB_UNORM_BLOCK = 131,
        BC1_RGB_SRGB_BLOCK = 132,
        BC1_RGBA_UNORM_BLOCK = 133,
        BC1_RGBA_SRGB_BLOCK = 134,
        BC2_UNORM_BLOCK = 135,
        BC2_SRGB_BLOCK = 136,
        BC3_UNORM_BLOCK = 137,
        BC3_SRGB_BLOCK = 138,
        BC4_UNORM_BLOCK = 139,
        BC4_SNORM_BLOCK = 140,
        BC5_UNORM_BLOCK = 141,
        BC5_SNORM_BLOCK = 142,
        BC6H_UFLOAT_BLOCK = 143,
        BC6H_SFLOAT_BLOCK = 144,
        BC7_UNORM_BLOCK = 145,
        BC7_SRGB_BLOCK = 146,
        ETC2_R8G8B8_UNORM_BLOCK = 147,
        ETC2_R8G8B8_SRGB_BLOCK = 148,
        ETC2_R8G8B8A1_UNORM_BLOCK = 149,
        ETC2_R8G8B8A1_SRGB_BLOCK = 150,
        ETC2_R8G8B8A8_UNORM_BLOCK = 151,
        ETC2_R8G8B8A8_SRGB_BLOCK = 152,
        EAC_R11_UNORM_BLOCK = 153,
        EAC_R11_SNORM_BLOCK = 154,
        EAC_R11G11_UNORM_BLOCK = 155,
        EAC_R11G11_SNORM_BLOCK = 156,
        ASTC_4x4_UNORM_BLOCK = 157,
        ASTC_4x4_SRGB_BLOCK = 158,
        ASTC_5x4_UNORM_BLOCK = 159,
        ASTC_5x4_SRGB_BLOCK = 160,
        ASTC_5x5_UNORM_BLOCK = 161,
        ASTC_5x5_SRGB_BLOCK = 162,
        ASTC_6x5_UNORM_BLOCK = 163,
        ASTC_6x5_SRGB_BLOCK = 164,
        ASTC_6x6_UNORM_BLOCK = 165,
        ASTC_6x6_SRGB_BLOCK = 166,
        ASTC_8x5_UNORM_BLOCK = 167,
        ASTC_8x5_SRGB_BLOCK = 168,
        ASTC_8x6_UNORM_BLOCK = 169,
        ASTC_8x6_SRGB_BLOCK = 170,
        ASTC_8x8_UNORM_BLOCK = 171,
        ASTC_8x8_SRGB_BLOCK = 172,
        ASTC_10x5_UNORM_BLOCK = 173,
        ASTC_10x5_SRGB_BLOCK = 174,
        ASTC_10x6_UNORM_BLOCK = 175,
        ASTC_10x6_SRGB_BLOCK = 176,
        ASTC_10x8_UNORM_BLOCK = 177,
        ASTC_10x8_SRGB_BLOCK = 178,
        ASTC_10x10_UNORM_BLOCK = 179,
        ASTC_10x10_SRGB_BLOCK = 180,
        ASTC_12x10_UNORM_BLOCK = 181,
        ASTC_12x10_SRGB_BLOCK = 182,
        ASTC_12x12_UNORM_BLOCK = 183,
        ASTC_12x12_SRGB_BLOCK = 184,
        G8B8G8R8_422_UNORM = 1000156000,
        B8G8R8G8_422_UNORM = 1000156001,
        G8_B8_R8_3PLANE_420_UNORM = 1000156002,
        G8_B8R8_2PLANE_420_UNORM = 1000156003,
        G8_B8_R8_3PLANE_422_UNORM = 1000156004,
        G8_B8R8_2PLANE_422_UNORM = 1000156005,
        G8_B8_R8_3PLANE_444_UNORM = 1000156006,
        R10X6_UNORM_PACK16 = 1000156007,
        R10X6G10X6_UNORM_2PACK16 = 1000156008,
        R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
        G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
        B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
        G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
        G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
        G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
        G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
        G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
        R12X4_UNORM_PACK16 = 1000156017,
        R12X4G12X4_UNORM_2PACK16 = 1000156018,
        R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
        G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
        B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
        G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
        G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
        G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
        G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
        G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
        G16B16G16R16_422_UNORM = 1000156027,
        B16G16R16G16_422_UNORM = 1000156028,
        G16_B16_R16_3PLANE_420_UNORM = 1000156029,
        G16_B16R16_2PLANE_420_UNORM = 1000156030,
        G16_B16_R16_3PLANE_422_UNORM = 1000156031,
        G16_B16R16_2PLANE_422_UNORM = 1000156032,
        G16_B16_R16_3PLANE_444_UNORM = 1000156033,
        G8_B8R8_2PLANE_444_UNORM = 1000330000,
        G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16 = 1000330001,
        G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16 = 1000330002,
        G16_B16R16_2PLANE_444_UNORM = 1000330003,
        A4R4G4B4_UNORM_PACK16 = 1000340000,
        A4B4G4R4_UNORM_PACK16 = 1000340001,
        ASTC_4x4_SFLOAT_BLOCK = 1000066000,
        ASTC_5x4_SFLOAT_BLOCK = 1000066001,
        ASTC_5x5_SFLOAT_BLOCK = 1000066002,
        ASTC_6x5_SFLOAT_BLOCK = 1000066003,
        ASTC_6x6_SFLOAT_BLOCK = 1000066004,
        ASTC_8x5_SFLOAT_BLOCK = 1000066005,
        ASTC_8x6_SFLOAT_BLOCK = 1000066006,
        ASTC_8x8_SFLOAT_BLOCK = 1000066007,
        ASTC_10x5_SFLOAT_BLOCK = 1000066008,
        ASTC_10x6_SFLOAT_BLOCK = 1000066009,
        ASTC_10x8_SFLOAT_BLOCK = 1000066010,
        ASTC_10x10_SFLOAT_BLOCK = 1000066011,
        ASTC_12x10_SFLOAT_BLOCK = 1000066012,
        ASTC_12x12_SFLOAT_BLOCK = 1000066013,
        PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
        PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
        PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
        PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
        PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
        PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
        PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
        PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
        MAX_ENUM = 0x7fffffff,
    };

    auto to_string(Format format) -> std::string_view;

    template <typename Properties>
    struct Flags final
    {
        typename Properties::Data data;
        inline constexpr auto operator|=(Flags const & other) -> Flags &
        {
            data |= other.data;
            return *this;
        }
        inline constexpr auto operator&=(Flags const & other) -> Flags &
        {
            data &= other.data;
            return *this;
        }
        inline constexpr auto operator^=(Flags const & other) -> Flags &
        {
            data ^= other.data;
            return *this;
        }
        [[nodiscard]] inline constexpr auto operator~() const -> Flags
        {
            return {~data};
        }
        [[nodiscard]] inline constexpr auto operator|(Flags const & other) const -> Flags
        {
            return {data | other.data};
        }
        [[nodiscard]] inline constexpr auto operator&(Flags const & other) const -> Flags
        {
            return {data & other.data};
        }
        [[nodiscard]] inline constexpr auto operator^(Flags const & other) const -> Flags
        {
            return {data ^ other.data};
        }
        [[nodiscard]] inline constexpr auto operator<=>(Flags const & other) const = default;

        operator bool() const
        {
            return data != 0;
        }
    };

    enum struct MsgSeverity
    {
        VERBOSE = 0x00000001,
        INFO = 0x00000010,
        WARNING = 0x00000100,
        FAILURE = 0x00001000,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct MsgType
    {
        GENERAL = 0x00000001,
        VALIDATION = 0x00000002,
        PERFORMANCE = 0x00000004,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct PresentMode
    {
        IMMEDIATE = 0,
        MAILBOX = 1,
        FIFO = 2,
        FIFO_RELAXED = 3,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct PresentOp
    {
        IDENTITY = 0x00000001,
        ROTATE_90 = 0x00000002,
        ROTATE_180 = 0x00000004,
        ROTATE_270 = 0x00000008,
        HORIZONTAL_MIRROR = 0x00000010,
        HORIZONTAL_MIRROR_ROTATE_90 = 0x00000020,
        HORIZONTAL_MIRROR_ROTATE_180 = 0x00000040,
        HORIZONTAL_MIRROR_ROTATE_270 = 0x00000080,
        INHERIT = 0x00000100,
        MAX_ENUM = 0x7fffffff,
    };

    struct ImageUsageFlagsProperties
    {
        using Data = u32;
    };
    using ImageUsageFlags = Flags<ImageUsageFlagsProperties>;
    struct ImageUsageFlagBits
    {
        static inline constexpr ImageUsageFlags NONE = {0x00000000};
        static inline constexpr ImageUsageFlags TRANSFER_SRC = {0x00000001};
        static inline constexpr ImageUsageFlags TRANSFER_DST = {0x00000002};
        static inline constexpr ImageUsageFlags SHADER_SAMPLED = {0x00000004};
        static inline constexpr ImageUsageFlags SHADER_STORAGE = {0x00000008};
        static inline constexpr ImageUsageFlags COLOR_ATTACHMENT = {0x00000010};
        static inline constexpr ImageUsageFlags DEPTH_STENCIL_ATTACHMENT = {0x00000020};
        static inline constexpr ImageUsageFlags TRANSIENT_ATTACHMENT = {0x00000040};
        static inline constexpr ImageUsageFlags FRAGMENT_DENSITY_MAP = {0x00000200};
        static inline constexpr ImageUsageFlags FRAGMENT_SHADING_RATE_ATTACHMENT = {0x00000100};
    };

    [[nodiscard]] auto to_string(ImageUsageFlags const &) -> std::string;

    struct MemoryFlagsProperties
    {
        using Data = u32;
    };
    using MemoryFlags = Flags<MemoryFlagsProperties>;
    struct MemoryFlagBits
    {
        static inline constexpr MemoryFlags NONE = {0x00000000};
        static inline constexpr MemoryFlags DEDICATED_MEMORY = {0x00000001};
        static inline constexpr MemoryFlags CAN_ALIAS = {0x00000200};
        static inline constexpr MemoryFlags HOST_ACCESS_SEQUENTIAL_WRITE = {0x00000400};
        static inline constexpr MemoryFlags HOST_ACCESS_RANDOM = {0x00000800};
        static inline constexpr MemoryFlags STRATEGY_MIN_MEMORY = {0x00010000};
        static inline constexpr MemoryFlags STRATEGY_MIN_TIME = {0x00020000};
    };

    enum struct ColorSpace
    {
        SRGB_NONLINEAR = 0,
        DISPLAY_P3_NONLINEAR = 1000104001,
        EXTENDED_SRGB_LINEAR = 1000104002,
        DISPLAY_P3_LINEAR = 1000104003,
        DCI_P3_NONLINEAR = 1000104004,
        BT709_LINEAR = 1000104005,
        BT709_NONLINEAR = 1000104006,
        BT2020_LINEAR = 1000104007,
        HDR10_ST2084 = 1000104008,
        DOLBYVISION = 1000104009,
        HDR10_HLG = 1000104010,
        ADOBERGB_LINEAR = 1000104011,
        ADOBERGB_NONLINEAR = 1000104012,
        PASS_THROUGH = 1000104013,
        EXTENDED_SRGB_NONLINEAR = 1000104014,
        DISPLAY_NATIVE = 1000213000,
        RGB_NONLINEAR = EXTENDED_SRGB_NONLINEAR,
        DCI_P3_LINEAR = DISPLAY_P3_LINEAR,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct ImageLayout
    {
        UNDEFINED = 0,
        GENERAL = 1,
        TRANSFER_SRC_OPTIMAL = 6,
        TRANSFER_DST_OPTIMAL = 7,
        READ_ONLY_OPTIMAL = 1000314000,
        ATTACHMENT_OPTIMAL = 1000314001,
        PRESENT_SRC = 1000001002,
        MAX_ENUM = 0x7fffffff,
    };

    [[nodiscard]] auto to_string(ImageLayout layout) -> std::string_view;

    struct DAXA_EXPORT_CXX ImageMipArraySlice
    {
        u32 base_mip_level = 0;
        u32 level_count = 1;
        u32 base_array_layer = 0;
        u32 layer_count = 1;

        friend auto operator<=>(ImageMipArraySlice const &, ImageMipArraySlice const &) = default;

        [[nodiscard]] auto contains(ImageMipArraySlice const & slice) const -> bool;
        [[nodiscard]] auto intersects(ImageMipArraySlice const & slice) const -> bool;
        [[nodiscard]] auto intersect(ImageMipArraySlice const & slice) const -> ImageMipArraySlice;
        [[nodiscard]] auto subtract(ImageMipArraySlice const & slice) const -> std::tuple<std::array<ImageMipArraySlice, 4>, usize>;
    };

    [[nodiscard]] auto to_string(ImageMipArraySlice image_mip_array_slice) -> std::string;

    struct DAXA_EXPORT_CXX ImageArraySlice
    {
        u32 mip_level = 0;
        u32 base_array_layer = 0;
        u32 layer_count = 1;

        friend auto operator<=>(ImageArraySlice const &, ImageArraySlice const &) = default;

        [[nodiscard]] static auto slice(ImageMipArraySlice const & mip_array_slice, u32 mip_level = 0) -> ImageArraySlice;

        [[nodiscard]] auto contained_in(ImageMipArraySlice const & slice) const -> bool;
    };

    [[nodiscard]] auto to_string(ImageArraySlice image_array_slice) -> std::string;

    struct DAXA_EXPORT_CXX ImageSlice
    {
        u32 mip_level = 0;
        u32 array_layer = 0;

        [[nodiscard]] friend auto operator<=>(ImageSlice const &, ImageSlice const &) = default;

        [[nodiscard]] static auto slice(ImageArraySlice const & mip_array_slice, u32 array_layer = 0) -> ImageSlice;

        [[nodiscard]] auto contained_in(ImageMipArraySlice const & slice) const -> bool;
        [[nodiscard]] auto contained_in(ImageArraySlice const & slice) const -> bool;
    };

    auto to_string(ImageSlice image_slice) -> std::string;

    enum struct Filter
    {
        NEAREST = 0,
        LINEAR = 1,
        CUBIC_IMG = 1000015000,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct ReductionMode
    {
        WEIGHTED_AVERAGE = 0,
        MIN = 1,
        MAX = 2,
        MAX_ENUM = 0x7fffffff,
    };

    struct Offset3D
    {
        i32 x = {};
        i32 y = {};
        i32 z = {};

        friend auto operator<=>(Offset3D const &, Offset3D const &) = default;
    };

    struct Extent2D
    {
        u32 x = {};
        u32 y = {};

        friend auto operator<=>(Extent2D const &, Extent2D const &) = default;
    };

    struct Extent3D
    {
        u32 x = {};
        u32 y = {};
        u32 z = {};

        friend auto operator<=>(Extent3D const &, Extent3D const &) = default;
    };

    struct DepthValue
    {
        f32 depth;
        u32 stencil;

        friend auto operator<=>(DepthValue const &, DepthValue const &) = default;
    };

    using ClearValue = Variant<std::array<f32, 4>, std::array<i32, 4>, std::array<u32, 4>, DepthValue>;

    struct AccessTypeFlagsProperties
    {
        using Data = u64;
    };
    using AccessTypeFlags = Flags<AccessTypeFlagsProperties>;
    struct AccessTypeFlagBits
    {
        static inline constexpr AccessTypeFlags NONE = {0x00000000};
        static inline constexpr AccessTypeFlags READ = {0x00008000};
        static inline constexpr AccessTypeFlags WRITE = {0x00010000};
        static inline constexpr AccessTypeFlags READ_WRITE = READ | WRITE;
    };

    [[nodiscard]] auto to_string(AccessTypeFlags flags) -> std::string;

    struct PipelineStageFlagsProperties
    {
        using Data = u64;
    };
    using PipelineStageFlags = Flags<PipelineStageFlagsProperties>;
    struct PipelineStageFlagBits
    {
        static inline constexpr PipelineStageFlags NONE = {0x00000000ull};

        static inline constexpr PipelineStageFlags TOP_OF_PIPE = {0x00000001ull};
        static inline constexpr PipelineStageFlags DRAW_INDIRECT = {0x00000002ull};
        static inline constexpr PipelineStageFlags VERTEX_SHADER = {0x00000008ull};
        static inline constexpr PipelineStageFlags TESSELLATION_CONTROL_SHADER = {0x00000010ull};
        static inline constexpr PipelineStageFlags TESSELLATION_EVALUATION_SHADER = {0x00000020ull};
        static inline constexpr PipelineStageFlags GEOMETRY_SHADER = {0x00000040ull};
        static inline constexpr PipelineStageFlags FRAGMENT_SHADER = {0x00000080ull};
        static inline constexpr PipelineStageFlags EARLY_FRAGMENT_TESTS = {0x00000100ull};
        static inline constexpr PipelineStageFlags LATE_FRAGMENT_TESTS = {0x00000200ull};
        static inline constexpr PipelineStageFlags COLOR_ATTACHMENT_OUTPUT = {0x00000400ull};
        static inline constexpr PipelineStageFlags COMPUTE_SHADER = {0x00000800ull};
        static inline constexpr PipelineStageFlags TRANSFER = {0x00001000ull};
        static inline constexpr PipelineStageFlags BOTTOM_OF_PIPE = {0x00002000ull};
        static inline constexpr PipelineStageFlags HOST = {0x00004000ull};
        static inline constexpr PipelineStageFlags ALL_GRAPHICS = {0x00008000ull};
        static inline constexpr PipelineStageFlags ALL_COMMANDS = {0x00010000ull};
        static inline constexpr PipelineStageFlags COPY = {0x100000000ull};
        static inline constexpr PipelineStageFlags RESOLVE = {0x200000000ull};
        static inline constexpr PipelineStageFlags BLIT = {0x400000000ull};
        static inline constexpr PipelineStageFlags CLEAR = {0x800000000ull};
        static inline constexpr PipelineStageFlags INDEX_INPUT = {0x1000000000ull};
        static inline constexpr PipelineStageFlags PRE_RASTERIZATION_SHADERS = {0x4000000000ull};
        static inline constexpr PipelineStageFlags TASK_SHADER = {0x00080000ull};
        static inline constexpr PipelineStageFlags MESH_SHADER = {0x00100000ull};
        static inline constexpr PipelineStageFlags ACCELERATION_STRUCTURE_BUILD = {0x02000000ull};
        static inline constexpr PipelineStageFlags RAY_TRACING_SHADER = {0x00200000ull};
    };

    [[nodiscard]] auto to_string(PipelineStageFlags flags) -> std::string;

    struct Access
    {
        PipelineStageFlags stages = PipelineStageFlagBits::NONE;
        AccessTypeFlags type = AccessTypeFlagBits::NONE;

        friend auto operator<=>(Access const &, Access const &) = default;
    };

    [[nodiscard]] DAXA_EXPORT_CXX auto operator|(Access const & a, Access const & b) -> Access;
    [[nodiscard]] DAXA_EXPORT_CXX auto operator&(Access const & a, Access const & b) -> Access;

    [[nodiscard]] DAXA_EXPORT_CXX auto to_string(Access access) -> std::string;

    namespace AccessConsts
    {
        static inline constexpr Access NONE = {.stages = PipelineStageFlagBits::NONE, .type = AccessTypeFlagBits::NONE};

        static inline constexpr Access TOP_OF_PIPE_READ = {.stages = PipelineStageFlagBits::TOP_OF_PIPE, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access DRAW_INDIRECT_READ = {.stages = PipelineStageFlagBits::DRAW_INDIRECT, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access VERTEX_SHADER_READ = {.stages = PipelineStageFlagBits::VERTEX_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access TESSELLATION_CONTROL_SHADER_READ = {.stages = PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access TESSELLATION_EVALUATION_SHADER_READ = {.stages = PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access GEOMETRY_SHADER_READ = {.stages = PipelineStageFlagBits::GEOMETRY_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access FRAGMENT_SHADER_READ = {.stages = PipelineStageFlagBits::FRAGMENT_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access EARLY_FRAGMENT_TESTS_READ = {.stages = PipelineStageFlagBits::EARLY_FRAGMENT_TESTS, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access LATE_FRAGMENT_TESTS_READ = {.stages = PipelineStageFlagBits::LATE_FRAGMENT_TESTS, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access COLOR_ATTACHMENT_OUTPUT_READ = {.stages = PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access COMPUTE_SHADER_READ = {.stages = PipelineStageFlagBits::COMPUTE_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access TRANSFER_READ = {.stages = PipelineStageFlagBits::TRANSFER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access BOTTOM_OF_PIPE_READ = {.stages = PipelineStageFlagBits::BOTTOM_OF_PIPE, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access HOST_READ = {.stages = PipelineStageFlagBits::HOST, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access ALL_GRAPHICS_READ = {.stages = PipelineStageFlagBits::ALL_GRAPHICS, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access READ = {.stages = PipelineStageFlagBits::ALL_COMMANDS, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access COPY_READ = {.stages = PipelineStageFlagBits::COPY, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access RESOLVE_READ = {.stages = PipelineStageFlagBits::RESOLVE, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access BLIT_READ = {.stages = PipelineStageFlagBits::BLIT, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access CLEAR_READ = {.stages = PipelineStageFlagBits::CLEAR, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access INDEX_INPUT_READ = {.stages = PipelineStageFlagBits::INDEX_INPUT, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access PRE_RASTERIZATION_SHADERS_READ = {.stages = PipelineStageFlagBits::PRE_RASTERIZATION_SHADERS, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access TASK_SHADER_READ = {.stages = PipelineStageFlagBits::TASK_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access MESH_SHADER_READ = {.stages = PipelineStageFlagBits::MESH_SHADER, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access ACCELERATION_STRUCTURE_BUILD_READ = {.stages = PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, .type = AccessTypeFlagBits::READ};
        static inline constexpr Access RAY_TRACING_SHADER_READ = {.stages = PipelineStageFlagBits::RAY_TRACING_SHADER, .type = AccessTypeFlagBits::READ};

        static inline constexpr Access TOP_OF_PIPE_WRITE = {.stages = PipelineStageFlagBits::TOP_OF_PIPE, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access DRAW_INDIRECT_WRITE = {.stages = PipelineStageFlagBits::DRAW_INDIRECT, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access VERTEX_SHADER_WRITE = {.stages = PipelineStageFlagBits::VERTEX_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access TESSELLATION_CONTROL_SHADER_WRITE = {.stages = PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access TESSELLATION_EVALUATION_SHADER_WRITE = {.stages = PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access GEOMETRY_SHADER_WRITE = {.stages = PipelineStageFlagBits::GEOMETRY_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access FRAGMENT_SHADER_WRITE = {.stages = PipelineStageFlagBits::FRAGMENT_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access EARLY_FRAGMENT_TESTS_WRITE = {.stages = PipelineStageFlagBits::EARLY_FRAGMENT_TESTS, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access LATE_FRAGMENT_TESTS_WRITE = {.stages = PipelineStageFlagBits::LATE_FRAGMENT_TESTS, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access COLOR_ATTACHMENT_OUTPUT_WRITE = {.stages = PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access COMPUTE_SHADER_WRITE = {.stages = PipelineStageFlagBits::COMPUTE_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access TRANSFER_WRITE = {.stages = PipelineStageFlagBits::TRANSFER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access BOTTOM_OF_PIPE_WRITE = {.stages = PipelineStageFlagBits::BOTTOM_OF_PIPE, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access HOST_WRITE = {.stages = PipelineStageFlagBits::HOST, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access ALL_GRAPHICS_WRITE = {.stages = PipelineStageFlagBits::ALL_GRAPHICS, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access WRITE = {.stages = PipelineStageFlagBits::ALL_COMMANDS, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access COPY_WRITE = {.stages = PipelineStageFlagBits::COPY, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access RESOLVE_WRITE = {.stages = PipelineStageFlagBits::RESOLVE, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access BLIT_WRITE = {.stages = PipelineStageFlagBits::BLIT, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access CLEAR_WRITE = {.stages = PipelineStageFlagBits::CLEAR, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access INDEX_INPUT_WRITE = {.stages = PipelineStageFlagBits::INDEX_INPUT, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access PRE_RASTERIZATION_SHADERS_WRITE = {.stages = PipelineStageFlagBits::PRE_RASTERIZATION_SHADERS, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access TASK_SHADER_WRITE = {.stages = PipelineStageFlagBits::TASK_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access MESH_SHADER_WRITE = {.stages = PipelineStageFlagBits::MESH_SHADER, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access ACCELERATION_STRUCTURE_BUILD_WRITE = {.stages = PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, .type = AccessTypeFlagBits::WRITE};
        static inline constexpr Access RAY_TRACING_SHADER_WRITE = {.stages = PipelineStageFlagBits::RAY_TRACING_SHADER, .type = AccessTypeFlagBits::WRITE};

        static inline constexpr Access TOP_OF_PIPE_READ_WRITE = {.stages = PipelineStageFlagBits::TOP_OF_PIPE, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access DRAW_INDIRECT_READ_WRITE = {.stages = PipelineStageFlagBits::DRAW_INDIRECT, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access VERTEX_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::VERTEX_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access TESSELLATION_CONTROL_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access TESSELLATION_EVALUATION_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access GEOMETRY_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::GEOMETRY_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access FRAGMENT_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::FRAGMENT_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access EARLY_FRAGMENT_TESTS_READ_WRITE = {.stages = PipelineStageFlagBits::EARLY_FRAGMENT_TESTS, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access LATE_FRAGMENT_TESTS_READ_WRITE = {.stages = PipelineStageFlagBits::LATE_FRAGMENT_TESTS, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access COLOR_ATTACHMENT_OUTPUT_READ_WRITE = {.stages = PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access COMPUTE_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::COMPUTE_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access TRANSFER_READ_WRITE = {.stages = PipelineStageFlagBits::TRANSFER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access BOTTOM_OF_PIPE_READ_WRITE = {.stages = PipelineStageFlagBits::BOTTOM_OF_PIPE, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access HOST_READ_WRITE = {.stages = PipelineStageFlagBits::HOST, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access ALL_GRAPHICS_READ_WRITE = {.stages = PipelineStageFlagBits::ALL_GRAPHICS, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access READ_WRITE = {.stages = PipelineStageFlagBits::ALL_COMMANDS, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access COPY_READ_WRITE = {.stages = PipelineStageFlagBits::COPY, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access RESOLVE_READ_WRITE = {.stages = PipelineStageFlagBits::RESOLVE, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access BLIT_READ_WRITE = {.stages = PipelineStageFlagBits::BLIT, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access CLEAR_READ_WRITE = {.stages = PipelineStageFlagBits::CLEAR, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access INDEX_INPUT_READ_WRITE = {.stages = PipelineStageFlagBits::INDEX_INPUT, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access PRE_RASTERIZATION_SHADERS_READ_WRITE = {.stages = PipelineStageFlagBits::PRE_RASTERIZATION_SHADERS, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access TASK_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::TASK_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access MESH_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::MESH_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access ACCELERATION_STRUCTURE_BUILD_READ_WRITE = {.stages = PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, .type = AccessTypeFlagBits::READ_WRITE};
        static inline constexpr Access RAY_TRACING_SHADER_READ_WRITE = {.stages = PipelineStageFlagBits::RAY_TRACING_SHADER, .type = AccessTypeFlagBits::READ_WRITE};
    } // namespace AccessConsts

    enum struct SamplerAddressMode
    {
        REPEAT = 0,
        MIRRORED_REPEAT = 1,
        CLAMP_TO_EDGE = 2,
        CLAMP_TO_BORDER = 3,
        MIRROR_CLAMP_TO_EDGE = 4,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct BorderColor
    {
        FLOAT_TRANSPARENT_BLACK = 0,
        INT_TRANSPARENT_BLACK = 1,
        FLOAT_OPAQUE_BLACK = 2,
        INT_OPAQUE_BLACK = 3,
        FLOAT_OPAQUE_WHITE = 4,
        INT_OPAQUE_WHITE = 5,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct CompareOp
    {
        NEVER = 0,
        LESS = 1,
        EQUAL = 2,
        LESS_OR_EQUAL = 3,
        GREATER = 4,
        NOT_EQUAL = 5,
        GREATER_OR_EQUAL = 6,
        ALWAYS = 7,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct BlendFactor
    {
        ZERO = 0,
        ONE = 1,
        SRC_COLOR = 2,
        ONE_MINUS_SRC_COLOR = 3,
        DST_COLOR = 4,
        ONE_MINUS_DST_COLOR = 5,
        SRC_ALPHA = 6,
        ONE_MINUS_SRC_ALPHA = 7,
        DST_ALPHA = 8,
        ONE_MINUS_DST_ALPHA = 9,
        CONSTANT_COLOR = 10,
        ONE_MINUS_CONSTANT_COLOR = 11,
        CONSTANT_ALPHA = 12,
        ONE_MINUS_CONSTANT_ALPHA = 13,
        SRC_ALPHA_SATURATE = 14,
        SRC1_COLOR = 15,
        ONE_MINUS_SRC1_COLOR = 16,
        SRC1_ALPHA = 17,
        ONE_MINUS_SRC1_ALPHA = 18,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct BlendOp
    {
        ADD = 0,
        SUBTRACT = 1,
        REVERSE_SUBTRACT = 2,
        MIN = 3,
        MAX = 4,
        MAX_ENUM = 0x7fffffff,
    };

    struct ColorComponentFlagsProperties
    {
        using Data = u32;
    };
    using ColorComponentFlags = Flags<ColorComponentFlagsProperties>;
    struct ColorComponentFlagBits
    {
        static inline constexpr ColorComponentFlags NONE = {0x00000000};
        static inline constexpr ColorComponentFlags R = {0x00000001};
        static inline constexpr ColorComponentFlags G = {0x00000002};
        static inline constexpr ColorComponentFlags B = {0x00000004};
        static inline constexpr ColorComponentFlags A = {0x00000008};
    };

    struct BlendInfo
    {
        BlendFactor src_color_blend_factor = BlendFactor::ONE;
        BlendFactor dst_color_blend_factor = BlendFactor::ZERO;
        BlendOp color_blend_op = BlendOp::ADD;
        BlendFactor src_alpha_blend_factor = BlendFactor::ONE;
        BlendFactor dst_alpha_blend_factor = BlendFactor::ZERO;
        BlendOp alpha_blend_op = BlendOp::ADD;
        ColorComponentFlags color_write_mask = ColorComponentFlagBits::R | ColorComponentFlagBits::G | ColorComponentFlagBits::B | ColorComponentFlagBits::A;

        friend auto operator<=>(BlendInfo const &, BlendInfo const &) = default;
    };

    enum struct TesselationDomainOrigin
    {
        LOWER_LEFT = 0,
        UPPER_LEFT = 1,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct ConservativeRasterizationMode
    {
        DISABLED = 0,
        OVERESTIMATE = 1,
        UNDERESTIMATE = 2,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct PrimitiveTopology
    {
        POINT_LIST = 0,
        LINE_LIST = 1,
        LINE_STRIP = 2,
        TRIANGLE_LIST = 3,
        TRIANGLE_STRIP = 4,
        TRIANGLE_FAN = 5,
        LINE_LIST_WITH_ADJACENCY = 6,
        LINE_STRIP_WITH_ADJACENCY = 7,
        TRIANGLE_LIST_WITH_ADJACENCY = 8,
        TRIANGLE_STRIP_WITH_ADJACENCY = 9,
        PATCH_LIST = 10,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct PolygonMode
    {
        FILL = 0,
        LINE = 1,
        POINT = 2,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct FrontFaceWinding
    {
        COUNTER_CLOCKWISE = 0,
        CLOCKWISE = 1,
        MAX_ENUM = 0x7fffffff,
    };

    struct FaceCullFlagsProperties
    {
        using Data = u32;
    };
    using FaceCullFlags = Flags<FaceCullFlagsProperties>;
    struct FaceCullFlagBits
    {
        static inline constexpr FaceCullFlags NONE = {0x00000000};
        static inline constexpr FaceCullFlags FRONT_BIT = {0x00000001};
        static inline constexpr FaceCullFlags BACK_BIT = {0x00000002};
        static inline constexpr FaceCullFlags FRONT_AND_BACK = {0x00000003};
    };

    enum struct AttachmentLoadOp
    {
        LOAD = 0,
        CLEAR = 1,
        DONT_CARE = 2,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct AttachmentStoreOp
    {
        STORE = 0,
        DONT_CARE = 1,
        MAX_ENUM = 0x7fffffff,
    };

    struct ViewportInfo
    {
        f32 x = {};
        f32 y = {};
        f32 width = {};
        f32 height = {};
        f32 min_depth = {};
        f32 max_depth = {};
    };

    struct Rect2D
    {
        i32 x = {};
        i32 y = {};
        u32 width = {};
        u32 height = {};
    };
    struct MemoryRequirements
    {
        u64 size = {};
        u64 alignment = {};
        u32 memory_type_bits = {};
    };

    struct MemoryBlockInfo
    {
        MemoryRequirements requirements = {};
        MemoryFlags flags = {};
    };

    struct DAXA_EXPORT_CXX MemoryBlock : ManagedPtr<MemoryBlock, daxa_MemoryBlock>
    {
        MemoryBlock() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() -> MemoryBlockInfo const &;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct TimelineQueryPoolInfo
    {
        u32 query_count = {};
        SmallString name = {};
    };

    struct DAXA_EXPORT_CXX TimelineQueryPool : ManagedPtr<TimelineQueryPool, daxa_TimelineQueryPool>
    {
        TimelineQueryPool() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> TimelineQueryPoolInfo const &;

        [[nodiscard]] auto get_query_results(u32 start_index, u32 count) -> std::vector<u64>;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    enum struct IndexType
    {
        uint16 = 0,
        uint32 = 1,
        uint8 = 1000265000,
        none = 1000165000,
    };

    // TODO: distinguish between GENERAL(raygen, miss & callable) cause shader handles must be set in order (raygen, miss, hit, callable)?
    enum struct ShaderGroup
    {
        GENERAL = 0,
        TRIANGLES_HIT_GROUP = 1,
        PROCEDURAL_HIT_GROUP = 2,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct InvocationReorderMode
    {
        NO_REORDER = 0,
        ALLOW_REORDER = 1,
        MAX_ENUM = 0x7fffffff,
    };

    enum struct QueueFamily
    {
        MAIN,
        COMPUTE,
        TRANSFER
    };

    auto to_string(QueueFamily family) -> std::string_view;
} // namespace daxa
