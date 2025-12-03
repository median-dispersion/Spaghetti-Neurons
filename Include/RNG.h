#ifndef RNG_H
#define RNG_H

#include <random>
#include <type_traits>

namespace rng {

    template <typename T>
    T range(T minimum, T maximum) {

        static thread_local std::random_device device;
        static thread_local std::mt19937 generator(device());

        if constexpr (std::is_integral<T>::value) {

            std::uniform_int_distribution<T> distribution(minimum, maximum);
            return distribution(generator);

        } else if constexpr (std::is_floating_point<T>::value) {

            std::uniform_real_distribution<T> distribution(minimum, maximum);
            return distribution(generator);
        
        } else {

            static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

        }

    }

};

#endif