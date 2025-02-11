#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace utilities {

    namespace matrix {
        constexpr bool RowMajor = true;
        constexpr bool ColumnMajor = false; 

        template<typename T, std::size_t Rows, std::size_t Cols>
        std::array<T, Rows * Cols> transformToColumnMajor(const std::array<T, Rows * Cols>& arr) {
            std::array<T, Rows * Cols> result;
            
            for (std::size_t i = 0; i < Rows; ++i) {
                for (std::size_t j = 0; j < Cols; ++j) {
                    result[j * Rows + i] = arr[i * Cols + j];
                }
            }
            
            return result;
        }

        template<typename T, std::size_t Rows, std::size_t Cols>
        std::array<T, Rows * Cols> transformToRowMajor(const std::array<T, Rows * Cols>& arr) {
            std::array<T, Rows * Cols> result;

            for (std::size_t i = 0; i < Rows; ++i) {
                for (std::size_t j = 0; j < Cols; ++j) {
                    result[i * Cols + j] = arr[j * Rows + i];
                }
            }

            return result;
        }

        // template<typename T, std::size_t Rows, std::size_t Cols, bool ToRowMajor>
        // constexpr std::array<T, Rows * Cols> transformMatrix(const std::array<T, Rows * Cols>& arr) {
        //     std::array<T, Rows * Cols> result;

        //     if constexpr (ToRowMajor) {
        //         // Convert from Column Major to Row Major
        //         for (std::size_t i = 0; i < Rows; ++i) {
        //             for (std::size_t j = 0; j < Cols; ++j) {
        //                 result[i * Cols + j] = arr[j * Rows + i];
        //             }
        //         }
        //     } else {
        //         // Convert from Row Major to Column Major
        //         for (std::size_t i = 0; i < Rows; ++i) {
        //             for (std::size_t j = 0; j < Cols; ++j) {
        //                 result[j * Rows + i] = arr[i * Cols + j];
        //             }
        //         }
        //     }

        //     return result;
        // }

        template<typename T, std::size_t Rows, std::size_t Cols, bool ToRowMajor>
        constexpr std::array<T, Rows * Cols> transformMatrix(const T* arr) {
            std::array<T, Rows * Cols> result;

            if constexpr (ToRowMajor) {
                // Convert from Column Major to Row Major
                for (std::size_t i = 0; i < Rows; ++i) {
                    for (std::size_t j = 0; j < Cols; ++j) {
                        result[i * Cols + j] = arr[j * Rows + i];
                    }
                }
            } else {
                // Convert from Row Major to Column Major
                for (std::size_t i = 0; i < Rows; ++i) {
                    for (std::size_t j = 0; j < Cols; ++j) {
                        result[j * Rows + i] = arr[i * Cols + j];
                    }
                }
            }

            return result;
        }

    }

    // void transformToColumnMajor(int* arr, const int rows, const int cols) {
    //     // Create a temporary vector to store the transformed array
    //     std::vector<int> temp(rows * cols);
        
    //     // Copy input array to temp vector
    //     std::copy(arr, arr + (rows * cols), temp.begin());
        
    //     // Transform from row major to column major
    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < cols; j++) {
    //             // Convert row major index to column major index
    //             // Row major: i * cols + j
    //             // Column major: j * rows + i
    //             arr[j * rows + i] = temp[i * cols + j];
    //         }
    //     }
    // }

    // template<typename T, std::size_t Rows, std::size_t Cols>
    // void transformToColumnMajor(T* arr) {
    //     // Create a temporary array with compile-time size
    //     std::array<T, Rows * Cols> temp;
        
    //     // Copy input array to temp
    //     std::copy(arr, arr + (Rows * Cols), temp.begin());
        
    //     // Transform from row major to column major
    //     for (std::size_t i = 0; i < Rows; i++) {
    //         for (std::size_t j = 0; j < Cols; j++) {
    //             // Convert row major index to column major index
    //             arr[j * Rows + i] = temp[i * Cols + j];
    //         }
    //     }
    // }

    // // For the absolute best performance:
    // template<typename T, std::size_t Rows, std::size_t Cols>
    // std::array<T, Rows * Cols> transformToColumnMajorFast(const T* __restrict arr) {
    //     std::array<T, Rows * Cols> result;
    //     T* __restrict res_ptr = result.data();
        
    //     for (std::size_t i = 0; i < Rows; ++i) {
    //         for (std::size_t j = 0; j < Cols; ++j) {
    //             res_ptr[j * Rows + i] = arr[i * Cols + j];
    //         }
    //     }
        
    //     return result;
    // }

    // template<typename T, std::size_t Rows, std::size_t Cols>
    // void compiler_optimized(T* __restrict arr) {
    //     static_assert(std::is_arithmetic<T>::value, "Type must be numeric");
    //     static_assert(Rows > 0 && Cols > 0, "Dimensions must be positive");

    //     // Align temporary storage to cache line boundary
    //     alignas(64) std::array<T, Rows * Cols> temp;
        
    //     // Fast copy to aligned temporary storage
    //     #pragma unroll 4
    //     for (std::size_t i = 0; i < Rows * Cols; i += 4) {
    //         if constexpr (Rows * Cols - i >= 4) {
    //             temp[i] = arr[i];
    //             temp[i + 1] = arr[i + 1];
    //             temp[i + 2] = arr[i + 2];
    //             temp[i + 3] = arr[i + 3];
    //         } else {
    //             for (std::size_t j = i; j < Rows * Cols; ++j) {
    //                 temp[j] = arr[j];
    //             }
    //         }
    //     }

    //     // Process blocks of elements to improve cache utilization
    //     constexpr std::size_t BLOCK_SIZE = 16;
        
    //     // Block-wise transformation
    //     #pragma GCC ivdep
    //     for (std::size_t jb = 0; jb < Cols; jb += BLOCK_SIZE) {
    //         const std::size_t jend = std::min(jb + BLOCK_SIZE, Cols);
    //         for (std::size_t ib = 0; ib < Rows; ib += BLOCK_SIZE) {
    //             const std::size_t iend = std::min(ib + BLOCK_SIZE, Rows);
                
    //             // Process each block
    //             #pragma unroll 4
    //             for (std::size_t j = jb; j < jend; ++j) {
    //                 for (std::size_t i = ib; i < iend; ++i) {
    //                     arr[j * Rows + i] = temp[i * Cols + j];
    //                 }
    //             }
    //         }
    //     }
    // }
    

}