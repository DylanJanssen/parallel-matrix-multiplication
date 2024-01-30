#include <iostream> 
#include "parallel_matrix.hpp"


int main()
{
    for (int size = 32; size <= 2048; size *= 2) 
    {
        Matrix<int> A(size), B(size), C(size), D(size); 
        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < A.size(); j++)
            {
                A[i][j] = i*j;
                B[i][j] = i*j; 
            }
        }
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply(A, B, C); 
        auto end = std::chrono::high_resolution_clock::now();
        auto single_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
        std::cout << size << " Single-threaded: " << single_duration << "s" << std::endl; 
        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < A.size(); j++)
            {
                A[i][j] = i*j;
                B[i][j] = i*j; 
            }
        }
        start = std::chrono::high_resolution_clock::now();
        multithreaded_matrix_multiply(A, B, D);
        end = std::chrono::high_resolution_clock::now();
        auto multi_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
        std::cout << size << " Multi-threaded: " << multi_duration << "s" << std::endl;
        std::cout << "Speedup: " << single_duration / multi_duration << "x" << std::endl; 
        if (C != D)
            std::cout << "Matrix multiplication incorrect" << std::endl; 
    }
}