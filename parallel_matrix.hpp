#ifndef PARALLEL_MATRIX_H 
#define PARALLEL_MATRIX_H 

/*
    Parallel matrix multiplication 
    Created by Dylan Janssen
    Currently only supports square matrices 
    Was developed as an example of multithreading for teaching purposes 
*/

#include <iostream> 
#include <thread> 
#include <vector> 

template <typename T> 
class Matrix 
{
private: 
    std::shared_ptr<T> data; // 1D array for contiguous memory 
    int sz; // size of the matrix 
    int stride; // row length
    int offset = 0; // used for sub-matrices 
    // private constructor for sub-matrices
    Matrix(int start_row, int start_col, int size, Matrix &orig) : 
        sz(size), stride(orig.stride), data(orig.data), offset(orig.offset + start_row * orig.stride + start_col) {}

public: 
    Matrix(int size) : sz(size), stride(size), data(new T[size * size]) {}

    T* operator[](int row){ return &(data.get()+offset)[row * stride]; }
    int size() { return sz; }
    std::vector<Matrix<T>> get_submatrices(); // returns a std::vector of 4 submatrices that contain reference to original data 
    bool operator==(Matrix<T> &rhs);
    bool operator!=(Matrix<T> &rhs) { return !((*this) == rhs); }
    template<typename U>
    friend std::ostream& operator<<(std::ostream &os, Matrix<U> &m); 
};

template <typename T> 
void matrix_multiply(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C);

template <typename T> 
void matrix_addition(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C);

template <typename T> 
void multithreaded_matrix_multiply(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C); 

template <typename T> 
void matrix_multiply(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C)
{
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A.size(); j++)
        {
            C[i][j] = 0; 
            for (int k = 0; k < A.size(); k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

template <typename T> 
void matrix_addition(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C)
{
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A.size(); j++)
            C[i][j] = A[i][j] + B[i][j]; 
}

template <typename U> 
std::ostream& operator<<(std::ostream &os, Matrix<U> &m)
{
    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m.size(); j++)
            os << m[i][j] << ' ';
        os << std::endl; 
    }
    return os; 
}

template <typename T> 
std::vector<Matrix<T>> Matrix<T>::get_submatrices()
{
    std::vector<Matrix<T>> subs;
    for (int i = 0; i < 4; i++)
    {
        int row = (i/2) * sz/2; // 0   0 1/2 1/2
        int col = (i%2) * sz/2; // 0 1/2   0 1/2
        subs.push_back(Matrix<T>(row, col, sz/2, *this)); 
    }
    return subs; 
}

template <typename T> 
void multithreaded_matrix_multiply(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C)
{
    Matrix<T> D(A.size()); // temporary memory 
    auto A_subs = A.get_submatrices(); 
    auto B_subs = B.get_submatrices(); 
    auto C_subs = C.get_submatrices(); 
    auto D_subs = D.get_submatrices(); 
    std::vector<thread> threads; 
    if (A_subs[0].size() >= 256)
    {
        for (int i = 0; i < 4; i++)
        {
            int A_idx = i / 2 * 2; 
            int B_idx = i % 2; 
            threads.push_back(thread(multithreaded_matrix_multiply<T>, ref(A_subs[A_idx]), ref(B_subs[B_idx]), ref(C_subs[i])));
            threads.push_back(thread(multithreaded_matrix_multiply<T>, ref(A_subs[A_idx+1]), ref(B_subs[B_idx+2]), ref(D_subs[i])));
        }
    }
    else 
    {
        for (int i = 0; i < 4; i++)
        {
            int A_idx = i / 2 * 2; 
            int B_idx = i % 2; 
            threads.push_back(thread(matrix_multiply<T>, ref(A_subs[A_idx]), ref(B_subs[B_idx]), ref(C_subs[i])));
            threads.push_back(thread(matrix_multiply<T>, ref(A_subs[A_idx+1]), ref(B_subs[B_idx+2]), ref(D_subs[i])));
        }
    }
    for (auto &t : threads)
        t.join(); 
    threads.clear(); 
    for (int i = 0; i < 4; i++)
        threads.push_back(thread(matrix_addition<T>, ref(C_subs[i]), ref(D_subs[i]), ref(C_subs[i])));
    for (auto &t : threads)
        t.join(); 
}
template <typename T> 
bool Matrix<T>::operator==(Matrix<T> &rhs)
{
    for (int i = 0; i < size(); i++)
        for (int j = 0; j < size(); j++)
            if ((*this)[i][j] != rhs[i][j])
                return false;
    return true;  
}

#endif 