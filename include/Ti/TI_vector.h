//Instead of reallocate every time, use over-allocation to reduce memory copy.

#if !defined(TI_VECTOR_H)
#define TI_VECTOR_H

#include <vector>
#define VECTOR_MAX_CHUNK_SIZE 1024
template <typename T>
class TI_vector {
public:
    CUDA_CALLABLE_MEMBER TI_vector<T>(unsigned p_sizeof_chunk=VECTOR_MAX_CHUNK_SIZE) {
        sizeof_chunk = p_sizeof_chunk;
        #ifdef __CUDA_ARCH__
            main = (T*) malloc(sizeof_chunk*sizeof(T));
            flag = 1;
        #else
            cudaMalloc( &main , sizeof_chunk*sizeof(T) );
            flag = 2;
        #endif
        num_main = 0;
    }
    CUDA_CALLABLE_MEMBER ~TI_vector<T>() {
        //TODO: Free main
    }
    TI_vector<T>(const std::vector<T>& p, unsigned p_sizeof_chunk=VECTOR_MAX_CHUNK_SIZE) {
        sizeof_chunk = p_sizeof_chunk;
        cudaMalloc( &main , sizeof_chunk*sizeof(T) );
        set(p);
    }
	TI_vector& operator=(const std::vector<T>& p) { return set(p); }
	TI_vector& set(const std::vector<T>& p) {
        flag = 2;
        num_main = p.size();
        T* temp = (T*) malloc(num_main*sizeof(T));
        for (unsigned i=0;i<num_main;i++) {
            temp[i] = p[i];
        }
        cudaMemcpy(main, temp, num_main*sizeof(T), cudaMemcpyHostToDevice);
        delete temp;
        return *this; 
    }

    CUDA_DEVICE void push_back(T p) {
        unsigned current_cursor;
        if (flag==2) current_cursor = atomicAdd(&num_main,1);
        else current_cursor = num_main++;
        // printf("current_cursor: %d\n", current_cursor);
        main[current_cursor] = p;
        if (num_main >= sizeof_chunk - 10) {
            printf("ALERT: TI_vector default chunk size not enough!");
            num_main = 0;
        }
    };

    CUDA_DEVICE T get (unsigned index) {
        return main[index];
    }
    CUDA_DEVICE T &operator[] (unsigned index) {
        return main[index];
    };
    CUDA_CALLABLE_MEMBER unsigned size() {
        return num_main;
    };
    CUDA_DEVICE void clear() {
        num_main = 0;
    }
    CUDA_DEVICE bool find(T value) {
        for (unsigned i=0;i<num_main;i++) {
            if (main[i]==value) {
                return true;
            }
        }
        return false;
    }
/* data */
    unsigned sizeof_chunk;
    T* main;
    unsigned num_main;
    int flag; //flag for type of memory management. 1: create on host and memcpy to dev; 2. create on dev and do not support mutex. (TODO: I don't know why flag 2 don't support atomicAdd().)
};

#endif // TI_VECTOR_H
