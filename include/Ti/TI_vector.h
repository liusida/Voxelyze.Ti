#if !defined(TI_VECTOR_H)
#define TI_VECTOR_H

#include <vector>

template <typename T>
class TI_vector {
public:
    TI_vector<T>(std::vector<T> p) {
        num_main = p.size();
        cudaMalloc( &main , num_main*sizeof(T) );
        T* temp = (T*) malloc(num_main*sizeof(T));
        for (unsigned i=0;i<num_main;i++) {
            temp[i] = p[i]; //p._ptr?
        }
        cudaMemcpy(main, temp, num_main*sizeof(T), cudaMemcpyHostToDevice);
        delete temp;
    }
    CUDA_CALLABLE_MEMBER TI_vector<T>() : sizeof_chunk(16), main(NULL), num_main(0) {};
    CUDA_CALLABLE_MEMBER void push_back(T p) {
        //TODO: instead of reallocate every time, use over-allocation to reduce memory copy.
        T* tmp = (T*) new long [num_main+1];
        if (main!=NULL) {
            memcpy(tmp, main, num_main*sizeof(T));
            delete main;
        }
        main = tmp;
        main[num_main] = p;
        num_main++;
    };
    CUDA_CALLABLE_MEMBER T get (unsigned index) {
        return main[index];
    }
    CUDA_CALLABLE_MEMBER T &operator[] (unsigned index) {
        return main[index];
    };
    CUDA_CALLABLE_MEMBER unsigned size() {
        return num_main;
    };
    CUDA_CALLABLE_MEMBER void clear() {
        delete main;
        main = NULL;
        num_main = 0;
    }
/* data */
    unsigned sizeof_chunk;
    T* main;
    unsigned num_main;
};

#endif // TI_VECTOR_H
