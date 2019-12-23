#if !defined(TI_VECTOR_H)
#define TI_VECTOR_H

template <typename T>
class TI_vector {
public:
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
