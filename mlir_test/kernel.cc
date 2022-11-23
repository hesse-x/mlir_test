#include <stdio.h>
#define EXPORT __attribute__((visibility("default")))
struct Handle;
using Handle_t = Handle *;

struct Handle {
  int val = 10086;
};

extern "C" {
EXPORT Handle_t CreateToyHandle();
EXPORT void DestroyToyHandle(Handle_t handle);
EXPORT void FuncA(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size);
EXPORT void FuncB(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size);
EXPORT void FuncC(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size);
}

namespace {
template <int Type, int N>
struct RepeatCall {
  static void Do() {
    printf("%d,", N);
    RepeatCall<Type, N - 1>::Do();
  }
};

template <int Type>
struct RepeatCall<Type, 0> {
  static void Do() {
    printf("\nType is : %d\n,", Type);
  }
};
}  // namespace

Handle_t CreateToyHandle() {
  return new Handle();
}
void DestroyToyHandle(Handle_t handle) {
  delete handle;
}
void FuncA(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size) {
  RepeatCall<0, 1000>::Do(); 
  printf("handle value : %d, in ptr : %p, out ptr : %p, in len : %ld, out len : %ld\n", handle->val, in, out, in_size, out_size);
}
void FuncB(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size) {
  RepeatCall<1, 1000>::Do(); 
  printf("handle value : %d, in ptr : %p, out ptr : %p, in len : %ld, out len : %ld\n", handle->val, in, out, in_size, out_size);
}
void FuncC(Handle_t handle, double *in, size_t in_size, double *out, size_t out_size) {
  RepeatCall<2, 1000>::Do(); 
  printf("handle value : %d, in ptr : %p, out ptr : %p, in len : %ld, out len : %ld\n", handle->val, in, out, in_size, out_size);
}
