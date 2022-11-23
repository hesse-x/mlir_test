1.编译mlir_test

bazel build mlir_test/...


2.编译kernel.o

clang++ mlir_test/kernel.cc -fPIC -shared -fvisibility=hidden -flto -c


3.使用llvm-arr生成静态库

llvm-ar-6.0 r libkernel.a kernel.o

rm kernel.o


4.执行测试

./bazel-bin/mlir_test/gen_lib_test
