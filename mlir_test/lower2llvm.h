/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#ifndef MLIR_TEST_LOWER22LLVM_H_
#define MLIR_TEST_LOWER22LLVM_H_
#include <memory>
namespace mlir {
class Pass;
namespace toy {
std::unique_ptr<::mlir::Pass> CreateLowerToLLVMPass();
}  // namespace toy
}  // namespace mlir
#endif  // MLIR_TEST_LOWER22LLVM_H_
