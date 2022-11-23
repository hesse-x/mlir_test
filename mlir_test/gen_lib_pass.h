/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#ifndef MLIR_TEST_GEN_DYNAMIC_LIB_H_
#define MLIR_TEST_GEN_DYNAMIC_LIB_H_
#include <memory>
#include "mlir_test/utils.h"
namespace mlir {
class Pass;
namespace toy {
std::unique_ptr<::mlir::Pass> CreateGenLibraryPass(std::unique_ptr<::toy::FileObject> *lib_file);
}  // namespace toy
}  // namespace mlir
#endif  // MLIR_TEST_GEN_DYNAMIC_LIB_H_
