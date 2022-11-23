/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include "mlir_test/dialect.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/TypeID.h"

#include "mlir_test/dialect.cc.inc"
#include "mlir_test/type.h"
namespace mlir {
namespace toy {
class ToyHandleType;
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir_test/ops.cc.inc"
      >();
  addTypes<ToyHandleType>();
  //  addInterfaces<ToyInlinerInterface>();
}
}  // namespace toy
}  // namespace mlir
