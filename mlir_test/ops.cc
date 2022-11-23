/*--------------------------------------------------------------
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 -------------------------------------------------------------*/
#include "mlir_test/dialect.h"
#include <assert.h>

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
namespace mlir {
namespace toy {
}  // namespace toy
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir_test/ops.cc.inc"
