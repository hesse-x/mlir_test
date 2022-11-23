/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#ifndef MLIR_TEST_DIALECT_H_
#define MLIR_TEST_DIALECT_H_
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir_test/type.h"
#include "mlir_test/dialect.h.inc"
#define GET_OP_CLASSES
#include "mlir_test/ops.h.inc"
#endif  // MLIR_TEST_DIALECT_H_
