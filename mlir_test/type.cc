/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include "mlir/IR/DialectImplementation.h"
#include "mlir_test/dialect.h"
#include "mlir_test/type.h"
namespace mlir {
namespace toy {
namespace detail {
ToyHandleTypeStorage *ToyHandleTypeStorage::construct(TypeStorageAllocator &allocator,
                                                          const KeyTy &key) {
  return new (allocator.allocate<ToyHandleTypeStorage>()) ToyHandleTypeStorage(key);
}
bool ToyHandleTypeStorage::operator==(const KeyTy &key) const {
  return true;
}

void printType(mlir::Type type, AsmPrinter &out) {  // NOLINT
  if (type.isa<ToyHandleType>()) {
    out << StringRef("toy_handle");
  }
}
}  // namespace detail

ToyHandleType ToyHandleType::get(MLIRContext *ctx) {
  return Base::get(ctx);
}

void ToyDialect::printType(mlir::Type type, DialectAsmPrinter &out) const {  // NOLINT
  detail::printType(type, out);
}

::mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &) const {  // NOLINT
  return ToyHandleType();
}
}  // namespace toy
}  // namespace mlir
