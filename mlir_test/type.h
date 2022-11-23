/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#ifndef MLIR_TEST_TYPE_H_
#define MLIR_TEST_TYPE_H_
#include "mlir/IR/Types.h"
namespace mlir {
namespace toy {
namespace detail {
struct ToyHandleTypeStorage : public TypeStorage {
  using KeyTy = int;
  explicit ToyHandleTypeStorage(const KeyTy &key) {}
  static ToyHandleTypeStorage *construct(TypeStorageAllocator &allocator,  // NOLINT
                                           const KeyTy &key);
  bool operator==(const KeyTy &key) const;
};
}  // namespace detail
class ToyHandleType
    : public Type::TypeBase<ToyHandleType, Type, detail::ToyHandleTypeStorage> {
 public:
  /// Inherit base constructors.
  using Base::Base;

  static ToyHandleType get(MLIRContext *ctx);
};
}  // namespace toy
}  // namespace mlir
#endif  // MLIR_TEST_TYPE_H_
