/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include <cstdio>
#include <numeric>
#include <vector>
#include <string>
#include "mlir_test/dialect.h"
#include "mlir_test/utils.h"
#include "mlir_test/gen_lib_pass.h"
#include "mlir_test/lower2llvm.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "gtest/gtest.h"  // NOLINT
std::vector<std::string> SplitEnv(const char *ctx) {
  const char *ch = ctx;
  std::vector<std::string> ret(1);
  auto it = &ret.back();
  while (*ch != '\0') {
    if (*ch == ':') {
      if (!ret.back().empty()) {
        ret.emplace_back();
        it = &ret.back();
      }
    } else {
      it->push_back(*ch);
    }
    ++ch;
  }
  return ret;
}
namespace mlir {
namespace toy {
#define __location builder.getUnknownLoc()
constexpr size_t len = 4;
TEST(GenLibTest, NormalTest) {
  MLIRContext ctx;
  ctx.getOrLoadDialect<ToyDialect>();
  ctx.getOrLoadDialect<StandardOpsDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<arith::ArithmeticDialect>();
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();

  OpBuilder builder(&ctx);
  ModuleOp module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  llvm::SmallVector<Type, 4> argTypes;
  Type f64_t = builder.getF64Type();
  argTypes.emplace_back(MemRefType::get({len}, f64_t));
  argTypes.emplace_back(MemRefType::get({6}, f64_t));
  auto funcType = builder.getFunctionType(argTypes, llvm::None);
  auto main_func = FuncOp::create(__location, "MyTest", funcType);
  module.push_back(main_func);
  auto &entryBlock = *main_func.addEntryBlock();
  //  Block &entryBlock = main_func.front();
  builder.setInsertionPointToStart(&entryBlock);
  auto handle = builder.create<CreateToyHandleOp>(__location, ToyHandleType::get(&ctx));
  builder.create<ToyFuncAOp>(__location, handle, main_func.getArgument(0),
                               main_func.getArgument(1));
  builder.create<DestroyToyHandleOp>(__location, handle);
  builder.create<ReturnOp>(__location);

  module.dump();
  // ---------------------------------------------------------
  std::unique_ptr<::toy::FileObject> tmp;

  PassManager pm(&ctx);
  applyPassManagerCLOptions(pm);
  OpPassManager &optPM = pm.nest<FuncOp>();
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createCSEPass());
  pm.addPass(CreateLowerToLLVMPass());
  pm.addPass(CreateGenLibraryPass(&tmp));
  auto ret = failed(pm.run(module));
  module.dump();
  EXPECT_FALSE(ret);
  ASSERT_TRUE(tmp);
  tmp->Keep();
}
}  // namespace toy
}  // namespace mlir
