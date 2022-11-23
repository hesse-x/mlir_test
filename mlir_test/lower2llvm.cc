/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include "mlir_test/lower2llvm.h"
#include <string>
#include <utility>
#include "mlir_test/dialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir {
namespace toy {
namespace {
template <typename LoweringClass, typename Op>
struct ToyOpLowering : public ConvertOpToLLVMPattern<Op> {
 protected:
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  static SymbolRefAttr GetOrInsertFunc(PatternRewriter &rewriter,  // NOLINT
                                       ModuleOp module,
                                       const StringRef symbol) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(symbol)) {
      return SymbolRefAttr::get(context, symbol);
    }
    SymbolRefAttr ret = LoweringClass::InsertFunc(rewriter, module, symbol);
    return ret;
  }
};

struct CreateToyHandleOpLowering
    : public ToyOpLowering<CreateToyHandleOpLowering, CreateToyHandleOp> {
  using ToyOpLowering<CreateToyHandleOpLowering, CreateToyHandleOp>::ToyOpLowering;
  LogicalResult matchAndRewrite(CreateToyHandleOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MLIRContext *context = parentModule->getContext();
    // Get a symbol reference to the printf function, inserting it if necessary.
    StringRef symbol("CreateToyHandle");
    auto FuncRef = GetOrInsertFunc(rewriter, parentModule, symbol);
    SmallVector<Value, 1> args;
    auto llvmI8Ty = IntegerType::get(context, 8);
    auto llvmVoidPtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
    rewriter.replaceOpWithNewOp<CallOp>(op, FuncRef, llvmVoidPtrTy, args);
    return success();
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr InsertFunc(PatternRewriter &rewriter,  // NOLINT
                                      ModuleOp module,
                                      StringRef symbol) {
    auto *context = module.getContext();
    // Create a function declaration for printf, the signature is:
    //   * `void *()`
    auto llvmI8Ty = IntegerType::get(context, 8);
    auto llvmVoidPtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidPtrTy, llvm::ArrayRef<mlir::Type>{},
                                                  /*isVarArg=*/false);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), symbol, llvmFnType);
    return SymbolRefAttr::get(context, symbol);
  }
};

struct DestroyToyHandleOpLowering
    : public ToyOpLowering<DestroyToyHandleOpLowering, DestroyToyHandleOp> {
  using ToyOpLowering<DestroyToyHandleOpLowering, DestroyToyHandleOp>::ToyOpLowering;
  LogicalResult matchAndRewrite(DestroyToyHandleOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    // Get a symbol reference to the printf function, inserting it if necessary.
    StringRef symbol("DestroyToyHandle");
    auto FuncRef = GetOrInsertFunc(rewriter, parentModule, symbol);
    SmallVector<Value, 1> args;
    args.emplace_back(adaptor.handle());
    rewriter.replaceOpWithNewOp<CallOp>(op, FuncRef, llvm::None, args);
    return success();
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr InsertFunc(PatternRewriter &rewriter,  // NOLINT
                                      ModuleOp module,
                                      StringRef symbol) {
    auto *context = module.getContext();
    // Create a function declaration for printf, the signature is:
    //   * `void (Handle*)`
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmI8Ty = IntegerType::get(context, 8);
    auto llvmVoidPtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, llvmVoidPtrTy, /*isVarArg=*/false);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), symbol, llvmFnType);
    return SymbolRefAttr::get(context, symbol);
  }
};

struct ToyFuncAOpLowering : public ToyOpLowering<ToyFuncAOpLowering, ToyFuncAOp> {
  using ToyOpLowering<ToyFuncAOpLowering, ToyFuncAOp>::ToyOpLowering;
  LogicalResult matchAndRewrite(ToyFuncAOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    // Get a symbol reference to the printf function, inserting it if necessary.
    StringRef symbol("FuncA");
    auto ToyRef = GetOrInsertFunc(rewriter, parentModule, symbol);

    SmallVector<Value, 1> len;
    SmallVector<Value, 4> args;
    args.emplace_back(adaptor.handle());
    {
      MemRefDescriptor d(adaptor.input());
      args.push_back(d.alignedPtr(rewriter, loc));
      len.push_back(d.size(rewriter, loc, 0));
      args.push_back(d.size(rewriter, loc, 0));
    }
    {
      MemRefDescriptor d(adaptor.output());
      args.push_back(d.alignedPtr(rewriter, loc));
      args.push_back(d.size(rewriter, loc, 0));
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, ToyRef, llvm::None, args);
    return success();
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr InsertFunc(PatternRewriter &rewriter,  // NOLINT
                                      ModuleOp module,
                                      StringRef symbol) {
    auto *context = module.getContext();
    // Create a function declaration for printf, the signature is:
    //   * `void (handle *, double *, int64, double *, int64)`
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmI8Ty = IntegerType::get(context, 8);
    auto llvmF64Ty = Float64Type::get(context);
    auto llvmVoidPtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
    auto llvmF64PtrTy = LLVM::LLVMPointerType::get(llvmF64Ty);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmVoidTy, {llvmVoidPtrTy, llvmF64PtrTy, llvmI64Ty, llvmF64PtrTy, llvmI64Ty},
        /*isVarArg=*/false);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), symbol, llvmFnType);
    return SymbolRefAttr::get(context, symbol);
  }
};

class ToyTypeConverter : public LLVMTypeConverter {
 public:
  explicit ToyTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](ToyHandleType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });
  }
};

struct ToyToLLVMLoweringPass : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    ToyTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateLoopToStdConversionPatterns(patterns);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    patterns.add<CreateToyHandleOpLowering>(typeConverter);
    patterns.add<ToyFuncAOpLowering>(typeConverter);
    patterns.add<DestroyToyHandleOpLowering>(typeConverter);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
}  // namespace toy
}  // namespace mlir
