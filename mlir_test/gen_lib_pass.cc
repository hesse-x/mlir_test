/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include "mlir_test/gen_lib_pass.h"
#include <new>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

#include "lld/Common/Driver.h"

namespace mlir {
namespace toy {
namespace {
static llvm::codegen::RegisterCodeGenFlags CGF;
class TextGeneratorUtil {
 public:
  void Enter(int n) { indents_.push_back(n); }
  void Leave() { indents_.pop_back(); }
  void EmitIndent() {
    for (auto indent : indents_) {
      str_.insert(str_.end(), indent, ' ');
    }
  }

  void AddLine(const char *line) {
    size_t len = strlen(line);
    AddLine(line, len);
  }

  void AddLine(const std::string &line) { AddLine(line.data(), line.size()); }

  void AddLine(const char *line, size_t len) {
    EmitIndent();
    str_.insert(str_.end(), line, line + len);
    str_.push_back('\n');
  }

  std::string &Str() { return str_; }

 private:
  std::vector<int> indents_;
  std::string str_;
};

struct GenLibraryPass : public PassWrapper<GenLibraryPass, OperationPass<ModuleOp>> {
  explicit GenLibraryPass(std::unique_ptr<::toy::FileObject> *lib_file) : lib_(lib_file) {
    assert(lib_file && "lib file is nullptr!");
  }
  void runOnOperation() final {
    auto module = getOperation();
    llvm::SmallVector<StringRef, 0> symbols;
    auto result = module.walk([&](LLVM::LLVMFuncOp func_op) -> WalkResult {
      return WalkResult(RunFunc(func_op, &symbols));
    });
    if (result.wasInterrupted())
      signalPassFailure();

    ::toy::TempFile temp_file("tmp_XXXXXX");
    ::toy::FileObject obj_file = GenFileObject(module, temp_file.name());
    std::string lds_file_name(temp_file.name());
    lds_file_name += ".lds";
    ::toy::FileObject lds_file(lds_file_name.c_str(), "w");
    if (!GenerateLDSFile(lds_file, symbols)) {
      return signalPassFailure();
    }
    fflush(obj_file);
    fflush(lds_file);
    auto lib = GenLibraryFile(module, temp_file.name(), obj_file, lds_file);
    std::swap(*lib_, lib);
    if (*lib_ == nullptr) {
      return signalPassFailure();
    }
  }

 private:
  static bool GenerateLDSFile(FILE *file, const llvm::SmallVector<StringRef, 0> &symbols) {
    assert(file && "file is nullptr!");
    TextGeneratorUtil text;
    text.AddLine("VERSION {");
    text.Enter(2);
    text.AddLine("{");
    text.Enter(2);
    text.AddLine("global:");
    text.Enter(2);
    for (const auto &symbol : symbols) {
      std::string line(symbol.begin(), symbol.end());
      line.push_back(';');
      text.AddLine(line);
    }
    text.Leave();
    text.AddLine("local:");
    text.Enter(2);
    text.AddLine("*;");
    text.Leave();
    text.Leave();
    text.AddLine("};");
    text.Leave();
    text.AddLine("}");
    const std::string &str = text.Str();
    return fwrite(str.data(), sizeof(char), str.size(), file) == str.size();
  }
  static LogicalResult RunFunc(LLVM::LLVMFuncOp func, llvm::SmallVector<StringRef, 0> *symbols) {
    assert(symbols && "symbol vector is nullptr!");
    auto symbol = func.getName();
    if (func.getBlocks().empty()) {
      return success();
    }
    assert(!symbol.empty() && "Fail to get the func op name!");
    symbols->emplace_back(symbol);
    return success();
  }

  static ::toy::FileObject GenFileObject(::mlir::ModuleOp module, StringRef tmp_name) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    // Register the translation from MLIR to LLVM IR.
    mlir::registerLLVMDialectTranslation(*(module.getContext()));
    std::unique_ptr<llvm::LLVMContext> llvm_ctx(new llvm::LLVMContext);
    std::unique_ptr<llvm::Module> llvm_module =
        std::move(translateModuleToLLVMIR(module, *llvm_ctx));

    llvm::Triple TheTriple;
    std::string CPUStr = llvm::codegen::getCPUStr(), FeaturesStr = llvm::codegen::getFeaturesStr();

    llvm::CodeGenOpt::Level OLvl = llvm::CodeGenOpt::Default;
    llvm::TargetOptions Options;
    auto InitializeOptions = [&](const llvm::Triple &TheTriple) {
      Options = llvm::codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
      //    Options.BinutilsVersion =
      //      llvm::TargetMachine::parseBinutilsVersion(BinutilsVersion);
      //    Options.DisableIntegratedAS = NoIntegratedAssembler;
      //    Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
      //    Options.MCOptions.AsmVerbose = AsmVerbose;
      //    Options.MCOptions.PreserveAsmComments = PreserveComments;
      //    Options.MCOptions.IASSearchPaths = IncludeDirs;
      //    Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
      //    if (DwarfDirectory.getPosition()) {
      //      Options.MCOptions.MCUseDwarfDirectory =
      //        DwarfDirectory ? MCTargetOptions::EnableDwarfDirectory
      //        : MCTargetOptions::DisableDwarfDirectory;
      //    } else {
      //      // -dwarf-directory is not set explicitly. Some assemblers
      //      // (e.g. GNU as or ptxas) do not support `.file directory'
      //      // syntax prior to DWARFv5. Let the target decide the default
      //      // value.
      //      Options.MCOptions.MCUseDwarfDirectory =
      //        MCTargetOptions::DefaultDwarfDirectory;
      //    }
    };
    llvm::Optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();
    const llvm::Target *TheTarget = nullptr;
    std::unique_ptr<llvm::TargetMachine> Target;

    auto SetDataLayout = [&](StringRef DataLayoutTargetTriple) {
      // If we are supposed to override the target triple, do so now.
      std::string IRTargetTriple = DataLayoutTargetTriple.str();
      //      if (!TargetTriple.empty())
      //        IRTargetTriple = Triple::normalize(TargetTriple);
      TheTriple = llvm::Triple(IRTargetTriple);
      if (TheTriple.getTriple().empty())
        TheTriple.setTriple(llvm::sys::getDefaultTargetTriple());

      std::string Error;
      TheTarget = llvm::TargetRegistry::lookupTarget(llvm::codegen::getMArch(), TheTriple, Error);
      assert(TheTarget && "Target is nullptr!");

      // On AIX, setting the relocation model to anything other than PIC is
      // considered a user error.
      assert(!(TheTriple.isOSAIX() && RM.hasValue() && *RM != llvm::Reloc::PIC_) &&
             "invalid relocation model, AIX only supports PIC");

      InitializeOptions(TheTriple);
      Target = std::unique_ptr<llvm::TargetMachine>(
          TheTarget->createTargetMachine(TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM,
                                         llvm::codegen::getExplicitCodeModel(), OLvl));
      assert(Target && "Could not allocate target machine!");

      return Target->createDataLayout();
    };
    llvm_module->setDataLayout(SetDataLayout(""));
    // ----------------------------
    if (llvm::codegen::getFloatABIForCalls() != llvm::FloatABI::Default)
      Options.FloatABIType = llvm::codegen::getFloatABIForCalls();

    llvm::legacy::PassManager PM;

    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(llvm_module->getTargetTriple()));

    // The -disable-simplify-libcalls flag actually disables all builtin optzns.
    //  if (DisableSimplifyLibCalls)
    //    TLII.disableAllFunctions();
    PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
    // Override function attributes based on CPUStr, FeaturesStr, and command line
    // flags.
    llvm::codegen::setFunctionAttributes(CPUStr, FeaturesStr, *llvm_module);

    assert(!(llvm::mc::getExplicitRelaxAll() &&
             llvm::codegen::getFileType() != llvm::CGFT_ObjectFile) &&
           ": warning: ignoring -mc-relax-all because filetype != obj");

    {
      // Manually do the buffering rather than using buffer_ostream,
      // so we can memcmp the contents in CompileTwice mode
      llvm::SmallVector<char, 0> Buffer;
      std::unique_ptr<llvm::raw_svector_ostream> BOS =
          std::make_unique<llvm::raw_svector_ostream>(Buffer);
      llvm::raw_pwrite_stream *OS = BOS.get();
      llvm::LLVMTargetMachine &LLVMTM = static_cast<llvm::LLVMTargetMachine &>(*Target);
      llvm::MachineModuleInfoWrapperPass *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM);

      // Construct a custom pass pipeline that starts after instruction
      // selection.
      // Target->addPassesToEmitFile(PM, *OS, nullptr,
      //                             llvm::codegen::getFileType(), /*this will gen assembly file*/
      //                             false, MMIWP);
      Target->addPassesToEmitFile(PM, *OS, nullptr, llvm::CGFT_ObjectFile, false, MMIWP);

      const_cast<llvm::TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())
          ->Initialize(MMIWP->getMMI().getContext(), *Target);

      llvm::SmallVector<char, 0> CompileTwiceBuffer;
      PM.run(*llvm_module);
      std::string str(Buffer.begin(), Buffer.end());

      std::string obj_file(tmp_name.str());
      obj_file += ".o";
      ::toy::FileObject ret(obj_file.c_str(), "w");
      size_t count = fwrite(Buffer.data(), sizeof(char), Buffer.size(), ret);
      assert(count == Buffer.size() && "Wirte object file failed!");
      return ret;
    }
  }

  static std::unique_ptr<::toy::FileObject> GenLibraryFile(::mlir::ModuleOp module,
                                                           StringRef tmp_name,
                                                           const ::toy::FileObject &obj_file,
                                                           const ::toy::FileObject &lds_script) {
    std::string lib_name(tmp_name.str());
    lib_name = "lib" + lib_name;
    lib_name += ".so";
    std::unique_ptr<::toy::FileObject> ret(new ::toy::FileObject(lib_name.c_str()));
    llvm::SmallVector<char, 0> std_buffer, err_buffer;
    std::unique_ptr<llvm::raw_svector_ostream> std_os =
        std::make_unique<llvm::raw_svector_ostream>(std_buffer);
    std::unique_ptr<llvm::raw_svector_ostream> err_os =
        std::make_unique<llvm::raw_svector_ostream>(err_buffer);
    std::string cur_lib_path(::toy::GetLibraryPath(reinterpret_cast<void *>(&GenLibraryFile)));
    int path_len = ::toy::GetFilePath(cur_lib_path.c_str());
    cur_lib_path.resize(path_len);
    if (cur_lib_path.size()) {
      cur_lib_path = "-L" + cur_lib_path;
    } else {
      cur_lib_path = "./";
    }
    // void *(*operator_new)(unsigned long) = ::operator new;
    // std::string std_path(::toy::GetLibraryPath(reinterpret_cast<void *>(operator_new)));

    // ld.lld -T tmp_xxxxxx.lds tmp_xxxxxx.o --shared -o libtmp_xxxxxx.so --lto-O3 --as-needed
    // -L/path/to/kernel -lkernel
    std::vector<const char *> args = {
        "ld.lld", obj_file.name(), "-o", lib_name.c_str(), "--shared",
        "--Bstatic", "--as-needed",/* "--lto-O3",*/ "-T", lds_script.name(), cur_lib_path.c_str(), "-L.",
        "-lkernel"};
    // Run the driver. If an error occurs, false will be returned.
    int argc = args.size();
    const char **argv = (const char **)(args.data());
    llvm::InitLLVM x(argc, argv);
    bool r = ::lld::elf::link(args, *std_os, *err_os, true, false);
    if (r) {
      return ret;
    } else {
      err_buffer.push_back('\0');
      printf("link error! %s\n", (const char *)(err_buffer.data()));
      ret.reset(nullptr);
      return ret;
    }
  }

 private:
  std::unique_ptr<::toy::FileObject> *lib_;
};
}  // namespace

std::unique_ptr<::mlir::Pass> CreateGenLibraryPass(std::unique_ptr<::toy::FileObject> *lib_file) {
  return std::make_unique<GenLibraryPass>(lib_file);
}
}  // namespace toy
}  // namespace mlir
