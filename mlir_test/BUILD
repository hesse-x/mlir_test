load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
td_library(
    name = "toy_td_files",
    srcs = [
        "ops.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "toy_inc_gen",
    tbl_outs = [
        (["-gen-op-decls"], "ops.h.inc"),
        (["-gen-op-defs"], "ops.cc.inc"),
        (["-gen-dialect-decls"], "dialect.h.inc"),
        (["-gen-dialect-defs"], "dialect.cc.inc"),
        (["-gen-typedef-decls"], "type.h.inc"),
        (["-gen-typedef-defs"], "type.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = ["toy_td_files"],
)

cc_library(
    name = "toy_dialect",
    hdrs = [
        "dialect.h",
        "type.h",
    ],
    srcs = [
        "type.cc",
        "ops.cc",
        "dialect.cc",
    ],
    deps = [
        ":toy_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
    ],
)

cc_library(
    name = "lower2llvm",
    hdrs = [
        "lower2llvm.h",
    ],
    srcs = [
        "lower2llvm.cc",
    ],
    deps = [
        ":toy_dialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardToLLVM",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticToLLVM",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "gen_lib_pass",
    hdrs = [
        "gen_lib_pass.h",
    ],
    srcs = [
        "gen_lib_pass.cc",
    ],
    deps = [
        ":utils",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86Info",
        "@llvm-project//llvm:X86UtilsAndDesc",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//lld:ELF",
    ],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    srcs = [
        "utils.cc",
    ],
)

cc_test(
    name = "gen_lib_test",
    srcs = [
        "gen_lib_test.cc",
    ],
    deps = [
        ":lower2llvm",
        ":gen_lib_pass",
	":utils",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardToLLVM",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticToLLVM",
        "@llvm-project//mlir:ToLLVMIRTranslation",
	"@com_google_googletest//:gtest",
	"@com_google_googletest//:gtest_main",
    ],
)
