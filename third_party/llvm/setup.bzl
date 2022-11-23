load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

_LLVM_TARGETS = [
#    "AArch64",
#    "AMDGPU",
#    "ARM",
#    "NVPTX",
#    "PowerPC",
#    "RISCV",
#    "SystemZ",
    "X86",
]

def llvm_setup(name):
    # Disable terminfo and zlib that are bundled with LLVM.
    llvm_disable_optional_support_deps()

    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        repo_mapping = {"@python_runtime": "@local_config_python"},
        targets = _LLVM_TARGETS,
    )
