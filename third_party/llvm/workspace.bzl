load("//third_party:repo.bzl", "tf_http_archive", "gen_self_ip_url")

def repo(name):
    """Imports LLVM."""
    LLVM_VERSION = "14.0.6"
    LLVM_SHA256 = "98f15f842700bdb7220a166c8d2739a03a72e775b67031205078f39dd756a055"
#    LLVM_SHA256 = "2deb62e5dd1a323f94ace5c8d15e8d05c6a2f65dfdc648aaf4830557bd0e0330"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-llvmorg-{version}".format(version = LLVM_VERSION),
        urls = [
#             gen_self_ip_url("llvm-project-llvmorg-14.0.6.tar.gz"),
            "https://github.com/llvm/llvm-project/archive/llvmorg-{version}.tar.gz".format(version = LLVM_VERSION),
#            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
#            "//third_party/llvm:infer_type.patch",  # TODO(b/231285230): remove once resolved
#            "//third_party/llvm:build.patch",
#            "//third_party/llvm:toolchains.patch",
#            "//third_party/llvm:temporary.patch",  # Cherry-picks and temporary reverts. Do not remove even if temporary.patch is empty.
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
