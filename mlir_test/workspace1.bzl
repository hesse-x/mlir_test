load("//third_party:repo.bzl", "tf_http_archive", "gen_self_ip_url")
load("//third_party/llvm:workspace.bzl", llvm = "repo")
def workspace1():
  tf_http_archive(
      name = "com_google_googletest",
      sha256 = "bc1cc26d1120f5a7e9eb450751c0b24160734e46a02823a573f3c6b6c0a574a7",
      strip_prefix = "googletest-e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b",
      urls = ["https://github.com/google/googletest/archive/e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b.zip"],
  )
  tf_http_archive(
      name = "bazel_skylib",
      sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
      urls = [
          gen_self_ip_url("bazel-skylib-1.0.2.tar.gz"),
          "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
      ],
  )

  llvm("llvm-raw")
