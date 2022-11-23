/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#include "mlir_test/utils.h"
#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace toy {
FileObject::FileObject(const char *file_name) {
  int len = std::strlen(file_name);
  if (len > 0) {
    file_name_ = new char[len + 1];
    std::memcpy(file_name_, file_name, len + 1);
  }
}

FileObject::FileObject(const char *file_name, const char *mode) {
  int len = std::strlen(file_name);
  if (len > 0) {
    file_name_ = new char[len + 1];
    std::memcpy(file_name_, file_name, len + 1);
  }
  Open(mode);
}

FileObject::~FileObject() {
  if (file_name_) {
    if (fp_) {
      fclose(fp_);
    }
    if (!is_keep_) {
      std::remove(file_name_);
    }
    delete[] file_name_;
    file_name_ = nullptr;
    fp_ = nullptr;
  }
}

FILE *FileObject::Open(const char *mode) {
  if (fp_ == nullptr) {
    if (file_name_) {
      fp_ = fopen(file_name_, mode);
    } else {
      fp_ = nullptr;
    }
  }
  return fp_;
}

TempFile::TempFile(const char *temp) : FileObject(temp) {
  if (file_name_) {
    int fd = mkstemp(file_name_);
    if (fd != -1)
      fp_ = fdopen(fd, "wr");
  }
}

const char *GetLibraryPath(void *addr) {
  Dl_info info;
  dladdr(addr, &info);
  return info.dli_fname;
}

int GetFilePath(const char *path) {
  int len = strlen(path);
  const char *it = path, *pre, *end = path + len;
  do {
    pre = it;
    it = std::find(pre + 1, end, '/');
  } while (it < end);
  return pre - path;
}
}  // namespace toy
