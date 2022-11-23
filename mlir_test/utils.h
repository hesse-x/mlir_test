/*--------------------------------------------------------------
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------*/
#ifndef MLIR_TEST_UTILS_H_
#define MLIR_TEST_UTILS_H_
struct _IO_FILE;
namespace toy {
class FileObject;
class TempFile;

class FileObject {
 public:
  explicit FileObject(const char *file_name);
  FileObject(const char *file_name, const char *mode);
  _IO_FILE *Open(const char *mode);
  FileObject(FileObject &&other) : file_name_(other.file_name_), fp_(other.fp_) {
    other.fp_        = nullptr;
    other.file_name_ = nullptr;
  }
  virtual ~FileObject();
  const char *name() const { return file_name_; }
  operator _IO_FILE *() { return fp_; }
  void Keep() {
    is_keep_ = true;
  }

 private:
  FileObject(const FileObject &) = delete;
  void operator=(const FileObject &) = delete;
  friend class TempFile;
  bool is_keep_ = false;
  char *file_name_ = nullptr;
  _IO_FILE *fp_    = nullptr;
};

class TempFile : public FileObject {
 public:
  explicit TempFile(const char *temp);
  virtual ~TempFile() = default;

 private:
  _IO_FILE *Open(const char *mode);
};

const char *GetLibraryPath(void *);

int GetFilePath(const char *path);
}  // namespace toy
#endif  // MLIR_TEST_UTILS_H_
