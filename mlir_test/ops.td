#ifndef MLIR_TEST_OPS_TD_
#define MLIR_TEST_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/*
 * --------------------------------------------------------------------
 * dialect toy
 * --------------------------------------------------------------------
 */
def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<ToyDialect, mnemonic,
       !listconcat([MemoryEffects<[MemRead, MemWrite]>], traits)>;
/*
* --------------------------------------------------------------------
* toy type
* --------------------------------------------------------------------
*/

def ToyHandleType :
    DialectType<ToyDialect, CPred<"$_self.isa<::mlir::toy::ToyHandleType>()">,
                "toy handle type", "::mlir::toy::ToyHandleType">;

/*
* --------------------------------------------------------------------
* toy ops
* --------------------------------------------------------------------
*/
def CreateToyHandleOp : Toy_Op<"create_toy_handle", [NoSideEffect]> {
  let summary = "create toy handle operation";
  let description = [{"create toy handle"}];

  // ToyHandle *CreateToyHandle()
  let results = (outs ToyHandleType:$handle);

  let assemblyFormat = [{
    `(` attr-dict `)` `->` type($handle)
  }];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::mlir::success(); }];
}

def DestroyToyHandleOp : Toy_Op<"destroy_toy_handle", [NoSideEffect]> {
  let summary = "destroy toy handle operation";
  let description = [{
    The "destroy toy handle" operation calculate input tensor(real type)
    fast Fourier transform(result complex type).
  }];

  // DestroyToyHandle(ToyHandle *exe)
  let arguments = (
      ins ToyHandleType:$handle
  );

  let assemblyFormat = [{
    `(` operands `)` `(` type(operands) `)` attr-dict
  }];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::mlir::success(); }];
}

def ToyFuncAOp : Toy_Op<"func_a", [NoSideEffect]> {
  let summary = "toy operation";
  let description = [{"toy FuncA"}];

  let arguments = (ins
      ToyHandleType:$handle,
      Arg<MemRefRankOf<[F64, Complex<F64>], [1]>, "", [MemRead]>:$input,
      Arg<MemRefRankOf<[F64, Complex<F64>], [1]>, "", [MemWrite]>:$output
  );

  let assemblyFormat = [{
    `(` operands `)` `(` type(operands) `)` attr-dict
  }];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::mlir::success(); }];
}
#endif
