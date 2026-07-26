//===- ListOps.cpp - MLIR List dialect operations -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/List/IR/List.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::list;

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

LogicalResult FromElementsOp::verify() {
  Type elementType = cast<ListType>(getResult().getType()).getElementType();
  for (auto [index, type] : llvm::enumerate(getElements().getTypes()))
    if (type != elementType)
      return emitOpError("expects operand #")
             << index << " to have the element type of the result list ("
             << elementType << "), but got " << type;
  return success();
}

//===----------------------------------------------------------------------===//
// GetElementsOp
//===----------------------------------------------------------------------===//

LogicalResult GetElementsOp::verify() {
  Type elementType = cast<ListType>(getInput().getType()).getElementType();
  for (auto [index, type] : llvm::enumerate(getResultTypes()))
    if (type != elementType)
      return emitOpError("expects result #")
             << index << " to have the element type of the operand list ("
             << elementType << "), but got " << type;
  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

void MapOp::build(OpBuilder &builder, OperationState &state, Value input,
                  IntegerType resultElementType) {
  state.addOperands(input);
  state.addTypes(ListType::get(resultElementType));
  Block &body = state.addRegion()->emplaceBlock();
  body.addArgument(cast<ListType>(input.getType()).getElementType(),
                   input.getLoc());
}

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input) || parser.parseKeyword("with") ||
      parser.parseLParen())
    return failure();

  OpAsmParser::Argument bodyArg;
  SMLoc bodyArgLoc = parser.getCurrentLocation();
  if (parser.parseArgument(bodyArg, /*allowType=*/true))
    return failure();
  auto inputElementType = dyn_cast_if_present<IntegerType>(bodyArg.type);
  if (!inputElementType)
    return parser.emitError(bodyArgLoc,
                            "expected the body argument to have an integer "
                            "type");

  if (parser.parseRParen() || parser.parseArrow())
    return failure();

  SMLoc resultElementTypeLoc = parser.getCurrentLocation();
  Type parsedResultElementType;
  if (parser.parseType(parsedResultElementType))
    return failure();
  auto resultElementType = dyn_cast<IntegerType>(parsedResultElementType);
  if (!resultElementType)
    return parser.emitError(resultElementTypeLoc,
                            "expected an integer result element type");

  if (parser.parseRegion(*result.addRegion(), bodyArg) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(input, ListType::get(inputElementType),
                            result.operands))
    return failure();

  result.addTypes(ListType::get(resultElementType));
  return success();
}

void MapOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput() << " with (";
  p.printRegionArgument(getBodyBlock().getArgument(0));
  p << ") -> " << cast<ListType>(getResult().getType()).getElementType() << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult MapOp::verify() {
  Block &body = getBodyBlock();
  if (body.getNumArguments() != 1)
    return emitOpError("expects the body to have exactly one argument");

  Type inputElementType = cast<ListType>(getInput().getType()).getElementType();
  if (body.getArgument(0).getType() != inputElementType)
    return emitOpError("expects the body argument type (")
           << body.getArgument(0).getType()
           << ") to match the element type of the operand list ("
           << inputElementType << ")";

  auto yieldOp =
      dyn_cast_if_present<YieldOp>(body.empty() ? nullptr : &body.back());
  if (!yieldOp)
    return emitOpError("expects the body to be terminated by '")
           << YieldOp::getOperationName() << "'";

  Type resultElementType =
      cast<ListType>(getResult().getType()).getElementType();
  if (yieldOp.getYielded().getType() != resultElementType)
    return emitOpError("expects the yielded type (")
           << yieldOp.getYielded().getType()
           << ") to match the element type of the result list ("
           << resultElementType << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/List/IR/ListOps.cpp.inc"
