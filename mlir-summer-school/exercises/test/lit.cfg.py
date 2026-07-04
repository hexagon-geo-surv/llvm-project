# -*- Python -*-
# Lit configuration for the school exercise tests.

import os

import lit.formats

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "SCHOOL"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.school_obj_root, "test")

config.excludes = ["CMakeLists.txt"]

# Provides the default substitutions (%s, %t, ...) and FileCheck/not/count.
llvm_config.use_default_substitutions()

# Make `school-opt` (from this project) and `mlir-opt` (from the LLVM build)
# usable in RUN lines.
school_tools_dir = os.path.join(config.school_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(
    ["school-opt", "mlir-opt"], [school_tools_dir, config.llvm_tools_dir]
)
