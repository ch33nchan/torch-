# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/cheencheen/Desktop/torch++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/cheencheen/Desktop/torch++/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_ops.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_ops.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_ops.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_ops.dir/flags.make

tests/CMakeFiles/test_ops.dir/test_ops.cpp.o: tests/CMakeFiles/test_ops.dir/flags.make
tests/CMakeFiles/test_ops.dir/test_ops.cpp.o: /Users/cheencheen/Desktop/torch++/tests/test_ops.cpp
tests/CMakeFiles/test_ops.dir/test_ops.cpp.o: tests/CMakeFiles/test_ops.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_ops.dir/test_ops.cpp.o"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_ops.dir/test_ops.cpp.o -MF CMakeFiles/test_ops.dir/test_ops.cpp.o.d -o CMakeFiles/test_ops.dir/test_ops.cpp.o -c /Users/cheencheen/Desktop/torch++/tests/test_ops.cpp

tests/CMakeFiles/test_ops.dir/test_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_ops.dir/test_ops.cpp.i"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cheencheen/Desktop/torch++/tests/test_ops.cpp > CMakeFiles/test_ops.dir/test_ops.cpp.i

tests/CMakeFiles/test_ops.dir/test_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_ops.dir/test_ops.cpp.s"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cheencheen/Desktop/torch++/tests/test_ops.cpp -o CMakeFiles/test_ops.dir/test_ops.cpp.s

# Object files for target test_ops
test_ops_OBJECTS = \
"CMakeFiles/test_ops.dir/test_ops.cpp.o"

# External object files for target test_ops
test_ops_EXTERNAL_OBJECTS =

tests/test_ops: tests/CMakeFiles/test_ops.dir/test_ops.cpp.o
tests/test_ops: tests/CMakeFiles/test_ops.dir/build.make
tests/test_ops: libtorchplusplus.a
tests/test_ops: tests/CMakeFiles/test_ops.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ops"
	cd /Users/cheencheen/Desktop/torch++/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ops.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_ops.dir/build: tests/test_ops
.PHONY : tests/CMakeFiles/test_ops.dir/build

tests/CMakeFiles/test_ops.dir/clean:
	cd /Users/cheencheen/Desktop/torch++/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_ops.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_ops.dir/clean

tests/CMakeFiles/test_ops.dir/depend:
	cd /Users/cheencheen/Desktop/torch++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cheencheen/Desktop/torch++ /Users/cheencheen/Desktop/torch++/tests /Users/cheencheen/Desktop/torch++/build /Users/cheencheen/Desktop/torch++/build/tests /Users/cheencheen/Desktop/torch++/build/tests/CMakeFiles/test_ops.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test_ops.dir/depend

