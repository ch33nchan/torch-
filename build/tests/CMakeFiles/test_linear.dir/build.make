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
include tests/CMakeFiles/test_linear.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_linear.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_linear.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_linear.dir/flags.make

tests/CMakeFiles/test_linear.dir/test_linear.cpp.o: tests/CMakeFiles/test_linear.dir/flags.make
tests/CMakeFiles/test_linear.dir/test_linear.cpp.o: /Users/cheencheen/Desktop/torch++/tests/test_linear.cpp
tests/CMakeFiles/test_linear.dir/test_linear.cpp.o: tests/CMakeFiles/test_linear.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_linear.dir/test_linear.cpp.o"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_linear.dir/test_linear.cpp.o -MF CMakeFiles/test_linear.dir/test_linear.cpp.o.d -o CMakeFiles/test_linear.dir/test_linear.cpp.o -c /Users/cheencheen/Desktop/torch++/tests/test_linear.cpp

tests/CMakeFiles/test_linear.dir/test_linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_linear.dir/test_linear.cpp.i"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cheencheen/Desktop/torch++/tests/test_linear.cpp > CMakeFiles/test_linear.dir/test_linear.cpp.i

tests/CMakeFiles/test_linear.dir/test_linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_linear.dir/test_linear.cpp.s"
	cd /Users/cheencheen/Desktop/torch++/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cheencheen/Desktop/torch++/tests/test_linear.cpp -o CMakeFiles/test_linear.dir/test_linear.cpp.s

# Object files for target test_linear
test_linear_OBJECTS = \
"CMakeFiles/test_linear.dir/test_linear.cpp.o"

# External object files for target test_linear
test_linear_EXTERNAL_OBJECTS =

tests/test_linear: tests/CMakeFiles/test_linear.dir/test_linear.cpp.o
tests/test_linear: tests/CMakeFiles/test_linear.dir/build.make
tests/test_linear: libtorchplusplus.a
tests/test_linear: lib/libgtest_main.a
tests/test_linear: lib/libgtest.a
tests/test_linear: tests/CMakeFiles/test_linear.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_linear"
	cd /Users/cheencheen/Desktop/torch++/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_linear.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/cheencheen/Desktop/torch++/build/tests && /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -D TEST_TARGET=test_linear -D TEST_EXECUTABLE=/Users/cheencheen/Desktop/torch++/build/tests/test_linear -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/Users/cheencheen/Desktop/torch++/build/tests -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=test_linear_TESTS -D CTEST_FILE=/Users/cheencheen/Desktop/torch++/build/tests/test_linear[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_XML_OUTPUT_DIR= -P /opt/homebrew/Cellar/cmake/3.30.5/share/cmake/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
tests/CMakeFiles/test_linear.dir/build: tests/test_linear
.PHONY : tests/CMakeFiles/test_linear.dir/build

tests/CMakeFiles/test_linear.dir/clean:
	cd /Users/cheencheen/Desktop/torch++/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_linear.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_linear.dir/clean

tests/CMakeFiles/test_linear.dir/depend:
	cd /Users/cheencheen/Desktop/torch++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cheencheen/Desktop/torch++ /Users/cheencheen/Desktop/torch++/tests /Users/cheencheen/Desktop/torch++/build /Users/cheencheen/Desktop/torch++/build/tests /Users/cheencheen/Desktop/torch++/build/tests/CMakeFiles/test_linear.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test_linear.dir/depend

