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
include examples/CMakeFiles/basic_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/basic_example.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/basic_example.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/basic_example.dir/flags.make

examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o: examples/CMakeFiles/basic_example.dir/flags.make
examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o: /Users/cheencheen/Desktop/torch++/examples/basic_examples.cpp
examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o: examples/CMakeFiles/basic_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o"
	cd /Users/cheencheen/Desktop/torch++/build/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o -MF CMakeFiles/basic_example.dir/basic_examples.cpp.o.d -o CMakeFiles/basic_example.dir/basic_examples.cpp.o -c /Users/cheencheen/Desktop/torch++/examples/basic_examples.cpp

examples/CMakeFiles/basic_example.dir/basic_examples.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/basic_example.dir/basic_examples.cpp.i"
	cd /Users/cheencheen/Desktop/torch++/build/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cheencheen/Desktop/torch++/examples/basic_examples.cpp > CMakeFiles/basic_example.dir/basic_examples.cpp.i

examples/CMakeFiles/basic_example.dir/basic_examples.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/basic_example.dir/basic_examples.cpp.s"
	cd /Users/cheencheen/Desktop/torch++/build/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cheencheen/Desktop/torch++/examples/basic_examples.cpp -o CMakeFiles/basic_example.dir/basic_examples.cpp.s

# Object files for target basic_example
basic_example_OBJECTS = \
"CMakeFiles/basic_example.dir/basic_examples.cpp.o"

# External object files for target basic_example
basic_example_EXTERNAL_OBJECTS =

examples/basic_example: examples/CMakeFiles/basic_example.dir/basic_examples.cpp.o
examples/basic_example: examples/CMakeFiles/basic_example.dir/build.make
examples/basic_example: libtorchplusplus.a
examples/basic_example: examples/CMakeFiles/basic_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/cheencheen/Desktop/torch++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable basic_example"
	cd /Users/cheencheen/Desktop/torch++/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/basic_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/basic_example.dir/build: examples/basic_example
.PHONY : examples/CMakeFiles/basic_example.dir/build

examples/CMakeFiles/basic_example.dir/clean:
	cd /Users/cheencheen/Desktop/torch++/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/basic_example.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/basic_example.dir/clean

examples/CMakeFiles/basic_example.dir/depend:
	cd /Users/cheencheen/Desktop/torch++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cheencheen/Desktop/torch++ /Users/cheencheen/Desktop/torch++/examples /Users/cheencheen/Desktop/torch++/build /Users/cheencheen/Desktop/torch++/build/examples /Users/cheencheen/Desktop/torch++/build/examples/CMakeFiles/basic_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : examples/CMakeFiles/basic_example.dir/depend

