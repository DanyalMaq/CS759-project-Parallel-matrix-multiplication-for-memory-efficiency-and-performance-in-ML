# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /opt/conda/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /opt/conda/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /fly/cnpy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /fly

# Include any dependencies generated for this target.
include CMakeFiles/cnpy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cnpy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cnpy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnpy.dir/flags.make

CMakeFiles/cnpy.dir/cnpy.cpp.o: CMakeFiles/cnpy.dir/flags.make
CMakeFiles/cnpy.dir/cnpy.cpp.o: /fly/cnpy/cnpy.cpp
CMakeFiles/cnpy.dir/cnpy.cpp.o: CMakeFiles/cnpy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/fly/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnpy.dir/cnpy.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnpy.dir/cnpy.cpp.o -MF CMakeFiles/cnpy.dir/cnpy.cpp.o.d -o CMakeFiles/cnpy.dir/cnpy.cpp.o -c /fly/cnpy/cnpy.cpp

CMakeFiles/cnpy.dir/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cnpy.dir/cnpy.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /fly/cnpy/cnpy.cpp > CMakeFiles/cnpy.dir/cnpy.cpp.i

CMakeFiles/cnpy.dir/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cnpy.dir/cnpy.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /fly/cnpy/cnpy.cpp -o CMakeFiles/cnpy.dir/cnpy.cpp.s

# Object files for target cnpy
cnpy_OBJECTS = \
"CMakeFiles/cnpy.dir/cnpy.cpp.o"

# External object files for target cnpy
cnpy_EXTERNAL_OBJECTS =

libcnpy.so: CMakeFiles/cnpy.dir/cnpy.cpp.o
libcnpy.so: CMakeFiles/cnpy.dir/build.make
libcnpy.so: /usr/lib/x86_64-linux-gnu/libz.so
libcnpy.so: CMakeFiles/cnpy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/fly/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcnpy.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnpy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnpy.dir/build: libcnpy.so
.PHONY : CMakeFiles/cnpy.dir/build

CMakeFiles/cnpy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnpy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnpy.dir/clean

CMakeFiles/cnpy.dir/depend:
	cd /fly && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /fly/cnpy /fly/cnpy /fly /fly /fly/CMakeFiles/cnpy.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cnpy.dir/depend
