# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liusida/simulator/Voxelyze.Ti

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liusida/simulator/Voxelyze.Ti/build

# Include any dependencies generated for this target.
include CMakeFiles/Voxelyze.Ti.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Voxelyze.Ti.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Voxelyze.Ti.dir/flags.make

CMakeFiles/Voxelyze.Ti.dir/main.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/main.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/main.cpp

CMakeFiles/Voxelyze.Ti.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/main.cpp > CMakeFiles/Voxelyze.Ti.dir/main.cpp.i

CMakeFiles/Voxelyze.Ti.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/main.cpp -o CMakeFiles/Voxelyze.Ti.dir/main.cpp.s

CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/main.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o: ../src/VX_Collision.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_Collision.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_Collision.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_Collision.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o: ../src/VX_External.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_External.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_External.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_External.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o: ../src/VX_LinearSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_LinearSolver.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_LinearSolver.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_LinearSolver.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o: ../src/VX_Link.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_Link.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_Link.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_Link.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o: ../src/VX_Material.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_Material.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_Material.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_Material.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o: ../src/VX_MaterialLink.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialLink.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialLink.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialLink.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o: ../src/VX_MaterialVoxel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialVoxel.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialVoxel.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_MaterialVoxel.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o: ../src/VX_MeshRender.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_MeshRender.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_MeshRender.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_MeshRender.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o: ../src/VX_Voxel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/VX_Voxel.cpp

CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/VX_Voxel.cpp > CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/VX_Voxel.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o


CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o: CMakeFiles/Voxelyze.Ti.dir/flags.make
CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o: ../src/Voxelyze.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o -c /home/liusida/simulator/Voxelyze.Ti/src/Voxelyze.cpp

CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liusida/simulator/Voxelyze.Ti/src/Voxelyze.cpp > CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.i

CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liusida/simulator/Voxelyze.Ti/src/Voxelyze.cpp -o CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.s

CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.requires:

.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.requires

CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.provides: CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.requires
	$(MAKE) -f CMakeFiles/Voxelyze.Ti.dir/build.make CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.provides.build
.PHONY : CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.provides

CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.provides.build: CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o


# Object files for target Voxelyze.Ti
Voxelyze_Ti_OBJECTS = \
"CMakeFiles/Voxelyze.Ti.dir/main.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o" \
"CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o"

# External object files for target Voxelyze.Ti
Voxelyze_Ti_EXTERNAL_OBJECTS =

Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/main.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/build.make
Voxelyze.Ti: CMakeFiles/Voxelyze.Ti.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable Voxelyze.Ti"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Voxelyze.Ti.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Voxelyze.Ti.dir/build: Voxelyze.Ti

.PHONY : CMakeFiles/Voxelyze.Ti.dir/build

CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/main.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_Collision.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_External.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_LinearSolver.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_Link.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_Material.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialLink.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_MaterialVoxel.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_MeshRender.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/VX_Voxel.cpp.o.requires
CMakeFiles/Voxelyze.Ti.dir/requires: CMakeFiles/Voxelyze.Ti.dir/src/Voxelyze.cpp.o.requires

.PHONY : CMakeFiles/Voxelyze.Ti.dir/requires

CMakeFiles/Voxelyze.Ti.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Voxelyze.Ti.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Voxelyze.Ti.dir/clean

CMakeFiles/Voxelyze.Ti.dir/depend:
	cd /home/liusida/simulator/Voxelyze.Ti/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liusida/simulator/Voxelyze.Ti /home/liusida/simulator/Voxelyze.Ti /home/liusida/simulator/Voxelyze.Ti/build /home/liusida/simulator/Voxelyze.Ti/build /home/liusida/simulator/Voxelyze.Ti/build/CMakeFiles/Voxelyze.Ti.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Voxelyze.Ti.dir/depend

