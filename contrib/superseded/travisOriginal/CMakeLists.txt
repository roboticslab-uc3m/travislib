
cmake_minimum_required(VERSION 2.6)

set(KEYWORD "example")

# Start a project.
project(${KEYWORD})

# Find OpenCV.
find_package(OpenCV REQUIRED)
# Find YARP.
#find_package(YARP REQUIRED)

# Search for source code.
file(GLOB folder_source *.cpp *.cc *.c)
file(GLOB folder_header *.h)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

# Automatically add include directories if needed.
foreach(header_file ${folder_header})
  get_filename_component(p ${header_file} PATH)
  include_directories(${p})
endforeach(header_file ${folder_header})

# Set up our main executable.
if (folder_source)
  add_executable(${KEYWORD} ${folder_source} ${folder_header})
#  target_link_libraries(${KEYWORD} ${OpenCV_LIBS} ${YARP_LIBRARIES})
  target_link_libraries(${KEYWORD} ${OpenCV_LIBS})
else (folder_source)
  message(FATAL_ERROR "No source code files found. Please add something")
endif (folder_source)

