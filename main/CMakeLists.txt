# Authors: Santiago Morante; Juan G. Victores

# reduce warning level with cmake 2.6
cmake_minimum_required(VERSION 2.6)
#cmake policies
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(travis)

set(KEYWORD "TravisLib")

# Find OpenCV.
find_package(OpenCV REQUIRED)

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
  add_library(${KEYWORD} ${folder_source} ${folder_header})
  target_link_libraries(${KEYWORD} ${OpenCV_LIBS})
else (folder_source)
  message(FATAL_ERROR "No source code files found. Please add something")
endif (folder_source)

set(TRAVIS_INCLUDE_DIRS CACHE INTERNAL "appended header dirs" FORCE)
set(TRAVIS_LINK_DIRS CACHE INTERNAL "appended link dirs" FORCE)
set(TRAVIS_LIBRARIES CACHE INTERNAL "appended libraries" FORCE)

# Exporting dependencies for TRAVISConfig.cmake quite manually for now... 
set(TRAVIS_INCLUDE_DIRS ${TRAVIS_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "appended header dirs")
set(TRAVIS_LIBRARIES ${TRAVIS_LIBRARIES} ${KEYWORD} CACHE INTERNAL "appended libraries") # Not for now as it is a header

IF(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
ENDIF(NOT LIBRARY_OUTPUT_PATH)
# export our variables to a TRAVISConfig.cmake creation
set(TRAVIS_LINK_DIRS ${TRAVIS_LINK_DIRS} ${LIBRARY_OUTPUT_PATH})
configure_file(${CMAKE_CURRENT_LIST_DIR}/conf/template/TRAVISConfig.cmake.in
               ${CMAKE_BINARY_DIR}/TRAVISConfig.cmake @ONLY)

