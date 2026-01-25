cmake_minimum_required(VERSION 3.5.1)
project(mcl)

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    videoio
)

include_directories(${OpenCV_INCLUDE_DIRS})
# --- LibTorch ---
find_package(Torch REQUIRED)

# --- Sources ---
file(GLOB_RECURSE project_mcl CONFIGURE_DEPENDS src/*.cpp)

# --- Target ---
add_executable(main ${project_mcl})

message(STATUS "OpenCV libs = ${OpenCV_LIBS}")

# --- Linking ---
target_link_libraries(main
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
    ${TORCH_LIBRARIES}
)


set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(main PRIVATE -Wall -Wextra)
