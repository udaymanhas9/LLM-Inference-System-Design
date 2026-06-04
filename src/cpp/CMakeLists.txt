add_library(inference_lib SHARED inference_lib.cpp)

target_include_directories(inference_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inference_lib PRIVATE llama)
target_compile_features(inference_lib PRIVATE cxx_std_17)

# Place libinference.dylib / .so at the repo root so ctypes finds it easily
set_target_properties(inference_lib PROPERTIES
    OUTPUT_NAME    "inference"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"   # Windows DLL
)
