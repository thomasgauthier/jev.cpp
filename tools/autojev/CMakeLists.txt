add_library(autojev-common STATIC autojev-common.cpp)

if (BUILD_SHARED_LIBS)
    set_target_properties(autojev-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(autojev-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autojev-common PUBLIC llama-common mtmd)
target_compile_features(autojev-common PUBLIC cxx_std_17)

add_executable(llama-autojev autojev.cpp)
target_link_libraries(llama-autojev PRIVATE autojev-common llama-common mtmd Threads::Threads)
target_compile_features(llama-autojev PRIVATE cxx_std_17)
