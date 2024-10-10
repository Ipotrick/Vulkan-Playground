
include(CMakePackageConfigHelpers)
file(WRITE ${CMAKE_BINARY_DIR}/config.cmake.in [=[
@PACKAGE_INIT@
include(${CMAKE_CURRENT_LIST_DIR}/daxa-targets.cmake)
check_required_components(daxa)

get_target_property(DAXA_PREV_DEFINITIONS daxa::daxa INTERFACE_COMPILE_DEFINITIONS)
set_target_properties(daxa::daxa PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "${DAXA_PREV_DEFINITIONS};DAXA_SHADER_INCLUDE_DIR=\"${PACKAGE_PREFIX_DIR}/include\""
)
]=])

# Re-exporting the find_package is necessary for Linux package management for some reason...
file(APPEND ${CMAKE_BINARY_DIR}/config.cmake.in [=[
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
]=])

if(DAXA_ENABLE_UTILS_FSR2)
    file(APPEND ${CMAKE_BINARY_DIR}/config.cmake.in [=[
find_package(fsr2 CONFIG REQUIRED)
]=])
endif()
if(DAXA_ENABLE_UTILS_IMGUI)
    file(APPEND ${CMAKE_BINARY_DIR}/config.cmake.in [=[
find_package(imgui CONFIG REQUIRED)
]=])
endif()
if(DAXA_ENABLE_UTILS_MEM)
# No package management work to do
endif()
if(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG)
    file(APPEND ${CMAKE_BINARY_DIR}/config.cmake.in [=[
find_package(glslang CONFIG REQUIRED)
find_package(Threads REQUIRED)
]=])
endif()
if(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_SPIRV_VALIDATION)
    file(APPEND ${CMAKE_BINARY_DIR}/config.cmake.in [=[
find_package(SPIRV-Tools CONFIG REQUIRED)
]=])
endif()
if(DAXA_ENABLE_UTILS_TASK_GRAPH)
# No package management work to do
endif()

configure_package_config_file(${CMAKE_BINARY_DIR}/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/daxa-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/daxa
    NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/daxa-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/daxa-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/daxa-config-version.cmake
    DESTINATION
    ${CMAKE_INSTALL_DATADIR}/daxa)
install(TARGETS daxa EXPORT daxa-targets)
if(BUILD_SHARED_LIBS AND WIN32)
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
endif()
install(EXPORT daxa-targets DESTINATION ${CMAKE_INSTALL_DATADIR}/daxa NAMESPACE daxa::)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ TYPE INCLUDE)
