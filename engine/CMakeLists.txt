cmake_minimum_required(VERSION 3.20)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(ANDROID)
  # The Android NDK ships Vulkan headers and the system Vulkan loader import library.
  find_path(Vulkan_INCLUDE_DIR vulkan/vulkan.h
    HINTS "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include")
  find_library(Vulkan_LIBRARY vulkan)
  if(NOT Vulkan_INCLUDE_DIR OR NOT Vulkan_LIBRARY)
    message(FATAL_ERROR "Android Vulkan headers/library not found in the selected NDK sysroot")
  endif()
  add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
  set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${Vulkan_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}")
else()
  find_package(Vulkan REQUIRED)
endif()

# Set -DGLSLANG_VALIDATOR=<path> to override; otherwise the compiler is searched on PATH.
if(NOT GLSLANG_VALIDATOR)
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
endif()

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPV_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})

# Each shader is written as a stage body. The build prepends the version line and
# common.glsl (cat, no preprocessor), then compiles the result to SPIR-V.
# Usage: engine_shader(<list-var> <source.glsl> <stage>)
function(engine_shader list_var src stage)
  get_filename_component(base ${src} NAME)
  set(gen ${SPV_DIR}/${base}.gen.glsl)
  set(spv ${SPV_DIR}/${base}.spv)
  # Generate the combined shader at configure time. CMake COMMAND does not run
  # through a shell, so using a literal `>` redirection here would create no file.
  file(READ ${SHADER_DIR}/version.glsl version_text)
  file(READ ${SHADER_DIR}/common.glsl common_text)
  file(READ ${src} shader_text)
  file(WRITE ${gen} "${version_text}\n${common_text}\n${shader_text}")
  # Force CMake to re-run the custom command when source files change.
  # We do this by adding the source files as dependencies and also adding a
  # configure-depends property that CMake checks.
  add_custom_command(
    OUTPUT ${spv}
    COMMAND ${GLSLANG_VALIDATOR} -V -S ${stage} --target-env vulkan1.3 -o ${spv} ${gen}
    DEPENDS ${gen} ${src} ${SHADER_DIR}/common.glsl ${SHADER_DIR}/version.glsl
    COMMENT "SPIR-V ${base}")
  set(${list_var} ${${list_var}} ${spv} PARENT_SCOPE)
endfunction()

# Ensure CMake re-configures when shader sources change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  engine/shaders/version.glsl
  engine/shaders/common.glsl)

set(SHADER_SPV)
engine_shader(SHADER_SPV ${SHADER_DIR}/tri.vert vert)
engine_shader(SHADER_SPV ${SHADER_DIR}/tri.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/fullscreen.vert vert)
engine_shader(SHADER_SPV ${SHADER_DIR}/transmittance.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/ms.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/skyview.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/sky_prefilter.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/aerial.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/depth_only.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/gtao.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/ao_blur.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/taa.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/copy.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/bloom_bright.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/bloom_blur.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/mesh.vert vert)
engine_shader(SHADER_SPV ${SHADER_DIR}/mesh.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/mesh_shadow.vert vert)
engine_shader(SHADER_SPV ${SHADER_DIR}/mesh_shadow.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/clouds.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/cloud_shadow.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/shafts.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/sky.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/ocean.vert vert)
engine_shader(SHADER_SPV ${SHADER_DIR}/ocean.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/tonemap.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/dof_prepare.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/dof_blur.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/dof_composite.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/exposure_downsample.frag frag)
engine_shader(SHADER_SPV ${SHADER_DIR}/exposure_reduce.frag frag)
add_custom_target(shaders ALL DEPENDS ${SHADER_SPV})

add_library(engine_rhi STATIC
  src/rhi/vulkan/vk_device.cpp
  src/rhi/vulkan/vk_command.cpp
  src/rg/render_graph.cpp
  src/renderer/ocean_renderer.cpp
  src/scene/ocean_mesh.cpp
  src/scene/landscape.cpp
  src/scene/gi_bake.cpp
  src/assets/texture.cpp)
target_include_directories(engine_rhi PUBLIC src)
target_link_libraries(engine_rhi PUBLIC Vulkan::Vulkan)
target_compile_options(engine_rhi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(engine_triangle src/main_triangle.cpp)
add_dependencies(engine_triangle shaders)
target_link_libraries(engine_triangle PRIVATE engine_rhi)
target_compile_definitions(engine_triangle PRIVATE ENGINE_SHADER_DIR="${SPV_DIR}")
target_compile_options(engine_triangle PRIVATE -Wall -Wextra -Wpedantic)

add_executable(engine_render src/main_render.cpp)
add_dependencies(engine_render shaders)
target_link_libraries(engine_render PRIVATE engine_rhi)
target_compile_definitions(engine_render PRIVATE ENGINE_SHADER_DIR="${SPV_DIR}"
                           ENGINE_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
target_compile_options(engine_render PRIVATE -Wall -Wextra -Wpedantic)

# Soft-body car crash model: header-only physics, headless test (no Vulkan dependency).
add_executable(engine_softcar_test src/physics/softcar_test.cpp)
target_include_directories(engine_softcar_test PRIVATE src)
target_compile_options(engine_softcar_test PRIVATE -Wall -Wextra)

# Road-vehicle dynamics: header-only, headless driving test.
add_executable(engine_vehicle_test src/physics/vehicle_test.cpp)
target_include_directories(engine_vehicle_test PRIVATE src)
target_compile_options(engine_vehicle_test PRIVATE -Wall -Wextra)

# Compute path: a storage-buffer fill dispatched through the RHI, read back and checked.
set(COMPUTE_FILL_SPV ${SPV_DIR}/compute_fill.comp.spv)
add_custom_command(
  OUTPUT ${COMPUTE_FILL_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V -S comp --target-env vulkan1.3 -o ${COMPUTE_FILL_SPV} ${SHADER_DIR}/compute_fill.comp
  DEPENDS ${SHADER_DIR}/compute_fill.comp
  COMMENT "SPIR-V compute_fill")
add_custom_target(compute_fill_spv DEPENDS ${COMPUTE_FILL_SPV})
add_executable(engine_compute_test src/rhi/compute_test.cpp)
target_link_libraries(engine_compute_test PRIVATE engine_rhi)
add_dependencies(engine_compute_test compute_fill_spv)

# FFT building block: 256-point Stockham FFT as compute passes, checked against a naive DFT.
set(FFT_STAGE_SPV ${SPV_DIR}/fft_stage.comp.spv)
add_custom_command(
  OUTPUT ${FFT_STAGE_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V -S comp --target-env vulkan1.3 -o ${FFT_STAGE_SPV} ${SHADER_DIR}/fft_stage.comp
  DEPENDS ${SHADER_DIR}/fft_stage.comp
  COMMENT "SPIR-V fft_stage")
add_custom_target(fft_stage_spv DEPENDS ${FFT_STAGE_SPV})
add_executable(engine_fft_test src/rhi/fft_test.cpp)
target_link_libraries(engine_fft_test PRIVATE engine_rhi)
add_dependencies(engine_fft_test fft_stage_spv)

# FFT ocean: Tessendorf spectrum evolved on the GPU, 2-D FFT as compute passes, checked against a direct sum.
set(OCEAN_EVOLVE_SPV ${SPV_DIR}/ocean_evolve.comp.spv)
set(FFT2D_SPV ${SPV_DIR}/fft2d_stage.comp.spv)
add_custom_command(
  OUTPUT ${OCEAN_EVOLVE_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V -S comp --target-env vulkan1.3 -o ${OCEAN_EVOLVE_SPV} ${SHADER_DIR}/ocean_evolve.comp
  DEPENDS ${SHADER_DIR}/ocean_evolve.comp
  COMMENT "SPIR-V ocean_evolve")
add_custom_command(
  OUTPUT ${FFT2D_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V -S comp --target-env vulkan1.3 -o ${FFT2D_SPV} ${SHADER_DIR}/fft2d_stage.comp
  DEPENDS ${SHADER_DIR}/fft2d_stage.comp
  COMMENT "SPIR-V fft2d_stage")
add_custom_target(ocean_spv DEPENDS ${OCEAN_EVOLVE_SPV} ${FFT2D_SPV})
add_executable(engine_ocean_test src/ocean/ocean_test.cpp)
target_include_directories(engine_ocean_test PRIVATE src)
target_link_libraries(engine_ocean_test PRIVATE engine_rhi)
add_dependencies(engine_ocean_test ocean_spv)

# The renderer's ocean passes need the compute SPIR-V as well.
add_dependencies(engine_render ocean_spv)
