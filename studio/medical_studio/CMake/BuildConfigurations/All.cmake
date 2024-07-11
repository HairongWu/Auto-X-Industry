set(MITK_CONFIG_PACKAGES )

set(_apple_package_excludes)

set(_package_excludes
  ${_apple_package_excludes}
  OpenMP
  SYSTEM_Boost
  Boost_LIBRARIES
  SYSTEM_PYTHON
  SUPERBUILD
)

get_cmake_property(_cache_vars CACHE_VARIABLES)
foreach(_cache_var ${_cache_vars})
  string(REGEX REPLACE "MITK_USE_(.+)" "\\1" _package "${_cache_var}")
  if(_package AND NOT _package STREQUAL _cache_var)
    list(FIND _package_excludes ${_package} _index)
    if(_index EQUAL -1)
      list(APPEND MITK_CONFIG_PACKAGES ${_package})
    endif()
  endif()
endforeach()

set(MITK_BUILD_ALL_APPS ON CACHE BOOL "Build all MITK applications" FORCE)
set(MITK_BUILD_ALL_PLUGINS ON CACHE BOOL "Build all MITK plugins" FORCE)
set(MITK_BUILD_EXAMPLES ON CACHE BOOL "Build the MITK examples" FORCE)
set(BLUEBERRY_BUILD_ALL_PLUGINS ON CACHE BOOL "Build all BlueBerry plugins" FORCE)
