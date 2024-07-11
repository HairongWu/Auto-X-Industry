#-----------------------------------------------------------------------------
# ITK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED ITK_DIR AND NOT EXISTS ${ITK_DIR})
  message(FATAL_ERROR "ITK_DIR variable is defined but corresponds to non-existing directory")
endif()

set(proj ITK)
mitk_query_custom_ep_vars()

set(proj_DEPENDENCIES GDCM ${${proj}_CUSTOM_DEPENDENCIES})

if(MITK_USE_HDF5)
  list(APPEND proj_DEPENDENCIES HDF5)
endif()

set(ITK_DEPENDS ${proj})

if(NOT DEFINED ITK_DIR)

  set(additional_cmake_args -DUSE_WRAP_ITK:BOOL=OFF)

  list(APPEND additional_cmake_args
    -DITKV4_COMPATIBILITY:BOOL=OFF
    -DITK_LEGACY_REMOVE:BOOL=ON
  )

  # Keep the behaviour of ITK 4.3 which by default turned on ITK Review
  # see MITK bug #17338
  list(APPEND additional_cmake_args
    -DModule_ITKReview:BOOL=ON
    -DModule_ITKOpenJPEG:BOOL=ON # for 4.7, the OpenJPEG is needed by review but the variable must be set
    -DModule_IsotropicWavelets:BOOL=ON
  )

  if(CTEST_USE_LAUNCHERS)
    list(APPEND additional_cmake_args
      "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
    )
  endif()

  ExternalProject_Add(${proj}
     LIST_SEPARATOR ${sep}
     UPDATE_COMMAND ""
     GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
     GIT_TAG 311b7060ef39e371f3cd209ec135284ff5fde735 # tag: v5.4.0
     CMAKE_GENERATOR ${gen}
     CMAKE_GENERATOR_PLATFORM ${gen_platform}
     CMAKE_ARGS
       ${ep_common_args}
       ${additional_cmake_args}
       -DITK_SKIP_PATH_LENGTH_CHECKS:BOOL=ON
       -DBUILD_EXAMPLES:BOOL=OFF
       -DITK_USE_SYSTEM_GDCM:BOOL=ON
       -DGDCM_DIR:PATH=${GDCM_DIR}
       -DITK_USE_SYSTEM_HDF5:BOOL=ON
       -DHDF5_DIR:PATH=${HDF5_DIR}
       -DModule_GrowCut:BOOL=ON
       ${${proj}_CUSTOM_CMAKE_ARGS}
     CMAKE_CACHE_ARGS
       ${ep_common_cache_args}
       ${${proj}_CUSTOM_CMAKE_CACHE_ARGS}
     CMAKE_CACHE_DEFAULT_ARGS
       ${ep_common_cache_default_args}
       ${${proj}_CUSTOM_CMAKE_CACHE_DEFAULT_ARGS}
     DEPENDS ${proj_DEPENDENCIES}
    )

  set(ITK_DIR ${ep_prefix})
  mitkFunctionInstallExternalCMakeProject(${proj})

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
