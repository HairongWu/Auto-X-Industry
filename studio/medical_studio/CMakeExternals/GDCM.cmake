#-----------------------------------------------------------------------------
# GDCM
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED GDCM_DIR AND NOT EXISTS ${GDCM_DIR})
  message(FATAL_ERROR "GDCM_DIR variable is defined but corresponds to non-existing directory")
endif()

# Check if an external ITK build tree was specified.
# If yes, use the GDCM from ITK, otherwise ITK will complain
if(ITK_DIR)
  find_package(ITK)
  if(ITK_GDCM_DIR)
    set(GDCM_DIR ${ITK_GDCM_DIR})
  endif()
endif()

set(proj GDCM)
mitk_query_custom_ep_vars()

set(proj_DEPENDENCIES ${${proj}_CUSTOM_DEPENDENCIES})
set(GDCM_DEPENDS ${proj})

if(NOT DEFINED GDCM_DIR)

  set(additional_args )
  if(CTEST_USE_LAUNCHERS)
    list(APPEND additional_args
      "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
    )
  endif()

  # On Mac some assertions fail that prevent reading certain DICOM files. Bug #19995
  if(APPLE)
    list(APPEND additional_args
      "-DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG} -DNDEBUG"
    )
  endif()

  ExternalProject_Add(${proj}
     LIST_SEPARATOR ${sep}
     GIT_REPOSITORY https://github.com/malaterre/GDCM.git
     GIT_TAG v3.0.14
     GIT_SUBMODULES ""
     CMAKE_GENERATOR ${gen}
     CMAKE_GENERATOR_PLATFORM ${gen_platform}
     CMAKE_ARGS
       ${ep_common_args}
       ${additional_args}
       -DGDCM_BUILD_SHARED_LIBS:BOOL=ON
       -DGDCM_BUILD_DOCBOOK_MANPAGES:BOOL=OFF
       ${${proj}_CUSTOM_CMAKE_ARGS}
     CMAKE_CACHE_ARGS
       ${ep_common_cache_args}
       ${${proj}_CUSTOM_CMAKE_CACHE_ARGS}
     CMAKE_CACHE_DEFAULT_ARGS
       ${ep_common_cache_default_args}
       ${${proj}_CUSTOM_CMAKE_CACHE_DEFAULT_ARGS}
     DEPENDS ${proj_DEPENDENCIES}
    )
  set(GDCM_DIR ${ep_prefix})
  mitkFunctionInstallExternalCMakeProject(${proj})

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  find_package(GDCM)

endif()
