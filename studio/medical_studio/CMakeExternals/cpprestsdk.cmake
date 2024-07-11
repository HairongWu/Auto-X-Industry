set(proj cpprestsdk)
set(proj_DEPENDENCIES Boost ZLIB)

if(MITK_USE_${proj})
  set(${proj}_DEPENDS ${proj})

  if(DEFINED ${proj}_DIR AND NOT EXISTS ${${proj}_DIR})
    message(FATAL_ERROR "${proj}_DIR variable is defined but corresponds to non-existing directory!")
  endif()

  if(NOT DEFINED ${proj}_DIR)
    set(cmake_cache_args
      ${ep_common_cache_args}
      -DBUILD_SAMPLES:BOOL=OFF
      -DBUILD_TESTS:BOOL=OFF
      -DWERROR:BOOL=OFF
    )

    if(OPENSSL_ROOT_DIR)
      list(APPEND cmake_cache_args
        -DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}
      )
    endif()

    ExternalProject_Add(${proj}
      GIT_REPOSITORY https://github.com/MITK/cpprestsdk.git
      GIT_TAG v2.10.19-patched
      SOURCE_SUBDIR Release
      CMAKE_ARGS
        "-DBoost_DIR:PATH=${Boost_DIR}"
        ${ep_common_args}
      CMAKE_CACHE_ARGS ${cmake_cache_args}
      CMAKE_CACHE_DEFAULT_ARGS ${ep_common_cache_default_args}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(${proj}_DIR ${ep_prefix})
  else()
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  endif()
endif()
