#! Helper function that gets all library search paths.
#!
#! Usage:
#!
#!   mitkFunctionGetLibrarySearchPaths(search_path intermediate_dir [DEBUG|MINSIZEREL|RELWITHDEBINFO])
#!
#!
#! The function creates the variable ${search_path}. The variable intermediate_dir contains
#! paths that should be added to the search_path but should not be checked for existance,
#! because the are not yet created. The option DEBUG, MINSIZEREL or RELWITHDEBINFO can be used to indicate that
#! not the paths for release configuration are requested but the debug, min size release or "release with debug info"
#! paths.
#!

function(mitkFunctionGetLibrarySearchPaths search_path intermediate_dir)

  cmake_parse_arguments(PARSE_ARGV 2 GLS "RELEASE;DEBUG;MINSIZEREL;RELWITHDEBINFO" "" "")

  set(_dir_candidates
      "${MITK_CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
      "${MITK_CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins"
      "${MITK_CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
      "${MITK_CMAKE_LIBRARY_OUTPUT_DIRECTORY}/plugins"
     )
  if(MITK_EXTERNAL_PROJECT_PREFIX)
    list(APPEND _dir_candidates
         "${MITK_EXTERNAL_PROJECT_PREFIX}/bin"
         "${MITK_EXTERNAL_PROJECT_PREFIX}/lib"
        )
  endif()

  get_property(_additional_paths GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)

  if(TARGET OpenSSL::SSL)
    if(GLS_DEBUG)
      get_target_property(_openssl_location OpenSSL::SSL IMPORTED_LOCATION_DEBUG)
    else()
      get_target_property(_openssl_location OpenSSL::SSL IMPORTED_LOCATION_RELEASE)
    endif()
    if(_openssl_location)
      get_filename_component(_openssl_location ${_openssl_location} DIRECTORY)
      set(_openssl_location "${_openssl_location}/../../bin")
      if(EXISTS ${_openssl_location})
        get_filename_component(_openssl_location ${_openssl_location} ABSOLUTE)
        list(APPEND _dir_candidates ${_openssl_location})
      endif()
    endif()
  endif()

  if(MITK_USE_HDF5)
    FIND_PACKAGE(HDF5 COMPONENTS C HL NO_MODULE REQUIRED shared)
    get_target_property(_location hdf5-shared LOCATION)
    get_filename_component(_location ${_location} PATH)
    list(APPEND _additional_paths ${_location})

    # This is a work-around. The hdf5-config.cmake file is not robust enough
    # to be included several times via find_pakcage calls.
    set(HDF5_LIBRARIES ${HDF5_LIBRARIES} PARENT_SCOPE)
  endif()

  if(_additional_paths)
    list(APPEND _dir_candidates ${_additional_paths})
  endif()

  # The code below is sub-optimal. It makes assumptions about
  # the structure of the build directories, pointed to by
  # the *_DIR variables. Instead, we should rely on package
  # specific "LIBRARY_DIRS" variables, if they exist.
  if(WIN32)
    list(APPEND _dir_candidates "${ITK_DIR}/bin")
  endif()

  if(MITK_USE_MatchPoint)
    if(WIN32)
      list(APPEND _dir_candidates "${MatchPoint_DIR}/bin")
    else()
      list(APPEND _dir_candidates "${MatchPoint_DIR}/lib")
    endif()
  endif()

  if(MITK_USE_Python3)
    list(APPEND _dir_candidates "${CTK_DIR}/CMakeExternals/Install/bin")
    get_filename_component(_python_dir "${Python3_EXECUTABLE}" DIRECTORY)
    list(APPEND _dir_candidates "${_python_dir}")
  endif()

  if(MITK_USE_CTK)
    list(APPEND _dir_candidates "${CTK_LIBRARY_DIRS}")
    foreach(_ctk_library ${CTK_LIBRARIES})
      if(${_ctk_library}_LIBRARY_DIRS)
        list(APPEND _dir_candidates "${${_ctk_library}_LIBRARY_DIRS}")
      endif()
    endforeach()
  endif()

  if(MITK_USE_BLUEBERRY)
    if(DEFINED CTK_PLUGIN_RUNTIME_OUTPUT_DIRECTORY)
      if(IS_ABSOLUTE "${CTK_PLUGIN_RUNTIME_OUTPUT_DIRECTORY}")
        list(APPEND _dir_candidates "${CTK_PLUGIN_RUNTIME_OUTPUT_DIRECTORY}")
      else()
        list(APPEND _dir_candidates "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CTK_PLUGIN_RUNTIME_OUTPUT_DIRECTORY}")
      endif()
    endif()
  endif()

  if(MITK_LIBRARY_DIRS)
    list(APPEND _dir_candidates ${MITK_LIBRARY_DIRS})
  endif()

  ###################################################################
  #get the search paths added by the mitkFunctionAddLibrarySearchPath
  file(GLOB _additional_path_info_files "${MITK_SUPERBUILD_BINARY_DIR}/MITK-AdditionalLibPaths/*.cmake")

  foreach(_additional_path_info_file ${_additional_path_info_files})
    get_filename_component(_additional_info_name ${_additional_path_info_file} NAME_WE)
    include(${_additional_path_info_file})
    if(GLS_DEBUG)
      list(APPEND _dir_candidates ${${_additional_info_name}_ADDITIONAL_DEBUG_LIBRARY_SEARCH_PATHS})
    elseif(GLS_MINSIZEREL)
      list(APPEND _dir_candidates ${${_additional_info_name}_ADDITIONAL_MINSIZEREL_LIBRARY_SEARCH_PATHS})
    elseif(GLS_RELWITHDEBINFO)
      list(APPEND _dir_candidates ${${_additional_info_name}_ADDITIONAL_RELWITHDEBINFO_LIBRARY_SEARCH_PATHS})
    else() #Release
      list(APPEND _dir_candidates ${${_additional_info_name}_ADDITIONAL_RELEASE_LIBRARY_SEARCH_PATHS})
    endif()
  endforeach(_additional_path_info_file ${_additional_path_info_files})


  ###############################################
  #sanitize all candidates and compile final list
  list(REMOVE_DUPLICATES _dir_candidates)

  set(_search_dirs )
  foreach(_dir ${_dir_candidates})
    if(EXISTS "${_dir}/${intermediate_dir}")
      list(APPEND _search_dirs "${_dir}/${intermediate_dir}")
    else()
      list(APPEND _search_dirs "${_dir}")
    endif()
  endforeach()

  # Determine the Qt6 library installation prefix
  if(QT_QMAKE_EXECUTABLE)
    if(NOT _qt_install_libs)
      if(WIN32)
        execute_process(COMMAND ${QT_QMAKE_EXECUTABLE} -query QT_INSTALL_BINS
                        OUTPUT_VARIABLE _qt_install_libs
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
      else()
        execute_process(COMMAND ${QT_QMAKE_EXECUTABLE} -query QT_INSTALL_LIBS
                        OUTPUT_VARIABLE _qt_install_libs
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
      endif()
      file(TO_CMAKE_PATH "${_qt_install_libs}" _qt_install_libs)
      set(_qt_install_libs ${_qt_install_libs} CACHE INTERNAL "Qt library installation prefix" FORCE)
    endif()
    if(_qt_install_libs)
      list(APPEND _search_dirs ${_qt_install_libs})
    endif()
  elseif(MITK_USE_Qt6)
    message(WARNING "The qmake executable could not be found.")
  endif()

  # Special handling for "internal" search dirs. The intermediate directory
  # might not have been created yet, so we can't check for its existence.
  # Hence we just add it for Windows without checking.
  set(_internal_search_dirs "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins")
  if(WIN32)
    foreach(_dir ${_internal_search_dirs})
      set(_search_dirs "${_dir}/${intermediate_dir}" ${_search_dirs})
    endforeach()
  else()
    set(_search_dirs ${_internal_search_dirs} ${_search_dirs})
  endif()
  list(REMOVE_DUPLICATES _search_dirs)

  set(${search_path} ${_search_dirs} PARENT_SCOPE)
endfunction()
