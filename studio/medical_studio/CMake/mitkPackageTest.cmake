if(BUILD_TESTING)

#package testing
  if(NOT MITK_FAST_TESTING)

    # package testing in windows only for release
    if(WIN32)
      add_test(NAME mitkPackageTest CONFIGURATIONS Release
               COMMAND ${CMAKE_COMMAND} --build ${MITK_BINARY_DIR} --config Release --target package)

      set_tests_properties(mitkPackageTest PROPERTIES TIMEOUT 14400)
    elseif(CMAKE_BUILD_TYPE)
      add_test( NAME mitkPackageTest
                COMMAND ${CMAKE_COMMAND} --build ${MITK_BINARY_DIR} --config ${CMAKE_BUILD_TYPE} --target package)

      set_tests_properties(mitkPackageTest PROPERTIES TIMEOUT 14400 RUN_SERIAL TRUE)
    endif()

  endif() # NOT MITK_FAST_TESTING

endif(BUILD_TESTING)
