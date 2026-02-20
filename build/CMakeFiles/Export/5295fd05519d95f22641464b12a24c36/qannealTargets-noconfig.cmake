#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "qanneal::qanneal_core" for configuration ""
set_property(TARGET qanneal::qanneal_core APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(qanneal::qanneal_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libqanneal_core.a"
  )

list(APPEND _cmake_import_check_targets qanneal::qanneal_core )
list(APPEND _cmake_import_check_files_for_qanneal::qanneal_core "${_IMPORT_PREFIX}/lib/libqanneal_core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
