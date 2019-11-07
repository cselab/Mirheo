# the version, fetched from git tag
macro(getMirheoVersion version_str version_number)
  execute_process(COMMAND
    git describe --abbrev=0
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE ${version_str}
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "v" "" ${version_number} ${${version_str}})
endmacro()

# the commit's SHA1, and whether the building workspace was dirty or not
macro(getMirheoSHA1 sha1)
  execute_process(COMMAND
    git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE ${sha1}
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endmacro()
