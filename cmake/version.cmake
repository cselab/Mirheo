# the version, fetched from git tag
function(getMirheoVersion VALUE)
  execute_process(COMMAND
    git describe --abbrev=0
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE version
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${VALUE} ${version} PARENT_SCOPE)
endfunction()

# the commit's SHA1, and whether the building workspace was dirty or not
function(getMirheoSHA1 VALUE)
  execute_process(COMMAND
    git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE ${sha1}
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${VALUE} ${sha1} PARENT_SCOPE)
endfunction()
