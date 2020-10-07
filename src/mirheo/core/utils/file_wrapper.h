// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <cstdio>
#include <string>

namespace mirheo
{

/** \brief Wrapper for c-stype FILE with RAII
 */
class FileWrapper
{
public:
    /// Used to construct special stream handlers for cout and cerr
    enum class SpecialStream {Cout, Cerr};

    /// return status when opening the files
    enum class Status {Success, Failed};

    /// default constructor
    FileWrapper();

    /** \brief Construct a FileWrapper and tries to open the file \p fname in \p mode mode.
        \param fname The name of the file to open
        \param mode The open mode, e.g. "r" for read mode (see docs of std::fopen)

        This method will die if the file was not found
     */
    FileWrapper(const std::string& fname, const std::string& mode);
    ~FileWrapper();

    FileWrapper           (const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;

    FileWrapper           (FileWrapper&&); ///< move constructor
    FileWrapper& operator=(FileWrapper&&); ///< move assignment

    /** \brief Open a file in a given mode
        \param fname The name of the file to open
        \param mode The open mode, e.g. "r" for read mode (see docs of std::fopen)
        \return Status::Success if the file was open succesfully, Status::Failed otherwise
    */
    Status open(const std::string& fname, const std::string& mode);

    /** \brief Set the wrapper to write to a special stream
        \param stream stdout or stderr
        \param forceFlushOnClose If set to \c true, the buffer will be flushed when close() is called.
        \return success status
    */
    Status open(SpecialStream stream, bool forceFlushOnClose);

    /// \return the C-style file handler
    FILE* get() {return file_;}

    /** \brief Close the current handler.

        This does not need to be called manually unless reopening a new file, since it will be called in the destructor.

        If the handler was pointing to a file, the file is close.
        If the handler was pointing to a special stream (cout, cerr), fflush may be called
        (see forceFlushOnClose parameter in open(SpecialStream, bool)) but the stream is not closed.
        If the handler did not point to anything, nothing happens.
     */
    void close();

    /// Wrapper around std::fread. Throws an exception if reading failed.
    void fread(void *ptr, size_t size, size_t count);

private:
    FILE *file_ {nullptr};
    bool forceFlushOnClose_{false};
};

} // namespace mirheo
