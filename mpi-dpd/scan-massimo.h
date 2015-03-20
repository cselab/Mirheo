/*
 *  scan-massimo.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created by Massimo Bernaschi on 2015-03-09.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

void scan_massimo(const int * const count[26], int * const result[26], const int sizes[26], cudaStream_t stream);
