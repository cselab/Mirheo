/*
 *  cudatimer.h
 *  Part of uDeviceX/balaprep/
 *
 *  Created and authored by Massimo Bernaschi on 2015-02-26.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

class CudaEventTimer {
public:
  CudaEventTimer() : tStarted(false), tStopped(false) {
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
  }
  ~CudaEventTimer() {
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);
  }
  void start(cudaStream_t s = 0) { cudaEventRecord(tStart, s); 
                                   tStarted = true; tStopped = false; }
  void stop(cudaStream_t s = 0)  { assert(tStarted);
                                   cudaEventRecord(tStop, s); 
                                   tStarted = false; tStopped = true; }
  float elapsed() {
    assert(tStopped);
    if (!tStopped) return 0; 
    cudaEventSynchronize(tStop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tStart, tStop);
    return milliseconds;
  }

private:
  bool tStarted, tStopped;
  cudaEvent_t tStart, tStop;
};
