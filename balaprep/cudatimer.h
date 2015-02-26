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
