#pragma once

#include <core/utils/cpu_gpu_defines.h>

template <typename ControlType>
class PidControl {
public:
    __HD__ PidControl(ControlType initError, float Kp, float Ki, float Kd) :
        Kp(Kp),
        Ki(Ki),
        Kd(Kd),
        oldError(initError),
        sumError(initError)
    {}

    __HD__ inline ControlType update(ControlType error)
    {
        ControlType derError;
        
        derError  = error - oldError;
        sumError += error;
        oldError  = error;

        return Kp * error + Ki * sumError + Kd * derError;
    }
    
private:
    float Kp, Ki, Kd;
    ControlType oldError, sumError;
};
