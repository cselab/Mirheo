#pragma once

#include <core/utils/cpu_gpu_defines.h>

#include <fstream>

template <typename ControlType>
class PidControl {
public:
    __HD__ PidControl(ControlType initError, real Kp, real Ki, real Kd) :
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

    friend std::ofstream& operator<<(std::ofstream& stream, const PidControl<ControlType>& pid)
    {
        stream << pid.oldError << std::endl
               << pid.sumError << std::endl;
        return stream;
    }
    
    friend std::ifstream& operator>>(std::ifstream& stream, PidControl<ControlType>& pid)
    {
        stream >> pid.oldError
               >> pid.sumError;
        return stream;
    }
    
private:
    real Kp, Ki, Kd;
    ControlType oldError, sumError;
};


