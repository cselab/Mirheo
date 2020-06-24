#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <fstream>

namespace mirheo
{

template <typename ControlType>
class PidControl {
public:
    __HD__ PidControl(ControlType initError, real Kp, real Ki, real Kd) :
        Kp_(Kp),
        Ki_(Ki),
        Kd_(Kd),
        oldError_(initError),
        sumError_(initError)
    {}

    __HD__ inline ControlType update(ControlType error)
    {
        ControlType derError;

        derError   = error - oldError_;
        sumError_ += error;
        oldError_  = error;

        return Kp_ * error + Ki_ * sumError_ + Kd_ * derError;
    }

    friend std::ofstream& operator<<(std::ofstream& stream, const PidControl<ControlType>& pid)
    {
        stream << pid.oldError_ << std::endl
               << pid.sumError_ << std::endl;
        return stream;
    }

    friend std::ifstream& operator>>(std::ifstream& stream, PidControl<ControlType>& pid)
    {
        stream >> pid.oldError_
               >> pid.sumError_;
        return stream;
    }

private:
    real Kp_, Ki_, Kd_;
    ControlType oldError_, sumError_;
};

} // namespace mirheo
