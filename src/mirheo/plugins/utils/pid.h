// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <fstream>

namespace mirheo
{

/** PID controller class.
    \tparam ControlType The Datatype of the scalar to control.
 */
template <typename ControlType>
class PidControl {
public:
    /** Initialize the PID.
        \param [in] initError The initial difference between the current state and the target.
        \param [in] Kp The proportional coefficient.
        \param [in] Ki The integral coefficient.
        \param [in] Kd The derivative coefficient.
     */
    __HD__ PidControl(ControlType initError, real Kp, real Ki, real Kd) :
        Kp_(Kp),
        Ki_(Ki),
        Kd_(Kd),
        oldError_(initError),
        sumError_(initError)
    {}

    /** Update the internal state of the PID controller.
        \param [in] error The difference between the current state and the target.
        \return The control variable value.
    */
    __HD__ inline ControlType update(ControlType error)
    {
        ControlType derError;

        derError   = error - oldError_;
        sumError_ += error;
        oldError_  = error;

        return Kp_ * error + Ki_ * sumError_ + Kd_ * derError;
    }

    /** Serialize a controller into a stream.
        \param [out] stream The stream that will contain the serialized data.
        \param [in] pid The current state to serialize.
        \return The stream.
     */
    friend std::ofstream& operator<<(std::ofstream& stream, const PidControl<ControlType>& pid)
    {
        stream << pid.oldError_ << std::endl
               << pid.sumError_ << std::endl;
        return stream;
    }

    /** Deserialize a controller from a stream.
        \param [in] stream The stream that contains the serialized data.
        \param [out] pid The deserialized state.
        \return The stream.
     */
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
