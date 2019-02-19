#pragma once

template <typename ControlType>
class PidControl {
public:
    PidControl(ControlType init_error, float Kp, float Ki, float Kd) :
        Kp(Kp), Ki(Ki), Kd(Kd), old_error(init_error), sum_error(init_error)
    {}

    ControlType update(ControlType error) {
        ControlType der_error;

        der_error  = error - old_error;
        sum_error += error;
        old_error  = error;

        return Kp * error + Ki * sum_error + Kd * der_error;
    }
    
private:
    float Kp, Ki, Kd;
    ControlType old_error, sum_error;
};
