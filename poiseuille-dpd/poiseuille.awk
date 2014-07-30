#!/bin/bash

function abs(x)
{
    return ((x < 0.0) ? -x : x)
}

BEGIN {
    sumki2 = 0
    sumfiki = 0
    vavg = 0
}

{
    xi = $1
    ki = abs(xi) * xi - 5 * xi
    sumki2 += ki * ki
    sumfiki += $2 * ki
    vavg += $2 * ($1 >= 0) * 5 / 50
}

END {
    print "sumfiki", sumfiki;
    print "sumki2", sumki2

   
    alpha = sumfiki / sumki2;
    print "alpha", alpha
    rho = 6
    gz = 0.055
    viscosity = -rho * gz / (2 * alpha)
    print "viscosity", viscosity 

    vavg /= 5
    #vavg *= 5 / 100
    print "viscosity2: ", rho * gz * 25 / (12 * vavg) 
}
