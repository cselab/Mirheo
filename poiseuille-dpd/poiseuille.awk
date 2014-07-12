#!/bin/bash

function abs(x)
{
    return ((x < 0.0) ? -x : x)
}

BEGIN {
    sumki2 = 0
    sumfiki = 0
}

{
    xi = $1
    ki = abs(xi) * xi - 5 * xi
    sumki2 += ki * ki
    sumfiki += $2 * ki   
}

END {
    print "sumfiki", sumfiki;
    print "sumki2", sumki2
    alpha = sumfiki / sumki2;
    print "alpha", alpha
    print "viscosity", -0.05 * 3 / alpha
}
