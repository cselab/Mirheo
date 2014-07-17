/*
 *  Misc.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cstdlib>
#include <cmath>
#include <time.h>

#include "minIni.h"

#pragma once

typedef float real;

extern minIni *configParser;

namespace Unroll
{
    template<int offset>
    struct UnrollerInternal
    {
        template<typename Action>
        static void step(Action& act, int i)
        {
            act(i + offset - 1);
            UnrollerInternal<offset - 1>::step(act, i);
        }
    };
    
    template<>
    struct UnrollerInternal<0> {
        template<typename Action>
        static void step(Action& act, size_t i) {
        }
    };
    
    
    //UnrollerP: loops over given size, partial unrolled
    template<int chunk = 8>
    struct UnrollerP {
        template<typename Action>
        static void step(int start, int end, Action act)
        {
            int i = start;
            for (; i < end - chunk; i += chunk)
                UnrollerInternal<chunk>::step(act, i);
            
            for (; i < end; ++i)
                act(i);
        }
    };
}




