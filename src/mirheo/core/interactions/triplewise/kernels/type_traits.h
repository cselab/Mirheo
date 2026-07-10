// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

namespace mirheo
{

/** \brief A type trait that states if a triplewise kernel outputs a force
    \tparam T The kernel type

    By default, all kernels do output a force.
    Please add a template specialization if it is not the case.
 */
template <class T>
struct outputsForce
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = true;
};

/** \brief A type trait that states if a triplewise kernel outputs density
    \tparam T The kernel type

    By default, all kernels do not output density.
    Please add a template specialization if it is the case.
 */
template <class T>
struct outputsDensity
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = !outputsForce<T>::value;
};

/** \brief A type trait that states if a triplewise kernel needs densities as input.
    \tparam T The kernel type

    By default, kernels do not need density as input.
    Please add a template specialization if it is the case.
 */
template <class T>
struct requiresDensity
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = false;
};

/** \brief A type trait that states if a triplewise kernel is of type Final.
    \tparam T The kernel type
 */
template <class T>
struct isFinal
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = outputsForce<T>::value;
};

} // namespace mirheo
