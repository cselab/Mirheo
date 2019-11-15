#!/usr/bin/env python

import mirheo as mir

def printList(channels):
    for name in channels:
        print("    * \"" + name + "\"")


print()
print("* Reserved particle channel fields:")
printList(mir.ParticleVectors.getReservedParticleChannels())

print()
print("* Reserved object channel fields:")
printList(mir.ParticleVectors.getReservedObjectChannels())

print()
print("* Reserved bisegment channel fields:")
printList(mir.ParticleVectors.getReservedBisegmentChannels())
print()
