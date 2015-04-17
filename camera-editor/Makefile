# Makefile

SHELL := /bin/bash

CC=gcc
LD=gcc

CFLAGS += -O2 -std=c++11
LFLAGS += -O2
GLUT_PATH=/Developer/GPU\ Computing/shared/inc/ 
INCLUDES= -I../source -I/usr/include -I/usr/local/include -I$(GLUT_PATH)
STATIC_LIBRARIES= -L/usr/lib  -L/usr/local/lib -lglut -lGLU -lGL -lstdc++ -lm

ifneq (,$(findstring Darwin,$(shell uname)))	
	STATIC_LIBRARIES += -framework OpenGL
endif	

.DEFAULT: camedit;

camedit: main.o 
	@echo linking stage
	$(LD) $(LFLAGS) main.o -o camedit  $(STATIC_LIBRARIES)
		
main.o:  main.cpp
	@echo MAKING main.o
	$(CC) $(CFLAGS) $(INCLUDES) -c -o main.o main.cpp
clean:
	rm -f *.o
	rm -f camedit
