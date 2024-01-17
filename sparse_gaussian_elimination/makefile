
# makefile for CC  add purify in front of link step
#
#
#     to run purify " purerun ../a.out "
#
#

# COMPILER FLAGS:
#-----------------
#

#COMPILER = cc
#CFLAGS = -c -g
#CFLAGS = -c  -fast
#   profiler
#CFLAGS = -c  -fast -pg


#LINK_FLAGS =
# profiler
#LINK_FLAGS = -pg
#

#
#     to run purify " purerun ../a.out "
#turn purify on and off (remember to include debugging info)
PURE =
#PURE = /software/purify-5.2/bin/purerun purify  -cache-dir=/u/paforsyth/.pscache


#
#    gcc compiler
COMPILER = gcc
#CFLAGS = -c -O
#CFLAGS = -c -finline-functions -ffast-math -O2
#CFLAGS = -c -g -Wall
CFLAGS = -c -g -fPIC
LINK_FLAGS =
CFLAGS_NO_OPT = -c
SO_FLAGS = -shared

a.out : main.o solver.o rcm.o  colamd.o mindeg.o
	$(PURE) $(COMPILER) $(LINK_FLAGS) main.o solver.o colamd.o\
                        mindeg.o rcm.o -lm

a.so : main.o solver.o rcm.o  colamd.o mindeg.o
	$(COMPILER) $(SO_FLAGS) main.o solver.o colamd.o\
                        mindeg.o rcm.o -o a.so

main.o : sparse.h solver.h main.c  mindeg.h
	$(COMPILER)  $(CFLAGS) main.c

solver.o : sparse.h solver.h solver.c
	$(COMPILER)  $(CFLAGS) solver.c

rcm.o :  rcm.c
	$(COMPILER)  $(CFLAGS) rcm.c

colamd.o : colamd.h colamd.c
	$(COMPILER)  $(CFLAGS) colamd.c

mindeg.o: mindeg.h colamd.h mindeg.c
	$(COMPILER)  $(CFLAGS) mindeg.c



