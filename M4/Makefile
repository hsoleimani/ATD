CC = gcc -O3 -Wall -g
LDFLAGS = -lgsl -lgslcblas -lm

GSL_INCLUDE = /usr/local/include/gsl/include
GSL_LIB = /usr/local/lib


LSOURCE = main.c opt.c
LHEADER = M4.h main.h opt.h

all: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) $(LDFLAGS) -o M4

clean:
	-rm -f M4


