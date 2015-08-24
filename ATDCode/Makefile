
.SUFFIXES: .c .u
CC= gcc
CFLAGS= -O3 -Wall -g
LDFLAGS= -lm

LOBJECTS= main.o cokus.o

LSOURCE= main.c cokus.c

lda:	$(LOBJECTS)
	$(CC) $(CFLAGS) $(LOBJECTS) -o ATD $(LDFLAGS)

clean:
	-rm -f *.o
