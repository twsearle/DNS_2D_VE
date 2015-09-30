
CFLAGS= -Wall -fpic -Iinclude $(TOBYCPATHS)
LIBFLAGS= -lhdf5 -lhdf5_hl -lfftw3 -lm 
MYFLAGS= -O3

ODIR= obj
SDIR= src
HEADERS= include/fields_2D.h include/fields_IO.h include/time_steppers.h include/time_steppers_linear.h

.PHONY : clean debug oscil

oscil: MYFLAGS= -DOSCIL_FLOW -O3
debug: MYFLAGS= -pg -g -ggdb -DMYDEBUG 

oscil debug : clean all

CC=gcc

#CFLAGS = $(DEBUGFLAGS) -Wall -fpic -Iinclude $(TOBYCPATHS) 
#-L/opt/local/lib -I/opt/local/include 

DNS_2D_Visco : obj/fields_2D.o obj/fields_IO.o obj/DNS_2D_Visco.o obj/time_steppers.o
	$(CC) -o DNS_2D_Visco obj/DNS_2D_Visco.o obj/fields_2D.o obj/fields_IO.o obj/time_steppers.o $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

DNS_2D_Newt : obj/fields_2D.o obj/fields_IO.o obj/DNS_2D_Newt.o obj/time_steppers.o
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/fields_2D.o obj/fields_IO.o obj/time_steppers.o $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

test_fields : obj/fields_2D.o obj/fields_IO.o obj/test_fields.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o obj/fields_IO.o $(CFLAGS) $(LIBFLAGS)

$(ODIR)/%.o : $(SDIR)/%.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) $(MYFLAGS)

all : obj/fields_IO.o obj/fields_2D.o obj/time_steppers.o obj/test_fields.o obj/DNS_2D_Newt.o obj/DNS_2D_Visco.o 
	#$(CC) -o test_fields obj/test_fields.o obj/fields_IO.o obj/fields_2D.o $(CFLAGS) $(LIBFLAGS)
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/time_steppers.o obj/fields_IO.o obj/fields_2D.o $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)
	$(CC) -o DNS_2D_Visco obj/DNS_2D_Visco.o obj/fields_2D.o obj/fields_IO.o obj/time_steppers.o $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

clean :
	rm -f ./obj/*.o DNS_2D_Newt test_fields test_fields_1 test_fields_2
	rm -f ./operators/*.h5
	rm -f ./initial.h5
