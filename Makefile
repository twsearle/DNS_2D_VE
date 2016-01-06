
CFLAGS= -Wall -fpic -Iinclude $(TOBYCPATHS)
LIBFLAGS= -lhdf5 -lhdf5_hl -lfftw3 -lm 
MYFLAGS= -O3

ODIR= obj
SDIR= src
HEADERS= include/fields_2D.h include/fields_1D.h include/fields_IO.h include/time_steppers.h include/time_steppers_linear.h

N_OBJ= obj/fields_2D.o obj/fields_IO.o obj/DNS_2D_Newt.o obj/time_steppers.o
N_LIN_OBJ= obj/fields_1D.o obj/fields_IO.o obj/DNS_2D_linear_Newt.o obj/time_steppers_linear.o

V_OBJ= obj/fields_2D.o obj/fields_IO.o obj/DNS_2D_Visco.o obj/time_steppers.o
V_LIN_OBJ= obj/fields_1D.o obj/fields_IO.o obj/DNS_2D_linear_Visco.o obj/time_steppers_linear.o

.PHONY : clean debug oscil

oscil: MYFLAGS= -DOSCIL_FLOW -O3 
	#-DMYDEBUG 
debug: MYFLAGS= -pg -g -ggdb -DMYDEBUG 

oscil debug : clean all 

CC=gcc

#CFLAGS = $(DEBUGFLAGS) -Wall -fpic -Iinclude $(TOBYCPATHS) 
#-L/opt/local/lib -I/opt/local/include 

DNS_2D_Visco : $(V_OBJ) 
	$(CC) -o DNS_2D_Visco $(V_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

DNS_2D_Newt : $(N_OBJ)
	$(CC) -o DNS_2D_Newt $(N_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

DNS_2D_linear_Newt : $(N_LIN_OBJ)
	$(CC) -o DNS_2D_linear_Newt $(N_LIN_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

DNS_2D_linear_Visco : $(V_LIN_OBJ)
	$(CC) -o DNS_2D_linear_Visco $(V_LIN_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

test_fields : obj/fields_2D.o obj/fields_1D.o obj/fields_IO.o obj/test_fields.o obj/test_fields_1D.o obj/stupid_1D.o obj/stupid_2D.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o obj/fields_IO.o $(CFLAGS) $(LIBFLAGS)
	$(CC) -o test_fields_1D obj/test_fields_1D.o obj/fields_1D.o obj/fields_IO.o $(CFLAGS) $(LIBFLAGS)
	$(CC) -o stupid_1D obj/stupid_1D.o obj/fields_1D.o obj/fields_IO.o $(CFLAGS) $(LIBFLAGS)
	$(CC) -o stupid_2D obj/stupid_2D.o obj/fields_2D.o obj/fields_IO.o $(CFLAGS) $(LIBFLAGS)

$(ODIR)/%.o : $(SDIR)/%.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) $(MYFLAGS) 

all : 	$(N_OBJ) $(V_OBJ) $(N_LIN_OBJ) $(V_LIN_OBJ)
	$(CC) -o DNS_2D_Newt $(N_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)
	$(CC) -o DNS_2D_Visco $(V_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)
	$(CC) -o DNS_2D_linear_Newt $(N_LIN_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)
	$(CC) -o DNS_2D_linear_Visco $(V_LIN_OBJ) $(CFLAGS) $(LIBFLAGS) $(MYFLAGS)

clean :
	rm -f ./obj/*.o DNS_2D_Visco DNS_2D_Newt test_fields test_fields_1D 
	rm -f ./operators/*.h5
	rm -f ./initial.h5
