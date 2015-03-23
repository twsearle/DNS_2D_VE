CC=gcc
DEBUGFLAGS = -pg -g -ggdb -DMYDEBUG
CFLAGS = -Wall -fpic -Iinclude $(TOBYCPATHS) 
#CFLAGS = $(DEBUGFLAGS) -Wall -fpic -Iinclude $(TOBYCPATHS) 
#-L/opt/local/lib -I/opt/local/include 

DNS_2D_Visco : obj/fields_2D.o obj/DNS_2D_Visco.o obj/time_steppers.o
	$(CC) -o DNS_2D_Visco obj/DNS_2D_Visco.o obj/fields_2D.o obj/time_steppers.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

DNS_2D_Newt : obj/fields_2D.o obj/DNS_2D_Newt.o obj/time_steppers.o
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/fields_2D.o obj/time_steppers.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

test_fields : obj/fields_2D.o obj/test_fields.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

test_fields_1 : obj/fields_2D.o obj/test_fields_1.o
	$(CC) -o test_fields_1 obj/test_fields_1.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

test_fields_2 : obj/fields_2D.o obj/test_fields_2.o
	$(CC) -o test_fields_2 obj/test_fields_2.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

obj/DNS_2D_Visco.o : src/DNS_2D_Visco.c include/fields_2D.h include/time_steppers.h
	$(CC) -c src/DNS_2D_Visco.c -o obj/DNS_2D_Visco.o $(CFLAGS) -O3

obj/DNS_2D_Newt.o : src/DNS_2D_Newt.c include/fields_2D.h include/time_steppers.h
	$(CC) -c src/DNS_2D_Newt.c -o obj/DNS_2D_Newt.o $(CFLAGS) -O3

obj/test_fields.o : src/test_fields.c include/fields_2D.h
	$(CC) -c src/test_fields.c -o obj/test_fields.o $(CFLAGS) -O3

#obj/test_fields_1.o : src/test_fields_1.c include/fields_2D.h
#	$(CC) -c src/test_fields_1.c -o obj/test_fields_1.o $(CFLAGS)
#
#obj/test_fields_2.o : src/test_fields_2.c include/fields_2D.h
#	$(CC) -c src/test_fields_2.c -o obj/test_fields_2.o $(CFLAGS)

obj/fields_2D.o : src/fields_2D.c include/fields_2D.h
	$(CC) -c src/fields_2D.c -o obj/fields_2D.o $(CFLAGS) -O3

obj/time_steppers.o : src/time_steppers.c include/time_steppers.h
	$(CC) -c src/time_steppers.c -o obj/time_steppers.o $(CFLAGS) -O3 

all : obj/fields_2D.o obj/time_steppers.o obj/DNS_2D_Newt.o obj/DNS_2D_Visco.o obj/test_fields.o  
	#obj/test_fields_1.o obj/test_fields_2.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3
#	$(CC) -o test_fields_1 obj/test_fields_1.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm
#	$(CC) -o test_fields_2 obj/test_fields_2.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/time_steppers.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3
	$(CC) -o DNS_2D_Visco obj/DNS_2D_Visco.o obj/fields_2D.o obj/time_steppers.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm -O3

debug : obj/fields_2D.o obj/time_steppers.o obj/DNS_2D_Newt.o obj/DNS_2D_Visco.o 
	$(CC) -c src/time_steppers.c -o obj/time_steppers.o $(DEBUGFLAGS) $(CFLAGS) 
	$(CC) -c src/fields_2D.c -o obj/fields_2D.o $(DEBUGFLAGS) $(CFLAGS) 
	$(CC) -c src/DNS_2D_Newt.c -o obj/DNS_2D_Newt.o $(DEBUGFLAGS) $(CFLAGS)
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/time_steppers.o obj/fields_2D.o $(DEBUGFLAGS) $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm
	$(CC) -c src/DNS_2D_Visco.c -o obj/DNS_2D_Visco.o $(DEBUGFLAGS) $(CFLAGS)
	$(CC) -o DNS_2D_Visco obj/DNS_2D_Visco.o obj/time_steppers.o obj/fields_2D.o $(DEBUGFLAGS) $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm


.PHONY : clean debug
clean :
	rm -f ./obj/*.o DNS_2D_Newt test_fields test_fields_1 test_fields_2
	rm -f ./operators/*.h5
	rm -f ./initial.h5
