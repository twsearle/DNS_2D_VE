CC=gcc
CFLAGS = -pg -g -ggdb -Wall -fpic -Iinclude $(TOBYCPATHS) #-L/opt/local/lib -I/opt/local/include 

DNS_2D_Newt : obj/fields_2D.o obj/DNS_2D_Newt.o
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm

test_fields : obj/fields_2D.o obj/test_fields.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm

test_fields_1 : obj/fields_2D.o obj/test_fields_1.o
	$(CC) -o test_fields_1 obj/test_fields_1.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm

obj/DNS_2D_Newt.o : src/DNS_2D_Newt.c include/fields_2D.h
	$(CC) -c src/DNS_2D_Newt.c -o obj/DNS_2D_Newt.o $(CFLAGS)

obj/test_fields.o : src/test_fields.c include/fields_2D.h
	$(CC) -c src/test_fields.c -o obj/test_fields.o $(CFLAGS)

obj/test_fields_1.o : src/test_fields_1.c include/fields_2D.h
	$(CC) -c src/test_fields_1.c -o obj/test_fields_1.o $(CFLAGS)

obj/fields_2D.o : src/fields_2D.c include/fields_2D.h
	$(CC) -c src/fields_2D.c -o obj/fields_2D.o $(CFLAGS) 

all : obj/fields_2D.o obj/DNS_2D_Newt.o obj/test_fields.o obj/test_fields_1.o
	$(CC) -o test_fields obj/test_fields.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm
	$(CC) -o test_fields_1 obj/test_fields_1.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm
	$(CC) -o DNS_2D_Newt obj/DNS_2D_Newt.o obj/fields_2D.o $(CFLAGS) -lhdf5 -lhdf5_hl -lfftw3 -lm

.PHONY : clean
clean :
	rm -f ./obj/*.o DNS_2D_Newt test_fields 
	rm -f ./operators/*.h5
	rm -f ./initial.h5
