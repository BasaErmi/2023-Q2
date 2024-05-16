CC=gcc
CFLAGS=-fPIC -Wall -O2
LDFLAGS=-shared
TARGET_LIB=libparallel_for.so

all: $(TARGET_LIB)

$(TARGET_LIB): parallel_for.o
	$(CC) $(LDFLAGS) -o $@ $^

parallel_for.o: parallel_for.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET_LIB) *.o