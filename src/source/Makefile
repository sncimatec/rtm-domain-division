CWPROOT = ../../lib/cwp

# source files.
SRC = ptsrc.cpp

OBJ = $(SRC:.cpp=.o)

OUT = ../../lib/libsource.a

# include directories
INCLUDES = -I../../include/source -I$(CWPROOT)/include

# C compiler flags
CFLAGS = -O3 -Wall -Wextra -pedantic -fpermissive 

# compiler
CC = dpcpp

.SUFFIXES: .cpp

default: $(OUT)

.cpp.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT)

