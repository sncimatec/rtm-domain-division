CWPROOT = ../../lib/cwp

# source files.
SRC = fd.cpp

OBJ = $(SRC:.cpp=.o)

OUT = ../../lib/libtimestep.a

# include directories
INCLUDES = -I../../include/timestep -I../../include/util -I$(CWPROOT)/include

# C compiler flags
CFLAGS = -O3 -Wall -Wextra -pedantic -fpermissive -w -fpermissive

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

