TBBROOT=/usr/local/tbb
CPP=g++
CFLAGS=-Wall -O3 -std=c++17
INCLUDE=$(TBBROOT)/include
LDFLAGS=
LIBS=tbb
LIBDIRS=$(TBBROOT)/lib
SOURCE=tbb-matrix-tran.cpp
HEADERS=$(shell find . -name '*.hpp')
EXECUTABLE=./tbb-matrix-tran


.PHONY: all clear clean purge

all: $(EXECUTABLE)



# Building Targets

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@

# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
