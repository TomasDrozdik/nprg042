CPP=g++
CFLAGS=-Wall -O3 -std=c++17 -fopenmp
INCLUDE=
LDFLAGS=
LIBS=
LIBDIRS=
SOURCE=minimum.cpp
HEADERS=$(shell find . -name '*.hpp')
EXECUTABLE=./minimum


.PHONY: all clear clean purge

all: $(EXECUTABLE)



# Building Targets

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@



# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
