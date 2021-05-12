CPP=/usr/local/cuda/bin/nvcc
CFLAGS=-O3 -std=c++14 -arch sm_70
INCLUDE=
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp')
TARGET=cuda-blur-stencil


.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cu $(HEADERS)
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@


# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGET)
