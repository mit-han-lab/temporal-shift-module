
# Disable default rebuild Makefile rule to avoid Makeile.o match
MAKEFLAGS += -r

CXX ?= aarch64-xilinx-linux-g++
CFLAGS += -O3 -g -Wall -Wpointer-arith -std=c++14 -ffast-math -mcpu=cortex-a53
LDFLAGS += -L./ -ln2cube -lpthread -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui
CFLAGS += -fdiagnostics-color=always

SRC = ./src
COMPILE_RESULTS = compile_results
BUILD = build

VPATH = $(SRC)
CPP_FILES = $(wildcard $(SRC)/*.cpp)
OBJ = $(patsubst $(SRC)/%.cpp, $(BUILD)/%.o, $(CPP_FILES))

TARGETS = zcu104.tsm_online ultra96v2.tsm_online
.PHONY: all clean copy $(TARGETS)

all : $(TARGETS)

$(TARGETS) : SUBDIR = $(patsubst %.tsm_online,%,$@)
$(TARGETS) : ELF = $(shell find $(SUBDIR)/$(COMPILE_RESULTS) -name *.elf)
$(TARGETS) : %.tsm_online : $(OBJ)
	mkdir -p $(SUBDIR)/$(BUILD)
	$(CXX) $(CFLAGS) $^ $(ELF) -o $*/tsm_online $(LDFLAGS)

%.copy : tsm_online
	scp ./tsm_online %(dir %@):~/tsm_online/


$(BUILD)/%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean : 
	$(RM) -rf $(BUILD)
	$(RM) tsm_online
