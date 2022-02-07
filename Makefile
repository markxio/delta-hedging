.RECIPEPREFIX +=
#######################################################################################
.PHONY: help
help:
  @echo "Makefile Usage:"
  @echo "  make build TARGET=<sw_emu/hw_emu/hw>"
  @echo "      Command to generate the design for specified Target and Device."
  @echo ""
  @echo "  make host "
  @echo "      Command to generate host."
  @echo ""
  @echo "  make device "
  @echo "      Command to generate hardware platform files(xo,xclbin)."
  @echo ""
  @echo "  make clean "
  @echo "      Command to remove the generated files."
  @echo ""
#######################################################################################
TARGET := sw_emu
PLATFORM := xilinx_u280_xdma_201920_3 #xilinx_u250_xdma_201830_2
HOST_EXE := host
XO := krnl_scenario.$(TARGET).xo
XCLBIN := krnl_scenario.$(TARGET).xclbin

KERNEL_NAME := krnl_scenario

VITIS_LIBRARIES_QF := /home/nx08/nx08/markkfpga/Vitis_Libraries/quantitative_finance
VITIS_LIBRARIES_L1 := /L1/include/
VITIS_LIBRARIES_L2 := /L2/include/

DEVICE_SRCDIR:=src/kernel
TEMP_DIR := ./_x.$(TARGET)

SOURCES  := $(wildcard $(DEVICE_SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(DEVICE_SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(DEVICE_SRCDIR)/%.cpp=%.xo)

# Host building global settings
CXXFLAGS := -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -Wall -O3 -std=c++11 -L$(XILINX_XRT)/lib/ -lpthread -lrt -lstdc++
CXXFLAGS2 := -lhostsupport -lxilinxopencl #-lOpenCL 

# Kernel compiler & linker global settings
KRNL_COMPILE_OPTS := -t $(TARGET) --config ../design.cfg --save-temps -j 48 -I$(VITIS_LIBRARIES_QF)$(VITIS_LIBRARIES_L1) -I$(VITIS_LIBRARIES_QF)$(VITIS_LIBRARIES_L2)
KRNL_LINK_OPTS := -t $(TARGET) --config ../link.cfg -j 48

build:  $(XO) $(XCLBIN) $(HOST_EXE) emconfig
  
.PHONY: device
.ONESHELL:
device: mkrefdir $(OBJECTS) $(XCLBIN)
  cp reference_files_$(TARGET)/$(XCLBIN) bin/.

mkrefdir:
  rm -Rf reference_files_$(TARGET)
  mkdir -p reference_files_$(TARGET)
  mkdir -p bin

# Building kernel
.PHONY: $(OBJECTS)
$(OBJECTS): %.xo : $(DEVICE_SRCDIR)/%.cpp
  cd reference_files_$(TARGET) ; v++ $(KRNL_COMPILE_OPTS) -c -k krnl_scenario -I'../include' -I'../$(<D)' -o'$@' ../$<

.PHONY: $(XCLBIN)
$(XCLBIN): $(OBJECTS)
  cd reference_files_$(TARGET) ; v++ $(KRNL_LINK_OPTS) -l -o'$@' $(+)

.PHONY: $(HOST_EXE)
# Building Host
$(HOST_EXE): src/host/host.cpp
  g++ $(CXXFLAGS) -o bin/host '$<' $(CXXFLAGS2)

.PHONY: emconfig
emconfig:
  emconfigutil --platform $(PLATFORM)

# Cleaning stuff
.PHONY: clean
clean:
  rm -f $(HOST_EXE) *krnl_scenario.$(TARGET).$(PLATFORM).* *.log *.json *.xo
