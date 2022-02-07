# Valuation of financial derivatives on FPGA with HLS
With this project we extend the existing monte carlo framework of the Vitis Quantitative Finance Library [3] and introduce a new and reusable path pricer for computing the replication error of a delta-hedging strategy in HLS and with Vitis. The implemented delta-hedging strategy follows Kamal’s 1998 research note [2]. The FPGA implementation is tested and validated against the CPU version built with QuantLib [4], the open-source library for quantitative finance. The performance of our implementation with the Vitis libraries on an Alveo U280 is compared to the QuantLib build run on an Intel Xeon Platinum 8260M. 

Further investigating the Vitis open-source libraries we showcase our approach with the Greek calculators in the presented path pricer implementation and highlight peculiarities for using the library’s existing components in this regard.


## Prerequisites

* [GNU g++](https://www.gnu.org/software/gcc/)
* [GNU Make](https://www.gnu.org/software/make/)
* [Xilinx Vitis v2020.2](https://www.xilinx.com/products/design-tools/vitis.html)
* [Vitis Libraries](https://xilinx.github.io/Vitis_Libraries/)
* [QuantLib](https://github.com/lballabio/QuantLib)
* [Quantuccia](https://github.com/pcaspers/Quantuccia)

## Usage

### Installation

To install this repository

```
git clone https://github.com/markxio/delta-hedging.git
cd delta-hedging
```

### Building

To build this repository for Alveo U280:

```
mkdir bin
make host
make device TARGET=hw DEVICE=xilinx_u280_xdma_201920_3
```

#### Compile and run CPU code

To compile the CPU reference code, go to src/cpu\_total and run `./compile.sh`. This initialises the git submodule for Quantuccia and compiles the reference code. Run the binary with `./DiscreteHedgingQuantuccia`

### Running

To run the project:

```
cd bin
./host ./krnl_scenario.hw.xclbin
```

### Parameters

  *  `SAMP_NUM`: overall number of samples/paths
  *  `SAMP_PER_SIM`: number of samples per simulation
  *  `MAX_SAMPLE`: maximum number of samples
  *  `DT_USED`: data type to be used, e.g. double
  *  `MCM_NM`: number of monte carlo modules in parallel (affects latency and resource utilization)

## Output

Returns the kernel result i.e. the computed replication error (Profit & Loss)

```
Kernel result
P&L: 0.000101
Execution complete, total runtime : 30.327 ms, (0.161 ms xfer on, 29.951 ms execute, 0.215 ms xfer off)
```
