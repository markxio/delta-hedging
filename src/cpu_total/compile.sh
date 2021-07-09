#!/bin/bash

QUANTUCCIA=./Quantuccia

if [ -z "$(ls -A $QUANTUCCIA)" ]; then
    cd ../..; git submodule update --init src/cpu_total/Quantuccia/
    cd src/cpu_total; 
else
    echo "git submodule already initalized -- compiling.."
fi

g++ -std=c++11 -O3 DiscreteHedgingQuantuccia.cpp -o DiscreteHedgingQuantuccia -I${QUANTUCCIA} -I/home/software/boost/1.69-GNU/include

# compile with QuantLib (installation required, see official QuantLib website)
#g++ -O3 DiscreteHedging.cpp -o DiscreteHedging -lQuantLib
