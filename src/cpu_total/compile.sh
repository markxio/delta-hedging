#!/bin/bash

QUANTUCCIA=./Quantuccia

if [ -z "$(ls -A $QUANTUCCIA)" ]; then
    cd ../..; git submodule update --init src/cpu_total/Quantuccia/
    cd src/cpu_total; 
else
    echo "git submodule already initalized -- compiling.."
fi

# -M option for depeendency list
# -MM option for dependency list without system headers, with -o flag outputs to txt file
# -H option gives dpendency list as tree
g++ -O3 DiscreteHedgingQuantuccia.cpp -o DiscreteHedgingQuantuccia -I${QUANTUCCIA}

# compile with QuantLib (install first)
g++ -O3 DiscreteHedging.cpp -o DiscreteHedging -lQuantLib
