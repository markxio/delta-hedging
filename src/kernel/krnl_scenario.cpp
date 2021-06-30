#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include "ap_int.h"
#include "hls_stream.h"

#include "../include/replicationerror.hpp"

extern "C" void krnl_scenario(double maturity, unsigned int strike, unsigned int underlying, double volatility, double riskFreeRate, const unsigned int scenarios, unsigned int hedgesNum, double *sampleInput, double *result) {
#pragma HLS INTERFACE m_axi port=sampleInput offset=slave bundle=sampleInput_port
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=result_port

#pragma HLS INTERFACE s_axilite port=maturity bundle=control
#pragma HLS INTERFACE s_axilite port=strike bundle=control
#pragma HLS INTERFACE s_axilite port=underlying bundle=control
#pragma HLS INTERFACE s_axilite port=volatility bundle=control
#pragma HLS INTERFACE s_axilite port=riskFreeRate bundle=control
#pragma HLS INTERFACE s_axilite port=scenarios bundle=control
#pragma HLS INTERFACE s_axilite port=hedgesNum bundle=control
#pragma HLS INTERFACE s_axilite port=sampleInput bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    double dividendYield = 0.0;
    bool optionType = 0; // 1 for put, 0 for call
    //const xf::fintech::Type optionType = xf::fintech::Type::Put;

    // MCEuropeanEngine args
    unsigned int seed = 3;
    double requiredTolerance = 0.001;
    unsigned int requiredSamples = SAMP_NUM;    
    unsigned int timeSteps = hedgesNum; 

    ap_uint<32> seeds[MCM_NM];
    for (int i = 0; i < MCM_NM; ++i) {
        seeds[i] = seed + i * 1000;
    }

    // implement MCEuropeanEngine with custom PathPricer
    rep::MCEuropeanEngine<DT_USED, MCM_NM>(underlying, volatility, dividendYield, riskFreeRate, maturity, strike, optionType, seeds, /*&result[0]*/ result, requiredTolerance, requiredSamples, timeSteps);		
}
