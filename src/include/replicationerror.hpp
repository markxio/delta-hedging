#include "xf_fintech/cf_b76.hpp"
#include "xf_fintech/mc_simulation.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_fintech/enums.hpp"
//#include "xf_fintech/utils.hpp" // FPTwoMul, FPTwoSub, FPTwoAdd, FPExp, divide_by_2, mul_by_2
#ifndef __SYNTHESIS__
#include <assert.h>
#endif

#define SAMP_NUM 50000 //4 // MAX_PATHS * MCM_NM
#define MAX_PATHS 5000 //4
#define MAX_STEPS 84 //21 //3
#define MAX_SAMPLE 134217727
#define MCM_NM 10 //1
#define DT_USED double

#define MAX(a, b) ((a) > (b) ? (a) : (b)) // from xf_fintech/utils.hpp

namespace rep {

namespace qf = xf::fintech;
namespace qfi = xf::fintech::internal;

// forward delta
template <typename DT>
void cfB76EngineDelta(unsigned int call, DT f, DT k, DT v, DT r, DT t, DT* delta, DT* sqrt_t_, DT* d1_) {
    DT sqrt_t = hls::sqrtf(t);
    DT d1 = (hls::logf(f / k) + (0.5f * v * v) * t) / (v * sqrt_t);
    DT exp_rt = hls::expf(-r * t);
    DT phi_d1 = qfi::phi<DT>(d1);

    DT delta_temp;
    if(call) {
        delta_temp = exp_rt * phi_d1;
    } else {
        DT phi_d1n = 1.0 - phi_d1; // phi(-d1);
        delta_temp = -exp_rt * phi_d1n;
    }
    *delta = delta_temp;
    *sqrt_t_ = sqrt_t;
    *d1_ = d1;
}

/// @brief Spot delta (also called cash delta)
///
/// This function is optimized to be
/// synthesized by the HLS compiler and as such uses the hls namespace for the
/// maths functions.  In addition, the code
/// is structured to calculate common elements (in parallel where possible) and
/// reuse as appropriate.
///
/// @tparam DT Data Type used for this function
/// @param[in]  s     underlying
/// @param[in]  v     volatility (decimal form)
/// @param[in]  r     risk-free rate (decimal form)
/// @param[in]  t     time to maturity
/// @param[in]  k     strike price
/// @param[in]  q     continuous dividend yield rate
/// @param[in]  call  control whether call or put is calculated
/// @param[out] delta model sensitivity
template <typename DT>
void cfBSMEngineDeltaSpot(DT s, DT v, DT r, DT t, DT k, DT q, unsigned int call, DT* delta) {
    DT sqrt_t = hls::sqrtf(t);
    DT d1 = (hls::logf(s / k) + (r - q + 0.5f * v * v) * t) / (v * sqrt_t);
    DT phi_d1 = qfi::phi<DT>(d1);
    DT exp_qt = hls::expf(-q * t);

    DT delta_temp;
    if(call) {
        delta_temp = exp_qt * phi_d1;
    } else {
        DT phi_d1n = 1.0 - phi_d1; // phi(-d1);
        delta_temp = -exp_qt * phi_d1n;
    }
    *delta = delta_temp;
}

/// @brief Single option price
///
/// Produces a single price for the given input
/// parameters.  This function is optimized to be
/// synthesized by the HLS compiler and as such uses the hls namespace for the
/// maths functions.  In addition, the code
/// is structured to calculate common elements (in parallel where possible) and
/// reuse as appropriate.
///
/// @tparam DT Data Type used for this function
/// @param[in]  s     underlying
/// @param[in]  v     volatility (decimal form)
/// @param[in]  r     risk-free rate (decimal form)
/// @param[in]  t     time to maturity
/// @param[in]  k     strike price
/// @param[in]  q     continuous dividend yield rate
/// @param[in]  call  control whether call or put is calculated
/// @param[out] price call/put premium
template <typename DT>
void cfBSMEnginePrice(DT s, DT v, DT r, DT t, DT k, DT q, unsigned int call, DT* price) {

    DT delta;
    cfBSMEngineDeltaSpot<DT>(s, v, r, t, k, q, call, &delta);

    DT exp_rt = hls::expf(-r * t);
    DT k_exp_rt = k * exp_rt;

    DT sqrt_t = hls::sqrtf(t);
    DT d1 = (hls::logf(s / k) + (r - q + 0.5f * v * v) * t) / (v * sqrt_t);
    DT d2 = d1 - v * sqrt_t;
    DT phi_d2 = qfi::phi<DT>(d2);

    DT price_temp;
    if(call) {
        DT delta_temp = delta;
        DT s_delta_temp = s * delta_temp;
        DT k_exp_re_phi_d2 = k_exp_rt * phi_d2;
        price_temp = s_delta_temp - k_exp_re_phi_d2;
    } else {
        DT delta_temp = -delta;
        DT s_delta_temp = s * delta_temp;
        DT phi_d2n = 1.0 - phi_d2; // phi(-d2);
        DT k_exp_re_phi_d2n = k_exp_rt * phi_d2n;
        price_temp = s_delta_temp + k_exp_re_phi_d2n;
    }

    *price = price_temp;
}

/// @tparam DT Data Type used for this function
/// @param[in]  call  option type (call=1, put=0) 
/// @param[in]  f     underlying forward price
/// @param[in]  k     strike price
/// @param[in]  v     volatility (decimal form)
/// @param[in]  r     risk-free rate (decimal form)
/// @param[in]  t     time to maturity
/// @param[out] price option price (aka premium)
/// @param[out] delta delta sensitivity
template <typename DT>
void cfB76EnginePrice(unsigned int call, DT f, DT k, DT v, DT r, DT t, DT* price, DT* delta) {
    DT sqrt_t = hls::sqrtf(t);
    DT d1 = (hls::logf(f / k) + (0.5f * v * v) * t) / (v * sqrt_t);

    DT delta_temp, sqrt_t_, d1_;
    cfB76EngineDelta(call, f, k, v, r, t, &delta_temp, &sqrt_t_, &d1_);

    DT d2 = d1 - v * sqrt_t;
    DT phi_d2 = qfi::phi<DT>(d2);
    DT phi_d2n = 1.0 - phi_d2; // phi(-d2);    
    DT exp_rt = hls::expf(-r * t);
    DT k_exp_rt = k * exp_rt;

    DT f_delta_temp = f * delta_temp;

    DT price_temp;
    if(call) {
        DT k_exp_re_phi_d2 = k_exp_rt * phi_d2;
        price_temp = f_delta_temp - k_exp_re_phi_d2;
    } else {
        DT k_exp_re_phi_d2n = k_exp_rt * phi_d2n;
        price_temp = f_delta_temp + k_exp_re_phi_d2n;
    }
    *price = price_temp;
    *delta = delta_temp;
}

/**
 * @brief Custom path pricer to compute replication error 
 * 
 * @tparam style option type, call or put
 * @tparam DT supported data type including DT and float data type, which
 * decides the precision of result, default DT-precision data type.
 * @tparam WithAntithetic antithetic is used  for variance reduction, default this
 * feature is disabled.
 * @tparam StepFirst
 * @tparam SampNum
 * @tparam MaxSteps
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seed for each RNG.
 * @param output output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 */
template <qf::enums::OptionStyle style, typename DT, bool StepFirst, int SampNum, bool WithAntithetic, int MaxSteps = 1024>
class ReplicationPathPricer { 
    public:
        const static unsigned int InN = WithAntithetic ? 2 : 1;
        const static unsigned int OutN = InN;

        const static bool byPassGen = false;

        // configuration of the path pricer
        DT strike;
        DT underlying;
        DT drift;
        DT discount;
        DT maturity;
        DT volatility;
        DT dividendYield;
        DT r;
        bool optionType;

        ReplicationPathPricer() {
#pragma HLS inline
        }

        void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
            // MCEuropeanEngine: Put=1, Call=0
            // cfB76Engine: Put=0, Call=1
            unsigned int call;
            if(optionType == 1) {
                call = 0;
            } else {
                call = 1;
            }

            // QuantLib adds t0 underlying
            unsigned int n = steps; 

            // discrete hedging interval
            DT dt = maturity/n;
                 
            /************************/
            /*** the initial deal ***/
            /************************/
            // option fair price (Black-Scholes) at t=0
            DT premium, delta_t0;
            cfBSMEnginePrice<DT>(underlying, volatility, r, maturity, strike, dividendYield, call, &premium);
            cfBSMEngineDeltaSpot<DT>(underlying, volatility, r, maturity, strike, dividendYield, call, &delta_t0); 

            DT exp_r_x_dt = std::exp( r*dt ); // move to host
            DT money_t0 = premium - delta_t0*underlying;         
            DT money_T = money_t0 * pow(exp_r_x_dt, steps);
            
            dataflowRegion(steps, paths, underlying, volatility, r, maturity, strike, dividendYield, exp_r_x_dt, money_T, dt, call, pathStrmIn, priceStrmOut);
        } // PE()

        void dataflowRegion(ap_uint<16> steps, ap_uint<16> paths, DT underlying, DT volatility, DT r, DT maturity, DT strike, DT dividendYield, DT exp_r_x_dt, DT money_T, DT dt, unsigned int call, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
            hls::stream<DT> s1, s1Stock, s1Stock_copy, lastStock, delta, lastDelta, hedge, money_out;

#pragma HLS STREAM variable=s1              depth=2
#pragma HLS STREAM variable=s1Stock         depth=2
#pragma HLS STREAM variable=s1Stock_copy    depth=2
#pragma HLS STREAM variable=lastStock       depth=2
#pragma HLS STREAM variable=delta           depth=3
#pragma HLS STREAM variable=lastDelta       depth=3
#pragma HLS STREAM variable=hedge           depth=3
#pragma HLS STREAM variable=money_out       depth=2
           
            DT maturity_minus_t[MAX_STEPS];

#pragma HLS dataflow
            computeMaturity(paths, steps, maturity, dt, maturity_minus_t);
            readPaths(paths, steps, pathStrmIn, s1);
            computeStock(paths, steps, underlying, s1, s1Stock, s1Stock_copy, lastStock);
            computeDelta(paths, steps, volatility, r, strike, dividendYield, call, s1Stock, maturity_minus_t, delta, lastDelta);
            computeHedge(paths, steps, exp_r_x_dt, s1Stock_copy, delta, hedge);
            reduceMoney(hedge, money_out);
            optionExpiration(paths, steps, exp_r_x_dt, strike, money_T, money_out, lastDelta, lastStock, priceStrmOut);
        }

        void computeMaturity(ap_uint<16> paths, ap_uint<16> steps, DT maturity, DT dt, DT *maturity_minus_t) {
maturity_timestep_loop:
            for (unsigned int step = 0; step < steps; step++) {
                maturity_minus_t[step] = maturity - step*dt;
            }
        }

        void readPaths(ap_uint<16> paths, ap_uint<16> steps, hls::stream<DT>& pathStrmIn, hls::stream<DT>& s1) {
read_path_loop:
            for (int i = 0; i < paths; i++) {
read_timestep_loop:
                for (int step = 0; step < steps; step++) {
                    DT logS = pathStrmIn.read();
                    DT s1_temp = qfi::FPExp(logS);
                    s1.write(s1_temp);
                }
            }
        }

        void computeStock(ap_uint<16> paths, ap_uint<16> steps, DT underlying, hls::stream<DT>& s1, hls::stream<DT>& stock_out, hls::stream<DT>& stock_out_copy, hls::stream<DT>& lastStock) {
stock_path_loop:
            for (unsigned int i = 0; i < paths; i++) {
                DT stock_temp = underlying;
                stock_out.write(stock_temp);
                DT stock_out_temp;
stock_timestep_loop:
                for (int step = 0; step < steps; step++) {
                    DT s1_temp = s1.read();
                    stock_out_temp = qfi::FPTwoMul(stock_temp, s1_temp); // 100.0 + 0.01 * (step+1)
                    stock_out.write(stock_out_temp);
                    stock_out_copy.write(stock_out_temp);
                    stock_temp = stock_out_temp;
                }
                lastStock.write(stock_out_temp);
            }
        }

        void computeDelta(ap_uint<16> paths, ap_uint<16> steps, DT volatility, DT r, DT strike, DT dividendYield, unsigned int call, hls::stream<DT>& stock_stream, DT *maturity, hls::stream<DT>& delta_out, hls::stream<DT>& lastDelta) {
delta_path_loop:
            for (unsigned int i = 0; i < paths; i++) { 
                DT delta_temp;
delta_timestep_loop:
                for (unsigned int step = 0; step < steps; step++) {
                    DT stock_temp = stock_stream.read();
                    cfBSMEngineDeltaSpot<DT>(stock_temp, volatility, r, maturity[step], strike, dividendYield, call, &delta_temp);
                    delta_out.write(delta_temp);
                }
                DT discard = stock_stream.read(); // only need delta for up to steps-1 (aka delta[path][steps-1] == stockAmount[path][steps])

                lastDelta.write(delta_temp);
            }
        }

        void computeHedge(ap_uint<16> paths, ap_uint<16> steps, DT exp_r_x_dt, hls::stream<DT>& stock, hls::stream<DT>& delta, hls::stream<DT>& hedge) {
hedge_path_loop:
            for (unsigned int i = 0; i < paths; i++) {
                DT delta_p0 = delta.read();
hedge_timestep_loop:
                for (unsigned int step = 0; step < steps-1; step++) {
#pragma HLS pipeline
                    DT stock_temp = stock.read();
                    DT delta_p1 = delta.read();
                    DT hedge_temp = (delta_p1 - delta_p0) * stock_temp * pow(exp_r_x_dt, (steps)-(step+1));
                    //printf("hedge: %f, delta_p0: %.10f, delta_p1: %.10f, stock_temp: %f, pow: %.10f\n", hedge_temp, delta_p0, delta_p1, stock_temp, pow(exp_r_x_dt, steps-(step+1)));
                    hedge.write(hedge_temp);
                    delta_p0 = delta_p1;
                }
                DT discardLastStock = stock.read(); // we do not delta-hedge for the very last stock
            }
        }

        void reduceMoney(hls::stream<DT>& hedge, hls::stream<DT>& hedgeCompoundInterestSum) {
reduce_path_loop:
            for (unsigned int path = 0; path < MAX_PATHS; path++) {
                DT sum = 0.0;
reduce_timestep_loop:
                for (unsigned int step = 0; step < MAX_STEPS-1; step++) {
                     sum += hedge.read();
                }
                hedgeCompoundInterestSum.write(sum);
            }
        }

        void optionExpiration(ap_uint<16> paths, ap_uint<16> steps, DT exp_r_x_dt, DT strike, DT money_T, hls::stream<DT>& money_account, hls::stream<DT>& lastDelta, hls::stream<DT>& lastStock, hls::stream<DT>& priceStrmOut) {
expiration_path_loop:
            for (unsigned int i = 0; i < paths; i++) {
                /*************************/
                /*** option expiration ***/
                /*************************/
                DT final_stock = lastStock.read();
                DT final_stockAmount = lastDelta.read();

                // note: money_account[] and money_T already compounded
                DT money_temp = money_account.read();
                DT final_money_account = money_T - money_temp; 
                
                // the hedger delivers the option payoff to the option holder
                DT optionPayoff = MAX(0, final_stock-strike);
                final_money_account -= optionPayoff;

                // and unwinds the hedge selling his stock position
                final_money_account += final_stockAmount*final_stock;

                // final Profit&Loss
                priceStrmOut.write(final_money_account);
            }
        } 

        void Pricing(ap_uint<16> steps,
                    ap_uint<16> paths,
                    hls::stream<DT> pathStrmIn[InN],
                    hls::stream<DT> priceStrmOut[OutN]) {
inn_loop:
                for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
                    PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
                }
        }
        }; // ReplicationPathPricer

        /**
         * @brief European Option Pricing Engine using Monte Carlo Method. This
         * implementation uses Black-Scholes valuation model with a custom 
         * PathPricer to compute the replication error of the discrete hedging 
         * strategy (re-hedging throughout option life at evenly sized steps)
         *
         * @tparam DT supported data type including DT and float data type, which
         * decides the precision of result, default DT-precision data type.
         * @tparam UN number of Monte Carlo Module in parallel, which affects the
         * latency and resources utilization, default 10.
         * @tparam Antithetic antithetic is used  for variance reduction, default this
         * feature is disabled.
         * @param underlying intial value of underlying asset at time 0.
         * @param volatility fixed volatility of underlying asset.
         * @param dividendYield the constant dividend rate for continuous dividends.
         * @param riskFreeRate risk-free interest rate.
         * @param timeLength the time length of contract from start to end.
         * @param strike the strike price also known as exericse price, which is settled
         * in the contract.
         * @param optionType option type. 1: put option, 0: call option.
         * @param seed array to store the inital seed for each RNG.
         * @param output output array.
         * @param requiredTolerance the tolerance required. If requiredSamples is not
         * set, when reaching the required tolerance, simulation will stop, default
         * 0.02.
         * @param requiredSamples the samples number required. When reaching the
         * required number, simulation will stop, default 1024.
         * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
         * time, default 100.
         * @param maxSamples the maximum sample number. When reaching it, the simulation
         * will stop, default 2,147,483,648.
         */
        template <typename DT = double, int UN = 10, bool Antithetic = false>
            void MCEuropeanEngine(DT underlying,
                    DT volatility,
                    DT dividendYield,
                    DT riskFreeRate, // model parameter
                    DT timeLength,
                    DT strike,
                    bool optionType, // option parameter
                    ap_uint<32>* seed,
                    DT* output,
                    DT requiredTolerance = 0.02,
                    unsigned int requiredSamples = 1024,
                    unsigned int timeSteps = 100,
                    unsigned int maxSamples = MAX_SAMPLE) {
                
                // number of samples per simulation
                const static int SN = MAX_PATHS; //1024; 

                // number of variate
                const static int VN = 1;

                // Step first or sample first for each simulation
                const static bool SF = true;

                // option style
                const qf::enums::OptionStyle sty = qf::enums::European;

                // RNG alias name
                typedef qf::MT19937IcnRng<DT> RNG;

                // Black-Scholes model
                qf::BSModel<DT> BSInst;

                // path generator instance
                qfi::BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

                // path pricer instance
                ReplicationPathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

                // RNG sequence instance
                qfi::RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

                // pre-process for "cold" logic.
                DT dt = timeLength / timeSteps;
                DT f_1 = qfi::FPTwoMul(riskFreeRate, timeLength);
                DT discount = qfi::FPExp(-f_1);

                BSInst.riskFreeRate = riskFreeRate;
                BSInst.dividendYield = dividendYield;
                BSInst.volatility = volatility;
                //
                BSInst.variance(dt);
                BSInst.stdDeviation();
                BSInst.updateDrift(dt);

                // configure the path generator and path pricer
init_pathpri_pathgen_loop:
                for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
                    // Path pricer
                    pathPriInst[i][0].optionType = optionType;
                    pathPriInst[i][0].strike = strike;
                    pathPriInst[i][0].underlying = underlying;
                    pathPriInst[i][0].discount = discount;
                    pathPriInst[i][0].maturity = timeLength;
                    pathPriInst[i][0].volatility = volatility;
                    pathPriInst[i][0].dividendYield = dividendYield;
                    pathPriInst[i][0].r = riskFreeRate;
                    // Path gen
                    pathGenInst[i][0].BSInst = BSInst;
                    // RNGSequnce
                    rngSeqInst[i][0].seed[0] = seed[i];
                }

                // call monte carlo simulation
                DT replicationError = qf::mcSimulation<DT, RNG, qfi::BSPathGenerator<DT, SF, SN, Antithetic>, ReplicationPathPricer<sty, DT, SF, SN, Antithetic>, qfi::RNGSequence<DT, RNG>, UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

                // output the price of option
                output[0] = replicationError;
            } // MCEuropeanEngine
} // rep
