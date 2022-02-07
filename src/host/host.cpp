#include <host_support.h>
#include <iomanip>
#include "host.hpp"
#include <malloc.h>

#define PAGESIZE 4096
#define HOST_DEBUG 0

static float getTimeOfComponent(cl::Event&);
static void init_device(char*, double, unsigned int, unsigned int, double, double, unsigned int, unsigned int, double*);
static void execute_on_device(cl::Event&,cl::Event&);
static float getTimeOfComponent(cl::Event&);

cl::CommandQueue * command_queue;
cl::Context * context;
cl::Program * program;
cl::Kernel * krnl_scenario;
cl::Buffer *buffer_results; // Buffers to transfer to and from the device

int main(int argc, char** argv)
{
    cl::Event kernelExecutionEvent, copyOffEvent;

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // kernel args
    double maturity = 1.0/12.0; // 1 month (1.0 is 12 months)
    unsigned int strike = 100; // price to pay at execution
    unsigned int underlying = 100; // initial stock price
    double volatility = 0.20; // 20%
    double riskFreeRate = 0.05; // 5%
    double dividendYield = 0.00; // for simplicity, assume stock doesn't pay dividend
    bool optionType = 0; // 1 for put, 0 for call
    unsigned int seed = 3;
    double requiredTolerance = 0.01;

    unsigned int scenarios = 50000; // requiredSamples, simulation stops when N=scenarios reached
    unsigned int hedgesNum = 84; // timeSteps, number of timesteps per scenario

    double *results=(double*) memalign(PAGESIZE, sizeof(double));  

    std::cout << "------------- Kernel Args ---------------" << std::endl; 

    std::cout << std::setw(18) << "seed"                << std::setw(18) << seed << std::endl;
    std::cout << std::setw(18) << "underlying"          << std::setw(18) << underlying << std::endl;
    std::cout << std::setw(18) << "volatility"          << std::setw(18) << volatility << std::endl;
    std::cout << std::setw(18) << "dividendYield"       << std::setw(18) << dividendYield << std::endl;
    std::cout << std::setw(18) << "riskFreeRate"        << std::setw(18) << riskFreeRate << std::endl;
    std::cout << std::setw(18) << "timeLength"          << std::setw(18) << maturity << std::endl;
    std::cout << std::setw(18) <<  "strike"             << std::setw(18) << strike << std::endl;
    std::cout << std::setw(18) << "optionType"          << std::setw(18) << optionType << std::endl;
    std::cout << std::setw(18) << "requiredTolerance"   << std::setw(18) << requiredTolerance << std::endl;
    std::cout << std::setw(18) << "requiredSamples"     << std::setw(18) << scenarios << std::endl;
    std::cout << std::setw(18) << "timeSteps"           << std::setw(18) << hedgesNum << std::endl;

    std::cout << "-----------------------------------------" << std::endl;

    init_device(argv[1], maturity, strike, underlying, volatility, riskFreeRate, scenarios, hedgesNum, results);
    execute_on_device(kernelExecutionEvent, copyOffEvent);

    // Compare the results of the Device to the simulation
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Kernel result" << std::endl;
    printf("P&L: %f\n", *results);
    std::cout << "-----------------------------------------" << std::endl;


    float copyOnTime=0;
    float kernelTime=getTimeOfComponent(kernelExecutionEvent);
    float copyOffTime=getTimeOfComponent(copyOffEvent);

    printf("Execution complete, total runtime : %.3f ms, (%.3f ms xfer on, %.3f ms execute, %.3f ms xfer off)\n", copyOnTime+kernelTime+copyOffTime, copyOnTime, kernelTime, copyOffTime);

    delete buffer_results;
    delete krnl_scenario;
    delete command_queue;
    delete context;
    delete program;

    return EXIT_SUCCESS;
}

static float getTimeOfComponent(cl::Event & event) {
    cl_ulong tstart, tstop;

    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &tstart);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &tstop);
    return (tstop-tstart)/1.E6;
}

/**
 * Performs execution on the device by transfering input data, running the kernel, and copying result data back
 * We use OpenCL events here to set the dependencies properly
 */
static void execute_on_device(cl::Event & kernelExecutionEvent, cl::Event & copyOffEvent) {
    cl_int err;

    OCL_CHECK(err, err = command_queue->enqueueTask(*krnl_scenario, nullptr, &kernelExecutionEvent));
    
    std::vector<cl::Event> data_transfer_wait_events;
    data_transfer_wait_events.push_back(kernelExecutionEvent);
    
    OCL_CHECK(err, err = command_queue->enqueueMigrateMemObjects({*buffer_results}, CL_MIGRATE_MEM_OBJECT_HOST, &data_transfer_wait_events, &copyOffEvent));

    OCL_CHECK(err, err = command_queue->finish());
}

/**
 * Initiates the FPGA device and sets up the OpenCL context
 */
static void init_device(char * binary_filename, double maturity, unsigned int strike, unsigned int underlying, double volatility, double riskFreeRate, unsigned int scenarios, unsigned int hedgesNum, double *results) {
    cl_int err;

    std::vector<cl::Device> devices;
    std::tie(program, context, devices)=initialiseDevice("Xilinx", "u280", binary_filename);
    
    // Create the command queue (and enable profiling so we can get performance data back)
    OCL_CHECK(err, command_queue=new cl::CommandQueue(*context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err));
    // Create a handle to the sum kernel
    OCL_CHECK(err, krnl_scenario=new cl::Kernel(*program, "krnl_scenario", &err));
    
    OCL_CHECK(err, buffer_results=new cl::Buffer(*context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double), results, &err));

    OCL_CHECK(err, err = krnl_scenario->setArg(0, maturity));
    OCL_CHECK(err, err = krnl_scenario->setArg(1, strike));
    OCL_CHECK(err, err = krnl_scenario->setArg(2, underlying));
    OCL_CHECK(err, err = krnl_scenario->setArg(3, volatility));
    OCL_CHECK(err, err = krnl_scenario->setArg(4, riskFreeRate));
    OCL_CHECK(err, err = krnl_scenario->setArg(5, scenarios));
    OCL_CHECK(err, err = krnl_scenario->setArg(6, hedgesNum));
    OCL_CHECK(err, err = krnl_scenario->setArg(7, *buffer_results));
}
