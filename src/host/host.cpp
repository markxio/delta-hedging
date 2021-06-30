#include <iomanip>
#include "host.hpp"
#include <malloc.h>

static float getTimeOfComponent(cl::Event&);

#define PAGESIZE 4096
#define HOST_DEBUG 0

int main(int argc, char** argv)
{
    cl::Event copyOnEvent, SRNExecutionEvent, copyOffEvent;

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    unsigned fileBufSize;

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


    double *sampleInput=(double*) memalign(PAGESIZE, sizeof(double));

    unsigned int resultsNum = 1;  
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
    // OPENCL HOST CODE AREA START

    // ------------------------------------------------------------------------------------
    // Step 1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
    //	   Search for Platform: Xilinx
    // Check if the current platform matches Target_Platform_Vendor
    // ------------------------------------------------------------------------------------
    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err)); // Create context
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err)); // Create Command Queue

    // ------------------------------------------------------------------
    // Step 1: Load Binary File from disk
    // ------------------------------------------------------------------
    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err)); // Create the program object from the binary and program the FPGA device with it
    OCL_CHECK(err, cl::Kernel krnl_scenario(program,"krnl_scenario", &err)); // create kernels

    // ================================================================
    // Step 2: Setup Buffers and run Kernels
    // ================================================================
    //   o) Allocate Memory to store the results
    //   o) Create Buffers in Global Memory to store data
    // ================================================================
    // .......................................................
    // Allocate Global Memory for source_in1
    // .......................................................
    OCL_CHECK(err, cl::Buffer buffer_sampleInput(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double), sampleInput, &err));
    OCL_CHECK(err, cl::Buffer buffer_results(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double), results, &err));

    // ============================================================================
    // Step 2: Set Kernel Arguments and Run the Application
    //         o) Set Kernel Arguments
    //         o) Copy Input Data from Host to Global Memory on the device
    //         o) Submit Kernels for Execution
    //         o) Copy Results from Global Memory, device to Host
    // ============================================================================
    if (HOST_DEBUG) printf("host -- about to set kernel args\n");
    OCL_CHECK(err, err = krnl_scenario.setArg(0, maturity));
    OCL_CHECK(err, err = krnl_scenario.setArg(1, strike));
    OCL_CHECK(err, err = krnl_scenario.setArg(2, underlying));
    OCL_CHECK(err, err = krnl_scenario.setArg(3, volatility));
    OCL_CHECK(err, err = krnl_scenario.setArg(4, riskFreeRate));
    OCL_CHECK(err, err = krnl_scenario.setArg(5, scenarios));
    OCL_CHECK(err, err = krnl_scenario.setArg(6, hedgesNum));
    OCL_CHECK(err, err = krnl_scenario.setArg(7, buffer_sampleInput));
    OCL_CHECK(err, err = krnl_scenario.setArg(8, buffer_results));
    if (HOST_DEBUG) printf("host -- now cp over buffers\n");
    // ------------------------------------------------------
    // Step 2: Copy Input data from Host to Global Memory on the device
    // ------------------------------------------------------
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_sampleInput},0/* 0 means from host*/, nullptr, &copyOnEvent));
    if (HOST_DEBUG) printf("host -- wait\n");
    copyOnEvent.wait();
    // ----------------------------------------
    // Step 2: Submit Kernels for Execution
    // ----------------------------------------
    if (HOST_DEBUG) printf("host -- enqueueTask\n");
    OCL_CHECK(err, err = q.enqueueTask(krnl_scenario, nullptr, &SRNExecutionEvent));
    if (HOST_DEBUG) printf("host -- wait\n");
    SRNExecutionEvent.wait();
    if (HOST_DEBUG) printf("host -- ExecutionEvent received\n");
    // --------------------------------------------------
    // Step 2: Copy Results from Device Global Memory to Host
    // --------------------------------------------------
    if (HOST_DEBUG) printf("host -- copy data back form device to host\n");
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_results},CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &copyOffEvent));
    if (HOST_DEBUG) printf("host -- wait\n");
    copyOffEvent.wait();

    q.finish();

    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Kernel result" << std::endl;
    printf("P&L: %f\n", *results);
    std::cout << "-----------------------------------------" << std::endl;


    float copyOnTime=getTimeOfComponent(copyOnEvent);
    float kernelTime=getTimeOfComponent(SRNExecutionEvent);
    float copyOffTime=getTimeOfComponent(copyOffEvent);

    printf("Execution complete, total runtime : %.3f ms, (%.3f ms xfer on, %.3f ms execute, %.3f ms xfer off)\n", copyOnTime+kernelTime+copyOffTime, copyOnTime, kernelTime, copyOffTime);

    // ============================================================================
    // Step 3: Release Allocated Resources
    // ============================================================================
    delete[] fileBuf;

    return EXIT_SUCCESS;
}

static float getTimeOfComponent(cl::Event & event) {
    cl_ulong tstart, tstop;

    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &tstart);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &tstop);
    return (tstop-tstart)/1.E6;
}


