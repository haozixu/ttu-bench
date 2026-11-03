#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define CHECK(x) do { CUresult _e = (x); if (_e) { const char* s=""; cuGetErrorString(_e,&s); printf("CUDA err %s @%d\n", s, __LINE__); exit(1);} } while (0)
#define OCHK(x) do { OptixResult _r = (x); if (_r) { printf("Optix err %d @%d: %s\n", (int)_r, __LINE__, optixGetErrorString(_r)); exit(1);} } while (0)

struct Params { OptixTraversableHandle gas; unsigned int* out; };

static void readPTX(const char* path, std::string& out) {
    FILE* f = fopen(path, "rb"); if (!f) { printf("no ptx %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(n); size_t rd = fread(&out[0], 1, n, f); (void)rd; fclose(f);
}

int main(int argc, char** argv) {
    const char* ptxPath = (argc > 1) ? argv[1] : "optix_min.ptx";

    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK(cuCtxSetCurrent(ctx));
    CUstream stream; CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    OCHK(optixInit());
    OptixDeviceContextOptions o = {};
    char log[2048]; size_t logSz = sizeof(log);
    OptixDeviceContext oxctx;
    OCHK(optixDeviceContextCreate(ctx, &o, &oxctx));
    OCHK(optixDeviceContextSetLogCallback(oxctx,
        [](unsigned level, const char* tag, const char* msg, void*) {
            printf("[ox %u %s] %s\n", level, tag, msg);
        }, nullptr, 0));

    float verts[9] = {1,0,0, 0,2,0, 0,0,3};
    CUdeviceptr dverts;
    CHECK(cuMemAlloc(&dverts, sizeof(verts)));
    CHECK(cuMemcpyHtoD(dverts, verts, sizeof(verts)));
    uint32_t triFlag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexBuffers = &dverts;
    bi.triangleArray.numVertices = 3;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float)*3;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    bi.triangleArray.numSbtRecords = 1;
    bi.triangleArray.flags = &triFlag;
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_NONE;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes asz;
    OCHK(optixAccelComputeMemoryUsage(oxctx, &ao, &bi, 1, &asz));
    CUdeviceptr dOut, dTmp;
    CHECK(cuMemAlloc(&dOut, asz.outputSizeInBytes));
    CHECK(cuMemAlloc(&dTmp, asz.tempSizeInBytes));
    OptixTraversableHandle gas;
    OCHK(optixAccelBuild(oxctx, stream, &ao, &bi, 1, dTmp, asz.tempSizeInBytes,
                         dOut, asz.outputSizeInBytes, &gas, nullptr, 0));
    printf("GAS: traversable=0x%llx output=0x%llx (%zu bytes)\n",
           (unsigned long long)gas, (unsigned long long)dOut, (size_t)asz.outputSizeInBytes);

    std::string ptx; readPTX(ptxPath, ptx);
    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues = 1;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    OptixModule mod;
    logSz = sizeof(log);
    OCHK(optixModuleCreate(oxctx, &mco, &pco, ptx.c_str(), ptx.size(), log, &logSz, &mod));
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc pgd[3] = {};
    pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;   pgd[0].raygen.module = mod; pgd[0].raygen.entryFunctionName = "__raygen__rg";
    pgd[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;     pgd[1].miss.module = mod;   pgd[1].miss.entryFunctionName = "__miss__ms";
    pgd[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; pgd[2].hitgroup.moduleCH = mod; pgd[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[3];
    OCHK(optixProgramGroupCreate(oxctx, pgd, 3, &pgo, log, &logSz, pg));
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixPipeline pipeline;
    OCHK(optixPipelineCreate(oxctx, &pco, &plo, pg, 3, log, &logSz, &pipeline));
    OCHK(optixPipelineSetStackSize(pipeline, 0, 0, 1024, 1));

    struct Rec { unsigned char header[OPTIX_SBT_RECORD_HEADER_SIZE]; void* data; unsigned char pad[8]; };
    Rec rg, ms, hg;
    OCHK(optixSbtRecordPackHeader(pg[0], &rg));
    OCHK(optixSbtRecordPackHeader(pg[1], &ms));
    OCHK(optixSbtRecordPackHeader(pg[2], &hg));
    CUdeviceptr drg, dms, dhg;
    CHECK(cuMemAlloc(&drg, sizeof(Rec))); CHECK(cuMemAlloc(&dms, sizeof(Rec)));
    CHECK(cuMemAlloc(&dhg, sizeof(Rec)));
    CHECK(cuMemcpyHtoD(drg, &rg, sizeof(Rec)));
    CHECK(cuMemcpyHtoD(dms, &ms, sizeof(Rec)));
    CHECK(cuMemcpyHtoD(dhg, &hg, sizeof(Rec)));

    Params hostp; hostp.gas = gas;
    CHECK(cuMemAlloc((CUdeviceptr*)&hostp.out, 64 * sizeof(unsigned int)));
    CHECK(cuMemsetD32((CUdeviceptr)hostp.out, 0, 64));
    CUdeviceptr dparams;
    CHECK(cuMemAlloc(&dparams, sizeof(Params)));
    CHECK(cuMemcpyHtoD(dparams, &hostp, sizeof(Params)));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = drg;
    sbt.missRecordBase = dms; sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = dhg; sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;

    OCHK(optixLaunch(pipeline, stream, dparams, sizeof(Params), &sbt, 1, 1, 1));
    CHECK(cuStreamSynchronize(stream));
    unsigned int res[2] = {0,0};
    CHECK(cuMemcpyDtoH(res, (CUdeviceptr)hostp.out, sizeof(res)));
    printf("optix ray result: out[0]=0x%08x (%s, primID=%u)\n",
           res[0], (res[0] & 0xFF000000u) == 0x10000000u ? "HIT" : "MISS", res[0] & 0xFFFF);

    // dump OptiX-generated AS image
    {
        unsigned char* asdump = (unsigned char*)malloc(asz.outputSizeInBytes);
        if (cuMemcpyDtoH(asdump, dOut, asz.outputSizeInBytes) == CUDA_SUCCESS) {
            FILE* f = fopen("optix_as.bin", "wb");
            fwrite(asdump, 1, asz.outputSizeInBytes, f); fclose(f);
            printf("optix AS dumped: %zu bytes\n", (size_t)asz.outputSizeInBytes);
        }
        free(asdump);
    }

    return 0;
}
