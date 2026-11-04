#include <cuda.h>
#include <cstdint>

using OptixLaunch = int (*)(void*, uint64_t, unsigned);
struct Launch {
    CUfunction camera, trace, shade;
    OptixLaunch optix;
    void* scene;
    CUdeviceptr params, rays, hits, triangles, colors;
    unsigned count, width, height, mode, block;
};

struct Reflection {
    Launch primary, secondary;
    CUfunction generate, compose;
    unsigned floor_start;
    float floor_y;
};

static CUresult launch(Launch* job, unsigned stage) {
    unsigned blocks = (job->count+job->block-1)/job->block;
    if (stage == 1 || stage == 3) {
        void* args[] = {&job->rays, &job->width, &job->height};
        CUresult status = cuLaunchKernel(job->camera, blocks, 1, 1, job->block, 1, 1, 0, nullptr, args, nullptr);
        if (status != CUDA_SUCCESS) return status;
    }
    if (stage == 0 || stage == 3) {
        if (job->optix) {
            if (job->optix(job->scene, job->params, job->count)) return CUDA_ERROR_UNKNOWN;
        } else {
            void* args[] = {&job->params};
            CUresult status = cuLaunchKernel(job->trace, blocks, 1, 1, job->block, 1, 1, 0, nullptr, args, nullptr);
            if (status != CUDA_SUCCESS) return status;
        }
    }
    if (stage == 2 || stage == 3) {
        void* args[] = {&job->hits, &job->triangles, &job->colors, &job->count, &job->mode};
        return cuLaunchKernel(job->shade, blocks, 1, 1, job->block, 1, 1, 0, nullptr, args, nullptr);
    }
    return CUDA_SUCCESS;
}

extern "C" int renderer_launch(Launch* job, unsigned stage) { return launch(job, stage); }

static CUresult reflect(Reflection* job, unsigned stage) {
    CUresult status = CUDA_SUCCESS;
    Launch* primary = &job->primary;
    Launch* secondary = &job->secondary;
    unsigned blocks = (primary->count+primary->block-1)/primary->block;
    if (stage == 3) {
        status = launch(primary,1);
        if (status == CUDA_SUCCESS) status = launch(primary,0);
    }
    if (status == CUDA_SUCCESS && (stage == 0 || stage == 3)) {
        void* args[] = {&primary->rays,&primary->hits,&secondary->rays,&primary->count,&job->floor_start,&job->floor_y};
        status = cuLaunchKernel(job->generate,blocks,1,1,primary->block,1,1,0,nullptr,args,nullptr);
    }
    if (status == CUDA_SUCCESS && (stage == 1 || stage == 3)) status = launch(secondary,0);
    if (status == CUDA_SUCCESS && (stage == 2 || stage == 3)) {
        void* args[] = {&primary->hits,&secondary->hits,&primary->triangles,&primary->colors,&primary->count,&job->floor_start};
        status = cuLaunchKernel(job->compose,blocks,1,1,primary->block,1,1,0,nullptr,args,nullptr);
    }
    return status;
}

extern "C" int renderer_reflect_launch(Reflection* job, unsigned stage) { return reflect(job,stage); }

extern "C" int renderer_reflect_time(Reflection* job, unsigned stage, unsigned iterations, float* milliseconds) {
    CUevent start = nullptr, end = nullptr;
    CUresult status = cuEventCreate(&start,0);
    if (status == CUDA_SUCCESS) status = cuEventCreate(&end,0);
    if (status == CUDA_SUCCESS) status = cuEventRecord(start,nullptr);
    for (unsigned iteration = 0; status == CUDA_SUCCESS && iteration < iterations; ++iteration)
        status = reflect(job,stage);
    if (status == CUDA_SUCCESS) status = cuEventRecord(end,nullptr);
    if (status == CUDA_SUCCESS) status = cuEventSynchronize(end);
    if (status == CUDA_SUCCESS) status = cuEventElapsedTime(milliseconds,start,end);
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
    return status;
}

extern "C" int renderer_time(Launch* job, unsigned stage, unsigned iterations, float* milliseconds) {
    CUevent start = nullptr, end = nullptr;
    CUresult status = cuEventCreate(&start, 0);
    if (status == CUDA_SUCCESS) status = cuEventCreate(&end, 0);
    if (status == CUDA_SUCCESS) status = cuEventRecord(start, nullptr);
    for (unsigned iteration = 0; status == CUDA_SUCCESS && iteration < iterations; ++iteration)
        status = launch(job, stage);
    if (status == CUDA_SUCCESS) status = cuEventRecord(end, nullptr);
    if (status == CUDA_SUCCESS) status = cuEventSynchronize(end);
    if (status == CUDA_SUCCESS) status = cuEventElapsedTime(milliseconds, start, end);
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
    return status;
}
