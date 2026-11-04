#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Ray { float ox, oy, oz, tmin, dx, dy, dz, tmax; };
struct Hit { unsigned primitive, t, u, v, front_face, instance_id; };
struct Params { OptixTraversableHandle gas; CUdeviceptr rays, hits; unsigned ray_flags, has_instances; };
struct SceneInfo { uint64_t base, size, handle, gas_base, gas_size, gas_handle; };
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    unsigned char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
static thread_local std::string last_error;

static void check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message ? message : "CUDA error");
    }
}

static void check(OptixResult result) {
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(optixGetErrorString(result));
}

struct Scene {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup programs[3] = {};
    OptixShaderBindingTable sbt = {};
    OptixTraversableHandle gas = 0;
    CUdeviceptr base = 0;
    size_t size = 0;
    double build_seconds = 0;
    bool has_instances = false;
    std::vector<CUdeviceptr> allocations;

    CUdeviceptr allocate(size_t bytes, const void* source = nullptr) {
        CUdeviceptr pointer;
        check(cuMemAlloc(&pointer, bytes));
        allocations.push_back(pointer);
        if (source) check(cuMemcpyHtoD(pointer, source, bytes));
        return pointer;
    }

    void build(const OptixBuildInput& input) {
        auto started = std::chrono::steady_clock::now();
        OptixAccelBuildOptions options = {};
        options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes;
        check(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes));
        size = sizes.outputSizeInBytes;
        base = allocate(size);
        CUdeviceptr temporary = allocate(sizes.tempSizeInBytes);
        check(optixAccelBuild(context, nullptr, &options, &input, 1,
                              temporary, sizes.tempSizeInBytes, base, size, &gas, nullptr, 0));
        check(cuCtxSynchronize());
        build_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }

    ~Scene() {
        if (pipeline) optixPipelineDestroy(pipeline);
        for (auto program : programs) if (program) optixProgramGroupDestroy(program);
        if (module) optixModuleDestroy(module);
        for (auto pointer : allocations) cuMemFree(pointer);
        if (context) optixDeviceContextDestroy(context);
    }
};

extern "C" const char* scene_error() { return last_error.c_str(); }

extern "C" int scene_create(const float* vertices, unsigned triangles, const char* ptx,
                              unsigned geometry_flags, const float* transforms,
                              const unsigned* instance_ids, unsigned instance_count,
                              SceneInfo* info, void** result) {
    Scene* scene = new Scene;
    try {
        CUcontext current;
        check(cuCtxGetCurrent(&current));
        if (!current || !triangles) throw std::runtime_error("requires current CUDA context and nonempty geometry");
        check(optixInit());
        OptixDeviceContextOptions options = {};
        check(optixDeviceContextCreate(current, &options, &scene->context));
        CUdeviceptr device_vertices = scene->allocate(triangles * 9 * sizeof(float), vertices);
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexBuffers = &device_vertices;
        input.triangleArray.numVertices = triangles * 3;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
        input.triangleArray.numSbtRecords = 1;
        input.triangleArray.flags = &geometry_flags;
        scene->build(input);
        *info = {scene->base, scene->size, scene->gas, scene->base, scene->size, scene->gas};
        if (instance_count) {
            std::vector<OptixInstance> instances(instance_count);
            for (unsigned index = 0; index < instance_count; ++index) {
                auto& instance = instances[index];
                std::memcpy(instance.transform, transforms + index * 12, 12 * sizeof(float));
                instance.instanceId = instance_ids[index];
                instance.visibilityMask = 255;
                instance.traversableHandle = scene->gas;
            }
            OptixBuildInput top = {};
            top.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            top.instanceArray.instances = scene->allocate(instances.size() * sizeof(OptixInstance), instances.data());
            top.instanceArray.numInstances = instance_count;
            scene->build(top);
            scene->has_instances = true;
            info->base = scene->base;
            info->size = scene->size;
            info->handle = scene->gas;
        }

        OptixModuleCompileOptions module_options = {};
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        OptixPipelineCompileOptions pipeline_options = {};
        pipeline_options.traversableGraphFlags = instance_count
            ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
            : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_options.numPayloadValues = 6;
        pipeline_options.numAttributeValues = 2;
        pipeline_options.pipelineLaunchParamsVariableName = "params";
        pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        char log[4096];
        size_t log_size = sizeof(log);
        std::string code(ptx);
        check(optixModuleCreate(scene->context, &module_options, &pipeline_options,
                                code.c_str(), code.size(), log, &log_size, &scene->module));
        OptixProgramGroupDesc descriptions[3] = {};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen.module = scene->module;
        descriptions[0].raygen.entryFunctionName = "__raygen__scene";
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss.module = scene->module;
        descriptions[1].miss.entryFunctionName = "__miss__scene";
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleCH = scene->module;
        descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__scene";
        OptixProgramGroupOptions program_options = {};
        log_size = sizeof(log);
        check(optixProgramGroupCreate(scene->context, descriptions, 3, &program_options,
                                      log, &log_size, scene->programs));
        OptixPipelineLinkOptions link = {};
        link.maxTraceDepth = 1;
        log_size = sizeof(log);
        check(optixPipelineCreate(scene->context, &pipeline_options, &link,
                                  scene->programs, 3, log, &log_size, &scene->pipeline));
        check(optixPipelineSetStackSize(scene->pipeline, 0, 0, 2048, instance_count ? 2 : 1));
        Record records[3] = {};
        for (unsigned index = 0; index < 3; ++index)
            check(optixSbtRecordPackHeader(scene->programs[index], &records[index]));
        scene->sbt.raygenRecord = scene->allocate(sizeof(Record), &records[0]);
        scene->sbt.missRecordBase = scene->allocate(sizeof(Record), &records[1]);
        scene->sbt.missRecordStrideInBytes = sizeof(Record);
        scene->sbt.missRecordCount = 1;
        scene->sbt.hitgroupRecordBase = scene->allocate(sizeof(Record), &records[2]);
        scene->sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        scene->sbt.hitgroupRecordCount = 1;
        *result = scene;
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        delete scene;
        return -1;
    }
}

extern "C" int scene_trace(void* object, const Ray* rays, unsigned count, unsigned ray_flags, Hit* hits) {
    auto scene = static_cast<Scene*>(object);
    CUdeviceptr device_rays = 0, device_hits = 0, device_params = 0;
    try {
        check(cuMemAlloc(&device_rays, count * sizeof(Ray)));
        check(cuMemAlloc(&device_hits, count * sizeof(Hit)));
        check(cuMemcpyHtoD(device_rays, rays, count * sizeof(Ray)));
        Params params = {scene->gas, device_rays, device_hits, ray_flags, unsigned(scene->has_instances)};
        check(cuMemAlloc(&device_params, sizeof(params)));
        check(cuMemcpyHtoD(device_params, &params, sizeof(params)));
        check(optixLaunch(scene->pipeline, nullptr, device_params, sizeof(params),
                          &scene->sbt, count, 1, 1));
        check(cuCtxSynchronize());
        check(cuMemcpyDtoH(hits, device_hits, count * sizeof(Hit)));
        cuMemFree(device_rays); cuMemFree(device_hits); cuMemFree(device_params);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        if (device_rays) cuMemFree(device_rays);
        if (device_hits) cuMemFree(device_hits);
        if (device_params) cuMemFree(device_params);
        return -1;
    }
}

extern "C" void scene_destroy(void* object) { delete static_cast<Scene*>(object); }

extern "C" double scene_build_seconds(void* object) { return static_cast<Scene*>(object)->build_seconds; }

extern "C" int scene_launch_device(void* object, uint64_t device_params, unsigned count) {
    auto scene = static_cast<Scene*>(object);
    try {
        if (!count || !device_params) throw std::runtime_error("requires nonempty device parameters");
        check(optixLaunch(scene->pipeline, nullptr, device_params, sizeof(Params),
                          &scene->sbt, count, 1, 1));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
