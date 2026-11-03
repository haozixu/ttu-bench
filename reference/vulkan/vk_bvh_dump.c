// Reference generator for the saved two-triangle Vulkan BLAS images.
//
// Triangles use distinctive float bit patterns:
//   tri A: (1.0,0,0) (0,2.0,0) (0,0,3.0)     0x3F800000 0x40000000 0x40400000
//   tri B: (100,5,5) (101,5,5) (100,6,5)     0x42C80000 0x42CA0000 0x40A00000/40C00000
//
// No shaders needed: vkCmdBuildAccelerationStructuresKHR is fixed function.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <dlfcn.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, _r); exit(1); } } while (0)

static const float VERTS[] = {
    /* tri A */ 1.0f, 0.0f, 0.0f,   0.0f, 2.0f, 0.0f,   0.0f, 0.0f, 3.0f,
    /* tri B */ 100.0f, 5.0f, 5.0f, 101.0f, 5.0f, 5.0f, 100.0f, 6.0f, 5.0f,
};
#define NVERTS 6
#define VSTRIDE (3 * sizeof(float))

static unsigned long long g_asMemSize = 0;

static void *alloc_and_bind(VkDevice dev, VkPhysicalDevice pd, VkBufferUsageFlags usage,
                            VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem,
                            VkMemoryPropertyFlags props) {
    *buf = VK_NULL_HANDLE; *mem = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(dev, &bi, NULL, buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) { type = i; break; }
    if (type == UINT32_MAX) { fprintf(stderr, "no memory type\n"); exit(1); }
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size, .memoryTypeIndex = type};
    CHECK(vkAllocateMemory(dev, &ai, NULL, mem));
    CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
    return NULL;
}

int main(int argc, char **argv) {
    // instance + physical device
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app};
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &npd, NULL));
    VkPhysicalDevice *pds = malloc(npd * sizeof(*pds));
    CHECK(vkEnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    int want = (argc > 1) ? atoi(argv[1]) : -1;
    for (uint32_t i = 0; i < npd; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pds[i], &props);
        printf("dev %u: %s vendor=0x%x\n", i, props.deviceName, props.vendorID);
        if (props.vendorID == 0x10DE && ((want >= 0 && (int)i == want) || (want < 0 && !pd)))
            pd = pds[i];
    }
    if (!pd) { fprintf(stderr, "no NVIDIA device\n"); return 1; }

    // check RT extension + queue family
    uint32_t next = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &next, NULL);
    VkExtensionProperties *exts = malloc(next * sizeof(*exts));
    vkEnumerateDeviceExtensionProperties(pd, NULL, &next, exts);
    int has_as = 0, has_bda = 0, has_dho = 0;
    for (uint32_t i = 0; i < next; i++) {
        if (!strcmp(exts[i].extensionName, "VK_KHR_acceleration_structure")) has_as = 1;
        if (!strcmp(exts[i].extensionName, "VK_KHR_buffer_device_address")) has_bda = 1;
        if (!strcmp(exts[i].extensionName, "VK_KHR_deferred_host_operations")) has_dho = 1;
    }
    printf("AS=%d BDA=%d DHO=%d\n", has_as, has_bda, has_dho);
    if (!has_as) { fprintf(stderr, "no VK_KHR_acceleration_structure\n"); return 1; }

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = malloc(nq * sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qf = i; break; }
    printf("queue family %u\n", qf);

    // device with features enabled
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf =
        {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
         .accelerationStructure = VK_TRUE};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda =
        {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
         .bufferDeviceAddress = VK_TRUE};
    asf.pNext = &bda;
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                    .pNext = &asf};
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
    const char *dev_exts[] = {"VK_KHR_acceleration_structure",
                              "VK_KHR_deferred_host_operations",
                              "VK_KHR_buffer_device_address",
                              "VK_KHR_external_memory_fd"};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 4, .ppEnabledExtensionNames = dev_exts};
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");

    // buffers: vertices (host), AS storage (host-visible for dump), scratch
    VkBuffer vbuf, asbuf, sbuf; VkDeviceMemory vmem, asmem, smem;
    alloc_and_bind(dev, pd,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(VERTS), &vbuf, &vmem, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    {
        // allocate AS memory with export support
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                 .size = 1 << 16,
                                 .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                 .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        CHECK(vkCreateBuffer(dev, &bi, NULL, &asbuf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, asbuf, &mr);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { type = i; break; }
        VkExportMemoryAllocateInfo emi = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                   .pNext = &emi, .allocationSize = mr.size, .memoryTypeIndex = type};
        CHECK(vkAllocateMemory(dev, &ai, NULL, &asmem));
        CHECK(vkBindBufferMemory(dev, asbuf, asmem, 0));
        g_asMemSize = mr.size;
    }
    alloc_and_bind(dev, pd,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        1 << 20, &sbuf, &smem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    void *m; CHECK(vkMapMemory(dev, vmem, 0, sizeof(VERTS), 0, &m));
    memcpy(m, VERTS, sizeof(VERTS)); vkUnmapMemory(dev, vmem);

    PFN_vkGetBufferDeviceAddressKHR gBDA =
        (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(dev, "vkGetBufferDeviceAddressKHR");
    VkBufferDeviceAddressInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                    .buffer = vbuf};
    VkDeviceAddress vaddr = gBDA(dev, &bi);

    // build sizes
    VkAccelerationStructureGeometryKHR geom = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData.deviceAddress = vaddr,
            .vertexStride = VSTRIDE,
            .maxVertex = NVERTS - 1,
            .indexType = VK_INDEX_TYPE_NONE_KHR}};
    uint32_t max_prim = 2;
    VkAccelerationStructureBuildGeometryInfoKHR binfo = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .geometryCount = 1, .pGeometries = &geom};
    VkAccelerationStructureBuildSizesInfoKHR sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &binfo, &max_prim, &sizes);
    printf("AS size=%llu scratch=%llu\n", (unsigned long long)sizes.accelerationStructureSize,
           (unsigned long long)sizes.buildScratchSize);

    // create + build N identical BLASes (different VAs) for pointer-field diffing
    VkBufferDeviceAddressInfo sbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = sbuf};
    VkDeviceAddress scratch = gBDA(dev, &sbi);
    PFN_vkGetAccelerationStructureDeviceAddressKHR gASAddr =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(dev,
            "vkGetAccelerationStructureDeviceAddressKHR");
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");

    VkAccelerationStructureKHR as[3];
    VkCommandBuffer cmds[3];
    for (int n = 0; n < 3; n++) {
        VkAccelerationStructureCreateInfoKHR asi = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .buffer = asbuf, .offset = 4096 * n, .size = sizes.accelerationStructureSize};
        CHECK(vkCreateAccelerationStructureKHR(dev, &asi, NULL, &as[n]));
        VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .queueFamilyIndex = qf};
        VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
        VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                           .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                           .commandBufferCount = 1};
        CHECK(vkAllocateCommandBuffers(dev, &cai, &cmds[n]));
        VkCommandBufferBeginInfo bbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CHECK(vkBeginCommandBuffer(cmds[n], &bbi));
        binfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        binfo.dstAccelerationStructure = as[n];
        binfo.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR range = {.primitiveCount = 2};
        const VkAccelerationStructureBuildRangeInfoKHR *prange = &range;
        vkCmdBuildAccelerationStructuresKHR(cmds[n], 1, &binfo, &prange);
        CHECK(vkEndCommandBuffer(cmds[n]));
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1, .pCommandBuffers = &cmds[n]};
        CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    }
    CHECK(vkQueueWaitIdle(queue));

    CHECK(vkMapMemory(dev, asmem, 0, 1 << 16, 0, &m));
    for (int n = 0; n < 3; n++) {
        VkAccelerationStructureDeviceAddressInfoKHR aai = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = as[n]};
        printf("BLAS %d: AS device address = 0x%llx\n", n,
               (unsigned long long)gASAddr(dev, &aai));
        char fn[64]; snprintf(fn, sizeof(fn), "as_dump_%d.bin", n);
        FILE *f = fopen(fn, "wb");
        fwrite((char *)m + 4096 * n, 1, sizes.accelerationStructureSize + 64, f);
        fclose(f);
        printf("  dumped %llu bytes to %s\n",
               (unsigned long long)(sizes.accelerationStructureSize + 64), fn);
    }

    void *libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    // export AS memory as opaque fd
    PFN_vkGetMemoryFdKHR gFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    fprintf(stderr, "gFd proc = %p\n", (void*)gFd);
    uint32_t nExt = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &nExt, NULL);
    fprintf(stderr, "device has %u extensions\n", nExt);
    VkMemoryGetFdInfoKHR fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                               .memory = asmem,
                               .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    int fd = -1;
    fprintf(stderr, "step: calling vkGetMemoryFdKHR\n");
    gFd(dev, &fi, &fd);
    fprintf(stderr, "step: fd=%d allocSize=%llu\n", fd, g_asMemSize);

    // import into CUDA
    {
        int (*cuInit2)(unsigned) = dlsym(libcuda, "cuInit");
        int (*cuDevGet)(int*, int) = dlsym(libcuda, "cuDeviceGet");
        int (*cuCtxCr)(void**, unsigned, int) = dlsym(libcuda, "cuCtxCreate_v2");
        cuInit2(0);
        int cdev; cuDevGet(&cdev, 0);
        void* cctx2; cuCtxCr(&cctx2, 0, cdev);
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC edesc = {};
        edesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        edesc.handle.fd = fd;
        edesc.size = g_asMemSize;
        CUexternalMemory ext = NULL;
        printf("IMPORT desc: type=%d fd=%d size=%llu\n", (int)edesc.type, edesc.handle.fd, edesc.size);
        CUresult rc = cuImportExternalMemory(&ext, &edesc);
        fprintf(stderr, "step: import rc=%d (type=%d fd=%d size=%llu)\n", (int)rc, (int)edesc.type, edesc.handle.fd, edesc.size);
        CUdeviceptr cptr = 0;
        if (rc == CUDA_SUCCESS) {
            {
                CUDA_EXTERNAL_MEMORY_BUFFER_DESC bdesc = {};
                bdesc.offset = 0; bdesc.size = g_asMemSize; bdesc.flags = 0;
                rc = cuExternalMemoryGetMappedBuffer((void**)&cptr, ext, &bdesc);
            }
            printf("CUDA import: rc=%d ptr=0x%llx (Vulkan VA 0x40e1000, allocSize=%llu)\n",
                   (int)rc, (unsigned long long)cptr, g_asMemSize);
    } else {
        printf("CUDA import FAILED rc=%d: size=%llu fd=%d\n", (int)rc, g_asMemSize, fd);
        }
    }

    // === CUDA cross-API probe: can a CUDA context read the Vulkan AS VA? ===
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (cuda) {
        int (*cuInit)(unsigned) = (int (*)(unsigned))dlsym(cuda, "cuInit");
        int (*cuDeviceGet)(int*, int) = (int (*)(int*, int))dlsym(cuda, "cuDeviceGet");
        int (*cuCtxCreate)(void**, unsigned, int) = (int (*)(void**, unsigned, int))dlsym(cuda, "cuCtxCreate_v2");
        int (*cuMemcpyDtoH)(void*, unsigned long long, size_t) = (int (*)(void*, unsigned long long, size_t))dlsym(cuda, "cuMemcpyDtoH_v2");
        int rc;
        if ((rc = cuInit(0)) == 0) {
            int dev; cuDeviceGet(&dev, 0);
            void* ctx; rc = cuCtxCreate(&ctx, 0, dev);
            printf("CUDA ctx rc=%d\n", rc);
            unsigned long long buf[4] = {0};
            rc = cuMemcpyDtoH(buf, 0x40e1000ULL + 0xd0, 8);   // AS self-pointer field
            printf("read Vulkan VA 0x40e1000+0xd0: rc=%d value=0x%llx (expect 0x40e1000)\n",
                   rc, buf[0]);
        }
    }
    return 0;
}
