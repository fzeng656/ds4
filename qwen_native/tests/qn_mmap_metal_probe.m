#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

static uint64_t round_down_u64(uint64_t x, uint64_t a) { return x & ~(a - 1); }
static uint64_t round_up_u64(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s shard file_offset payload_bytes\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    uint64_t file_off = strtoull(argv[2], NULL, 10);
    uint64_t bytes = strtoull(argv[3], NULL, 10);
    long page_l = sysconf(_SC_PAGESIZE);
    uint64_t page = (uint64_t)page_l;
    uint64_t map_off = round_down_u64(file_off, page);
    uint64_t leading = file_off - map_off;
    uint64_t map_bytes = round_up_u64(leading + bytes, page);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    void *map = mmap(NULL, (size_t)map_bytes, PROT_READ, MAP_SHARED, fd, (off_t)map_off);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    uint8_t *payload = (uint8_t *)map + leading;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device const uint *in [[buffer(0)]], device ulong *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) { if (gid==0) { ulong s=0; for(uint i=0;i<1024;i++) s += (ulong)in[i]; out[0]=s; } }";
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"probe"];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { fprintf(stderr, "pipeline: %s\n", [[err description] UTF8String]); return 1; }

        id<MTLBuffer> in = [dev newBufferWithBytesNoCopy:payload length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!in) { fprintf(stderr, "newBufferWithBytesNoCopy failed (payload alignment=%llu)\n", (unsigned long long)((uintptr_t)payload % page)); return 1; }
        id<MTLBuffer> out = [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:ps];
        [ce setBuffer:in offset:0 atIndex:0];
        [ce setBuffer:out offset:0 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [ce endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "command error: %s\n", [[cb.error description] UTF8String]); return 1; }
        uint64_t gpu = *(uint64_t *)out.contents;
        uint64_t cpu = 0; const uint32_t *u = (const uint32_t *)payload;
        for (int i=0;i<1024;i++) cpu += u[i];
        printf("page=%llu file_off=%llu leading=%llu map_bytes=%llu payload=%llu\n", (unsigned long long)page,(unsigned long long)file_off,(unsigned long long)leading,(unsigned long long)map_bytes,(unsigned long long)bytes);
        printf("cpu_checksum=%llu gpu_checksum=%llu match=%s\n", (unsigned long long)cpu,(unsigned long long)gpu,cpu==gpu?"YES":"NO");
        in = nil; out = nil; ps = nil; fn = nil; lib = nil; q = nil;
    }
    int mad = posix_madvise(map, (size_t)map_bytes, POSIX_MADV_DONTNEED);
    printf("madvise=%d (%s)\n", mad, mad?strerror(mad):"ok");
    munmap(map, (size_t)map_bytes); close(fd);
    return 0;
}
