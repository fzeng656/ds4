#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/task_info.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { void *map; size_t file_bytes; size_t map_bytes; int fd; } Mapping;

static void mem_report(const char *tag) {
    mach_task_basic_info_data_t basic;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&basic, &n);
    task_vm_info_data_t vm;
    n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
    printf("MEM %-18s resident=%.2f MiB footprint=%.2f MiB virtual=%.2f MiB\n",
           tag, basic.resident_size/1048576.0, vm.phys_footprint/1048576.0,
           basic.virtual_size/1048576.0);
    fflush(stdout);
}

static Mapping map_file(const char *path) {
    Mapping m = {0};
    m.fd = open(path, O_RDONLY);
    if (m.fd < 0) { perror("open"); exit(2); }
    struct stat st; if (fstat(m.fd,&st)) { perror("stat"); exit(2); }
    m.file_bytes = (size_t)st.st_size;
    size_t p = (size_t)getpagesize();
    m.map_bytes = (m.file_bytes + p-1) & ~(p-1);
    m.map = mmap(NULL, m.map_bytes, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.map == MAP_FAILED) { perror("mmap"); exit(2); }
    return m;
}


static void residency_report(const char *tag, Mapping *maps, int count) {
    size_t p=(size_t)getpagesize(), total_pages=0, resident_pages=0;
    for(int i=0;i<count;i++){
        size_t pages=maps[i].map_bytes/p;
        unsigned char *vec=calloc(pages,1);
        if(!vec) exit(9);
        if(mincore(maps[i].map,maps[i].map_bytes,(char*)vec)!=0){ perror("mincore"); exit(9); }
        for(size_t j=0;j<pages;j++) if(vec[j]&1) resident_pages++;
        total_pages+=pages; free(vec);
    }
    printf("RES %-18s pages=%zu/%zu (%.2f%%) resident=%.2f MiB\n",
           tag,resident_pages,total_pages,total_pages?100.0*resident_pages/total_pages:0.0,
           resident_pages*p/1048576.0); fflush(stdout);
}

static void unmap_file(Mapping *m) {
    if (m->map && m->map != MAP_FAILED) munmap(m->map,m->map_bytes);
    if (m->fd >= 0) close(m->fd);
    memset(m,0,sizeof(*m)); m->fd=-1;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr,"usage: %s shard...\n",argv[0]); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void touch_pages(device const uchar *in [[buffer(0)]], device uint *out [[buffer(1)]], constant ulong &nbytes [[buffer(2)]], uint tid [[thread_position_in_grid]]) { ulong off=(ulong)tid*16384ul; if(off>=nbytes) return; ulong last=nbytes>=4?nbytes-4:0; if(off>last) off=last; device const uint *p=(device const uint *)(in+off); out[tid]=*p ^ (tid*2654435761u); }";
        NSError *err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
        if(!lib){ fprintf(stderr,"metal compile: %s\n",err.localizedDescription.UTF8String); return 3; }
        id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch_pages"] error:&err];
        if(!ps){ fprintf(stderr,"pipeline: %s\n",err.localizedDescription.UTF8String); return 3; }

        int count=argc-1;
        Mapping *maps=calloc((size_t)count,sizeof(*maps));
        NSMutableArray<id<MTLBuffer>> *bufs=[NSMutableArray arrayWithCapacity:(NSUInteger)count];
        NSMutableArray<id<MTLBuffer>> *outs=[NSMutableArray arrayWithCapacity:(NSUInteger)count];
        mem_report("start");
        size_t total=0;
        for(int i=0;i<count;i++){ maps[i]=map_file(argv[i+1]); total+=maps[i].file_bytes; }
        printf("mapped %.2f MiB across %d shards\n", total/1048576.0,count);
        mem_report("after mmap");
        residency_report("after mmap",maps,count);

        for(int i=0;i<count;i++){
            id<MTLBuffer> inbuf=[dev newBufferWithBytesNoCopy:maps[i].map length:maps[i].map_bytes options:MTLResourceStorageModeShared deallocator:nil];
            if(!inbuf){ fprintf(stderr,"no-copy buffer failed %d\n",i); return 4; }
            NSUInteger pages=(maps[i].file_bytes+16383)/16384;
            id<MTLBuffer> outbuf=[dev newBufferWithLength:pages*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            [bufs addObject:inbuf]; [outs addObject:outbuf];
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:ps]; [ce setBuffer:inbuf offset:0 atIndex:0]; [ce setBuffer:outbuf offset:0 atIndex:1];
            uint64_t nb=maps[i].file_bytes; [ce setBytes:&nb length:sizeof(nb) atIndex:2];
            NSUInteger tg=MIN((NSUInteger)256,ps.maxTotalThreadsPerThreadgroup);
            [ce dispatchThreads:MTLSizeMake(pages,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted){ fprintf(stderr,"GPU failed %ld %s\n",(long)cb.status,cb.error.localizedDescription.UTF8String); return 5; }
        }
        mem_report("after GPU touch");
        residency_report("after GPU touch",maps,count);
        sleep(2);
        mem_report("after 2s hold");
        residency_report("after 2s hold",maps,count);

        // Drop Metal references, retain mmap mappings, then advise kernel pages are disposable.
        [bufs removeAllObjects]; [outs removeAllObjects];
        mem_report("buffers released");
        for(int i=0;i<count;i++){
            int rc=posix_madvise(maps[i].map,maps[i].map_bytes,POSIX_MADV_DONTNEED);
            if(rc) fprintf(stderr,"madvise %d rc=%d\n",i,rc);
        }
        mem_report("after madvise");
        residency_report("after madvise",maps,count);
        sleep(2);
        mem_report("after 2s reclaim");
        residency_report("after 2s reclaim",maps,count);
        for(int i=0;i<count;i++){ int rc=msync(maps[i].map,maps[i].map_bytes,MS_INVALIDATE); if(rc) perror("msync invalidate"); }
        residency_report("after invalidate",maps,count);

        // Recreate Metal views on the SAME mmap and fault pages back in.
        for(int i=0;i<count;i++){
            id<MTLBuffer> inbuf=[dev newBufferWithBytesNoCopy:maps[i].map length:maps[i].map_bytes options:MTLResourceStorageModeShared deallocator:nil];
            NSUInteger pages=(maps[i].file_bytes+16383)/16384;
            id<MTLBuffer> outbuf=[dev newBufferWithLength:pages*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:ps]; [ce setBuffer:inbuf offset:0 atIndex:0]; [ce setBuffer:outbuf offset:0 atIndex:1];
            uint64_t nb=maps[i].file_bytes; [ce setBytes:&nb length:sizeof(nb) atIndex:2];
            NSUInteger tg=MIN((NSUInteger)256,ps.maxTotalThreadsPerThreadgroup);
            [ce dispatchThreads:MTLSizeMake(pages,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted){ fprintf(stderr,"GPU refault failed %ld %s\n",(long)cb.status,cb.error.localizedDescription.UTF8String); return 6; }
        }
        mem_report("after GPU refault");
        residency_report("after GPU refault",maps,count);

        // Mapping remains valid: CPU-read one word from each mapping.
        uint64_t chk=0; for(int i=0;i<count;i++) chk += *(volatile uint32_t*)maps[i].map;
        printf("post-reclaim mapping checksum=%llu\n",(unsigned long long)chk);
        mem_report("after reread");
        for(int i=0;i<count;i++) unmap_file(&maps[i]);
        mem_report("after unmap");
        free(maps);
    }
    return 0;
}
