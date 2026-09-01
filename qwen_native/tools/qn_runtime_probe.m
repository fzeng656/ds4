#import <Foundation/Foundation.h>
#include "../qn_runtime.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    @autoreleasepool {
        qn_runtime_config_status status = {0};
        char err[512] = {0};
        if (qn_runtime_configure_from_env(&status, err, sizeof(err))) {
            fprintf(stderr, "runtime probe failed: %s\n", err);
            return 1;
        }
        printf("mode=%s prefill_mps=%d moe_bm32=%d metallib=%s\n",
               status.stable_mode ? "stable" : "experimental",
               status.prefill_mps_enabled,
               status.moe_metallib_configured,
               status.moe_metallib_path[0] ? status.moe_metallib_path : "<unset>");
        return 0;
    }
}
