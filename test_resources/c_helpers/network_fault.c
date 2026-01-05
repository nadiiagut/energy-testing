/**
 * network_fault.c - LD_PRELOAD library for network fault injection
 *
 * Intercepts socket syscalls to simulate network faults:
 * - Packet drops (blackhole)
 * - Latency injection (slow)
 * - Connection failures
 *
 * Usage:
 *   gcc -shared -fPIC -o network_fault.so network_fault.c -ldl
 *   NETFAULT_MODE=blackhole NETFAULT_DROP_PERCENT=30 LD_PRELOAD=./network_fault.so <command>
 *
 * Environment variables:
 *   NETFAULT_MODE: none, blackhole, slow, fail
 *   NETFAULT_DROP_PERCENT: 0-100 (for blackhole mode)
 *   NETFAULT_DELAY_MS: milliseconds (for slow mode)
 *   NETFAULT_TARGET_PORT: port to target (0 = all)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

/* Original function pointers */
static ssize_t (*real_send)(int, const void *, size_t, int) = NULL;
static ssize_t (*real_recv)(int, void *, size_t, int) = NULL;
static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;

/* Fault injection modes */
typedef enum {
    MODE_NONE = 0,
    MODE_BLACKHOLE,
    MODE_SLOW,
    MODE_FAIL
} fault_mode_t;

/* Configuration */
static fault_mode_t g_mode = MODE_NONE;
static int g_drop_percent = 30;
static int g_delay_ms = 100;
static int g_target_port = 0;
static int g_initialized = 0;

/* Statistics */
static unsigned long g_packets_intercepted = 0;
static unsigned long g_packets_dropped = 0;
static unsigned long g_packets_delayed = 0;

/**
 * Initialize the fault injection library
 */
static void init_fault_injection(void) {
    if (g_initialized) return;
    
    /* Load original functions */
    real_send = dlsym(RTLD_NEXT, "send");
    real_recv = dlsym(RTLD_NEXT, "recv");
    real_connect = dlsym(RTLD_NEXT, "connect");
    
    /* Seed random */
    srand(time(NULL) ^ getpid());
    
    /* Read configuration from environment */
    const char *mode = getenv("NETFAULT_MODE");
    if (mode) {
        if (strcmp(mode, "blackhole") == 0) g_mode = MODE_BLACKHOLE;
        else if (strcmp(mode, "slow") == 0) g_mode = MODE_SLOW;
        else if (strcmp(mode, "fail") == 0) g_mode = MODE_FAIL;
    }
    
    const char *drop = getenv("NETFAULT_DROP_PERCENT");
    if (drop) g_drop_percent = atoi(drop);
    
    const char *delay = getenv("NETFAULT_DELAY_MS");
    if (delay) g_delay_ms = atoi(delay);
    
    const char *port = getenv("NETFAULT_TARGET_PORT");
    if (port) g_target_port = atoi(port);
    
    g_initialized = 1;
    
    if (g_mode != MODE_NONE) {
        fprintf(stderr, "[NETFAULT] Initialized: mode=%d drop=%d%% delay=%dms port=%d\n",
                g_mode, g_drop_percent, g_delay_ms, g_target_port);
    }
}

/**
 * Check if we should apply fault to this packet
 */
static int should_drop(void) {
    return (rand() % 100) < g_drop_percent;
}

/**
 * Apply delay
 */
static void apply_delay(void) {
    if (g_delay_ms > 0) {
        usleep(g_delay_ms * 1000);
        g_packets_delayed++;
    }
}

/**
 * Intercepted send()
 */
ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
    init_fault_injection();
    g_packets_intercepted++;
    
    switch (g_mode) {
        case MODE_BLACKHOLE:
            if (should_drop()) {
                g_packets_dropped++;
                return len;  /* Pretend we sent it */
            }
            break;
        case MODE_SLOW:
            apply_delay();
            break;
        case MODE_FAIL:
            errno = ECONNRESET;
            return -1;
        default:
            break;
    }
    
    return real_send(sockfd, buf, len, flags);
}

/**
 * Intercepted recv()
 */
ssize_t recv(int sockfd, void *buf, size_t len, int flags) {
    init_fault_injection();
    g_packets_intercepted++;
    
    switch (g_mode) {
        case MODE_BLACKHOLE:
            if (should_drop()) {
                g_packets_dropped++;
                errno = EAGAIN;
                return -1;  /* Simulate timeout */
            }
            break;
        case MODE_SLOW:
            apply_delay();
            break;
        case MODE_FAIL:
            errno = ECONNRESET;
            return -1;
        default:
            break;
    }
    
    return real_recv(sockfd, buf, len, flags);
}

/**
 * Intercepted connect()
 */
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    init_fault_injection();
    
    if (g_mode == MODE_FAIL) {
        errno = ECONNREFUSED;
        return -1;
    }
    
    if (g_mode == MODE_SLOW) {
        apply_delay();
    }
    
    return real_connect(sockfd, addr, addrlen);
}

/**
 * Get statistics (callable from test code)
 */
void netfault_get_stats(unsigned long *intercepted, 
                        unsigned long *dropped, 
                        unsigned long *delayed) {
    if (intercepted) *intercepted = g_packets_intercepted;
    if (dropped) *dropped = g_packets_dropped;
    if (delayed) *delayed = g_packets_delayed;
}

/**
 * Reset statistics
 */
void netfault_reset_stats(void) {
    g_packets_intercepted = 0;
    g_packets_dropped = 0;
    g_packets_delayed = 0;
}
