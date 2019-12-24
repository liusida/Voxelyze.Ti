#if !defined(UTILS_DEBUG_H)
#define UTILS_DEBUG_H

#define DEBUG_HOST_ENABLED true
#define debugHost(cmd) { if (DEBUG_HOST_ENABLED) {printf(("\n[debugHost] %s(%d): <%s> "), __FILE__, __LINE__,__FUNCTION__); cmd; } }

#ifdef __CUDACC__
#define DEBUG_DEV_ENABLED true
#define debugDev(cmd) { if (DEBUG_DEV_ENABLED) {printf(("\n[debugDev] %s(%d): <%s> "), __FILE__, __LINE__,__FUNCTION__); cmd; } }
#endif

#endif // UTILS_DEBUG_H
