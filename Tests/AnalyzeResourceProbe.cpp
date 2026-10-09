// Test-only Darwin sampler and owning-parent exit accounting. Never published.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double clockSeconds(clockid_t id) {
    timespec value{};
    clock_gettime(id, &value);
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) / 1e9;
}

extern "C" int ea_process(int pid, char* buffer, size_t size) {
    proc_bsdinfo before{}, after{};
    rusage_info_v2 usage{};
    char path[PROC_PIDPATHINFO_MAXSIZE]{};
    if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &before, sizeof(before)) != sizeof(before))
        return 0;
    const int rc = proc_pid_rusage(pid, RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&usage));
    const int pathRc = proc_pidpath(pid, path, sizeof(path));
    if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &after, sizeof(after)) != sizeof(after))
        return 0; // A short-lived process exited during the read; never attribute partial data.
    if (before.pbi_start_tvsec != after.pbi_start_tvsec || before.pbi_start_tvusec != after.pbi_start_tvusec)
        return -1;
    // The caller supplies only controlled paths; JSON escaping is still explicit.
    char escaped[PROC_PIDPATHINFO_MAXSIZE * 2]{};
    mach_timebase_info_data_t timebase{};
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0) return -1;
    const auto nanoseconds = [&](uint64_t ticks) {
        return static_cast<unsigned long long>(static_cast<long double>(ticks) * timebase.numer / timebase.denom);
    };
    size_t used = 0;
    for (const char* p = path; *p && used + 2 < sizeof(escaped); ++p) {
        if (*p == '\\' || *p == '"') escaped[used++] = '\\';
        if (static_cast<unsigned char>(*p) < 32) return -1;
        escaped[used++] = *p;
    }
    const int written = snprintf(buffer, size,
        "{\"pid\":%d,\"ppid\":%u,\"pgid\":%u,\"start_identity\":\"%llu:%llu\","
        "\"start_epoch\":%.9f,\"state\":%u,\"executable\":\"%s\",\"usage_available\":%s,"
        "\"rss_bytes\":%llu,\"physical_footprint_bytes\":%llu,\"user_ns\":%llu,\"system_ns\":%llu,"
        "\"disk_read_bytes\":%llu,\"disk_write_bytes\":%llu,\"start_abstime\":%llu,\"exit_abstime\":%llu,"
        "\"user_abstime\":%llu,\"system_abstime\":%llu,\"timebase_numer\":%u,\"timebase_denom\":%u}",
        pid, after.pbi_ppid, after.pbi_pgid,
        after.pbi_start_tvsec, after.pbi_start_tvusec,
        static_cast<double>(after.pbi_start_tvsec) + static_cast<double>(after.pbi_start_tvusec)/1e6,
        after.pbi_status, pathRc > 0 ? escaped : "", rc == 0 ? "true" : "false",
        usage.ri_resident_size, usage.ri_phys_footprint, nanoseconds(usage.ri_user_time), nanoseconds(usage.ri_system_time),
        usage.ri_diskio_bytesread, usage.ri_diskio_byteswritten, usage.ri_proc_start_abstime, usage.ri_proc_exit_abstime,
        usage.ri_user_time, usage.ri_system_time, timebase.numer, timebase.denom);
    return written > 0 && static_cast<size_t>(written) < size ? 1 : -1;
}

extern "C" int ea_children(int pid, int* pids, int bytes) {
    return proc_listchildpids(pid, pids, bytes);
}

extern "C" int ea_host(char* buffer, size_t size) {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    const auto rc = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count);
    mach_port_deallocate(mach_task_self(), host);
    int pressure = 0;
    size_t length = sizeof(pressure);
    if (rc != KERN_SUCCESS || sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &length, nullptr, 0) != 0)
        return -1;
    xsw_usage swap{};
    length = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &length, nullptr, 0) != 0) return -1;
    const int written = snprintf(buffer, size,
        "{\"pressure\":%d,\"swapins_pages\":%llu,\"swapouts_pages\":%llu,\"pageouts_pages\":%llu,"
        "\"page_size_bytes\":%d,\"swap_used_bytes\":%llu}",
        pressure, vm.swapins, vm.swapouts, vm.pageouts, getpagesize(), swap.xsu_used);
    return written > 0 && static_cast<size_t>(written) < size ? 1 : -1;
}

static bool privateScheduler() {
    const char* expected = getenv("EA_PHASE24X_SCHEDULER");
    char actual[PROC_PIDPATHINFO_MAXSIZE]{};
    return expected && proc_pidpath(getpid(), actual, sizeof(actual)) > 0 && strcmp(expected, actual) == 0;
}

static void receipt(const char* json) {
    const char* path = getenv("EA_PHASE24X_LIFECYCLE");
    if (!path) return;
    const int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return; // Missing receipts fail the offline coverage gate.
    const size_t length = strlen(json);
    ssize_t written;
    do { written = write(fd, json, length); } while (written < 0 && errno == EINTR);
    close(fd);
}

extern "C" pid_t measured_fork() {
    const bool enabled = privateScheduler();
    const double before = clockSeconds(CLOCK_REALTIME);
    const pid_t child = fork();
    const int saved = errno;
    if (enabled && child > 0) {
        char identity[8192]{}, record[10000]{};
        const int rc = ea_process(child, identity, sizeof(identity));
        snprintf(record, sizeof(record), "{\"event\":\"fork\",\"utc_epoch\":%.9f,\"fork_before_epoch\":%.9f,\"pid\":%d,\"identity\":%s}\n",
                 clockSeconds(CLOCK_REALTIME), before, child, rc == 1 ? identity : "null");
        receipt(record);
    }
    errno = saved;
    return child;
}

extern "C" int measured_execv(const char* path, char* const argv[]) {
    const char* target = getenv("EA_PHASE24X_WORKER");
    if (target && strcmp(path, target) == 0) {
        char app[100]{};
        snprintf(app, sizeof(app), "phase24x_analyze_%d", getpid());
        setenv("PGAPPNAME", app, 1); // Session attribution only; no database mutation.
    }
    return execv(path, argv);
}

extern "C" pid_t measured_waitpid(pid_t pid, int* status, int options) {
    if (!privateScheduler()) return waitpid(pid, status, options);
    char before[8192]{};
    const int rc = pid > 0 ? ea_process(pid, before, sizeof(before)) : 0;
    rusage usage{};
    int localStatus = 0;
    const pid_t result = wait4(pid, status ? status : &localStatus, options, &usage);
    const int saved = errno;
    const int value = result > 0 ? (status ? *status : localStatus) : 0;
    if (result > 0 && (WIFEXITED(value) || WIFSIGNALED(value))) {
        char record[12000]{};
        snprintf(record, sizeof(record),
            "{\"event\":\"reaped\",\"utc_epoch\":%.9f,\"pid\":%d,\"status\":%d,\"identity_before_wait\":%s,"
            "\"peak_rss_bytes\":%ld,\"user_seconds\":%.9f,\"system_seconds\":%.9f,\"input_blocks\":%ld,\"output_blocks\":%ld}\n",
            clockSeconds(CLOCK_REALTIME), result, value, rc == 1 ? before : "null", usage.ru_maxrss,
            static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec)/1e6,
            static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec)/1e6,
            usage.ru_inblock, usage.ru_oublock);
        receipt(record);
    }
    errno = saved;
    return result;
}

#define INTERPOSE(replacement, original) \
    __attribute__((used)) static const struct { const void* replacement; const void* original; } \
    pair_##original __attribute__((section("__DATA,__interpose"))) = { \
        reinterpret_cast<const void*>(&replacement), reinterpret_cast<const void*>(&original) };
INTERPOSE(measured_fork, fork)
INTERPOSE(measured_execv, execv)
INTERPOSE(measured_waitpid, waitpid)
