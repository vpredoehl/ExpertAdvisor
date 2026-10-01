#include <csignal>
#include <cstdlib>
#include <fstream>
#include <libproc.h>
#include <string>
#include <unistd.h>

namespace
{
volatile sig_atomic_t terminateRequested = 0;

void HandleTerm(int)
{
    terminateRequested = 1;
}
} // namespace

int main(int argc, char* argv[])
{
    if (::setsid() < 0) return 2;
    struct sigaction action{};
    sigemptyset(&action.sa_mask);
    action.sa_handler = HandleTerm;
    if (::sigaction(SIGTERM, &action, nullptr) != 0) return 3;

    std::string readyPath;
    std::string commandLine;
    for (int index = 0; index < argc; ++index)
    {
        if (!commandLine.empty()) commandLine.push_back(' ');
        commandLine += argv[index];
        const std::string argument{argv[index]};
        constexpr const char* prefix = "--ready-file=";
        if (argument.starts_with(prefix))
            readyPath = argument.substr(std::char_traits<char>::length(prefix));
    }
    if (readyPath.empty()) return 4;

    proc_bsdinfo info{};
    const int bytes = ::proc_pidinfo(
        ::getpid(), PROC_PIDTBSDINFO, 0, &info, sizeof(info));
    if (bytes != static_cast<int>(sizeof(info))) return 5;
    char* resolved = ::realpath(argv[0], nullptr);
    if (resolved == nullptr) return 6;
    const std::string executable{resolved};
    std::free(resolved);

    std::ofstream ready{readyPath, std::ios::trunc};
    if (!ready) return 7;
    ready << ::getpid() << '|' << ::getpgrp() << '|'
          << info.pbi_start_tvsec << ':' << info.pbi_start_tvusec << '|'
          << executable << '|' << commandLine << '\n';
    ready.close();
    if (!ready) return 8;

    while (!terminateRequested) ::pause();
    return 0;
}
