#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

/* CreateProcessW allows at most 32767 characters including the null terminator. */
#define COMMAND_SIZE 32767

#pragma comment(lib, "user32.lib")

static BOOL WINAPI console_handler(DWORD event) {
    /* The Python child receives the same console event. */
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

static const wchar_t *argument_tail(void) {
    const wchar_t *command = GetCommandLineW();
    if (*command == L'"') {
        const wchar_t *end = wcschr(command + 1, L'"');
        return end ? end + 1 : L"";
    }
    return command + wcscspn(command, L" \t");
}

static int run(BOOL graphical) {
    const wchar_t *arguments = argument_tail();
    if (!graphical && wcscmp(arguments + wcsspn(arguments, L" \t"),
        L"--fcc-notify-environment") == 0) {
        SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)L"Environment",
            SMTO_ABORTIFHUNG, 1000, NULL);
        return 0;
    }

    wchar_t launcher[COMMAND_SIZE];
    wchar_t python[COMMAND_SIZE];
    wchar_t command[COMMAND_SIZE];
    DWORD size = GetModuleFileNameW(NULL, launcher, COMMAND_SIZE);
    if (!size || size >= COMMAND_SIZE) {
        return 1;
    }
    wchar_t *filename = wcsrchr(launcher, L'\\');
    if (!filename || swprintf_s(python, COMMAND_SIZE, L"%.*ls\\..\\runtime\\%ls",
        (int)(filename - launcher), launcher, graphical ? L"pythonw.exe" : L"python.exe") < 0) {
        return 1;
    }
    if (swprintf_s(command, COMMAND_SIZE,
        L"\"%ls\" -X utf8 -m free_claude_code.cli.windows_entrypoint \"%ls\"%ls",
        python, launcher, arguments) < 0) {
        return 1;
    }

    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
    HANDLE job = CreateJobObjectW(NULL, NULL);
    /* Harness launchers may start an FCC server that outlives the command. */
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation,
        &limits, sizeof(limits))) {
        if (job) {
            CloseHandle(job);
        }
        return 1;
    }
    startup.cb = sizeof(startup);
    SetConsoleCtrlHandler(console_handler, TRUE);
    if (!CreateProcessW(python, command, NULL, NULL, TRUE, CREATE_SUSPENDED,
        NULL, NULL, &startup, &process)) {
        CloseHandle(job);
        return 1;
    }

    DWORD result = 1;
    if (AssignProcessToJobObject(job, process.hProcess) &&
        ResumeThread(process.hThread) != (DWORD)-1) {
        if (WaitForSingleObject(process.hProcess, INFINITE) == WAIT_OBJECT_0) {
            GetExitCodeProcess(process.hProcess, &result);
        }
    } else {
        TerminateProcess(process.hProcess, 1);
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(job);
    return (int)result;
}

int wmain(void) {
    return run(FALSE);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    (void)instance;
    (void)previous;
    (void)command;
    (void)show;
    return run(TRUE);
}
