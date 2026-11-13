// Keep the application executable inside its bundle while running the installed FCC venv.
#import <AppKit/AppKit.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FCC_PYTHON_VERSION
#error "Build with the Python version required by pyproject.toml"
#endif

// Public, opaque CPython 3.14 initialization API. No Python structures are embedded here.
typedef struct PyInitConfig PyInitConfig;

static int fail(NSString *reason) {
    fprintf(stderr, "Free Claude Code: %s\n", reason.UTF8String);
    [NSApplication sharedApplication];
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Free Claude Code could not start";
    alert.informativeText = [reason stringByAppendingString:
        @"\n\nRerun the Free Claude Code installer to repair the desktop launcher."];
    [alert addButtonWithTitle:@"OK"];
    [NSApp activateIgnoringOtherApps:YES];
    [alert runModal];
    return 1;
}

static NSString *absolutePath(NSDictionary *settings, NSString *key) {
    id value = settings[key];
    return [value isKindOfClass:[NSString class]] && [value isAbsolutePath] ? value : nil;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        NSURL *settingsURL = [[NSBundle mainBundle] URLForResource:@"runtime" withExtension:@"plist"];
        NSError *error = nil;
        NSDictionary *settings = settingsURL
            ? [NSDictionary dictionaryWithContentsOfURL:settingsURL error:&error] : nil;
        if (!settings) {
            return fail(@"The application's Python configuration is missing or unreadable.");
        }
        NSString *executable = absolutePath(settings, @"PythonExecutable");
        NSString *library = absolutePath(settings, @"PythonLibrary");
        NSFileManager *files = [NSFileManager defaultManager];
        if (!executable || ![files isExecutableFileAtPath:executable] ||
            !library || ![files isReadableFileAtPath:library]) {
            return fail(@"The installed Python runtime is missing or inaccessible.");
        }
        if (![settings[@"PythonVersion"] isEqual:@FCC_PYTHON_VERSION]) {
            return fail(@"The application and its Python configuration have different versions.");
        }
        void *runtime = dlopen(library.fileSystemRepresentation, RTLD_NOW | RTLD_GLOBAL);
        if (!runtime) {
            return fail([NSString stringWithFormat:@"Python could not be loaded: %s", dlerror()]);
        }

#define LOAD(result, name, arguments) \
        result (*name) arguments = (result (*) arguments)dlsym(runtime, #name); \
        if (!name) return fail(@"The installed Python runtime is incompatible (missing " @#name @").")

        LOAD(const char *, Py_GetVersion, (void));
        LOAD(PyInitConfig *, PyInitConfig_Create, (void));
        LOAD(void, PyInitConfig_Free, (PyInitConfig *));
        LOAD(int, PyInitConfig_SetInt, (PyInitConfig *, const char *, int64_t));
        LOAD(int, PyInitConfig_SetStr, (PyInitConfig *, const char *, const char *));
        LOAD(int, PyInitConfig_SetStrList, (PyInitConfig *, const char *, size_t, char *const *));
        LOAD(int, PyInitConfig_GetError, (PyInitConfig *, const char **));
        LOAD(int, Py_InitializeFromInitConfig, (PyInitConfig *));
        LOAD(int, Py_RunMain, (void));
#undef LOAD

        const char *version = Py_GetVersion();
        size_t expectedLength = strlen(FCC_PYTHON_VERSION);
        if (strncmp(version, FCC_PYTHON_VERSION, expectedLength) != 0 ||
            (version[expectedLength] != ' ' && version[expectedLength] != '\0')) {
            return fail(@"The installed Python version does not match this desktop launcher.");
        }
        unsetenv("PYTHONEXECUTABLE");
        unsetenv("__PYVENV_LAUNCHER__");
        PyInitConfig *config = PyInitConfig_Create();
        if (!config) return fail(@"Python initialization ran out of memory.");
        // Finder need not supply a locale. Isolated CPython otherwise defaults to ASCII text I/O.
        int status = PyInitConfig_SetInt(config, "utf8_mode", 1);
        if (status == 0) status = PyInitConfig_SetStr(config, "executable", executable.UTF8String);
        if (status == 0) status = PyInitConfig_SetInt(config, "parse_argv", 0);
        if (status == 0) status = PyInitConfig_SetInt(config, "site_import", 1);
        if (status == 0) status = PyInitConfig_SetInt(config, "install_signal_handlers", 1);
        if (status == 0) status = PyInitConfig_SetStrList(config, "argv", (size_t)argc, argv);
        if (status == 0) status = PyInitConfig_SetStr(config, "run_command",
            "from free_claude_code.cli.desktop_entrypoint import launch; launch()\n");
        if (status == 0) status = Py_InitializeFromInitConfig(config);
        if (status < 0) {
            const char *message = NULL;
            PyInitConfig_GetError(config, &message);
            NSString *reason = message ? [NSString stringWithUTF8String:message] : @"Python initialization failed.";
            PyInitConfig_Free(config);
            return fail(reason);
        }
        PyInitConfig_Free(config);
        int exitCode = Py_RunMain();
        if (exitCode != 0) return fail(@"Free Claude Code could not finish starting. Check its logs for details.");
        return 0;
    }
}
