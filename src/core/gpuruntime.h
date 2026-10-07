#pragma once

#include "visiontypes.h"

namespace vision
{
struct GpuRuntimePaths
{
    QString directory;
    QString python;
    QString onnxLibrary;
    QStringList libraryDirectories;
    bool prepared = false;
    QString error;
};

// GPU components live in user data independently of the bundled CPU runtime.
QString gpuRuntimeDirectory();
GpuRuntimePaths gpuRuntimePaths();
QString gpuScriptPath(const QString &name);
QString computeDeviceKey(ComputeDevice device);
} // namespace vision
