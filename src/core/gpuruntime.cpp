#include "gpuruntime.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

namespace vision
{
QString computeDeviceKey(ComputeDevice device)
{
    switch (device)
    {
    case ComputeDevice::Auto:
        return QStringLiteral("auto");
    case ComputeDevice::CUDA:
        return QStringLiteral("cuda");
    case ComputeDevice::CPU:
        return QStringLiteral("cpu");
    }
    return QStringLiteral("cpu");
}

QString gpuRuntimeDirectory()
{
    const QString override = qEnvironmentVariable("VISION_STUDIO_GPU_RUNTIME_DIR");
    if (!override.isEmpty())
        return QDir(override).absolutePath();
    QString data = qEnvironmentVariable("VISION_STUDIO_DATA_DIR");
    if (data.isEmpty())
        data = QDir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation))
                   .filePath(QStringLiteral("vision-studio"));
    return QDir(data).absoluteFilePath(QStringLiteral("gpu-runtime"));
}

QString gpuScriptPath(const QString &name)
{
    if (name.contains('/') || name.contains('\\') || name.contains(QStringLiteral("..")))
        return {};
    const QString app = QCoreApplication::applicationDirPath();
    const QStringList candidates = {QDir(app).filePath("../scripts/" + name),
                                    QDir(app).filePath("../../scripts/" + name),
                                    QDir(QStringLiteral(VISION_PROJECT_DIR)).filePath("scripts/" + name)};
    for (const QString &path : candidates)
        if (QFileInfo(path).isFile())
            return QFileInfo(path).absoluteFilePath();
    return {};
}

GpuRuntimePaths gpuRuntimePaths()
{
    GpuRuntimePaths paths;
    paths.directory = gpuRuntimeDirectory();
    paths.python = QDir(paths.directory).filePath(QStringLiteral("bin/python"));
    QFile ready(QDir(paths.directory).filePath(QStringLiteral("ready.json")));
    if (!ready.open(QIODevice::ReadOnly) || ready.size() > 65536)
    {
        paths.error = QStringLiteral("GPU 环境尚未准备，请在「更多 → 计算环境」中准备 GPU 支持。");
        return paths;
    }
    QJsonParseError parse;
    const QJsonDocument document = QJsonDocument::fromJson(ready.readAll(), &parse);
    const QJsonObject metadata = document.object();
    if (parse.error != QJsonParseError::NoError || !document.isObject() ||
        metadata.value("schema").toInt() != 1 ||
        metadata.value("torch_version").toString() != "2.9.1+cu128" ||
        metadata.value("torch_cuda").toString() != "12.8" ||
        metadata.value("ort_version").toString() != "1.23.2")
    {
        paths.error = QStringLiteral("GPU 环境记录不完整或版本不匹配，请重新检查 GPU 支持。");
        return paths;
    }
    if (!QFileInfo(paths.python).isExecutable())
    {
        paths.error = QStringLiteral("GPU 环境缺少可运行的 Python，请重新准备 GPU 支持。");
        return paths;
    }
    const QDir lib(QDir(paths.directory).filePath(QStringLiteral("lib")));
    const QStringList pythonDirectories = lib.entryList({QStringLiteral("python3.*")}, QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QString &pythonDirectory : pythonDirectories)
    {
        const QDir packages(lib.filePath(pythonDirectory + QStringLiteral("/site-packages")));
        const QDir ort(packages.filePath(QStringLiteral("onnxruntime/capi")));
        const QString library = ort.filePath(QStringLiteral("libonnxruntime.so.1.23.2"));
        if (!QFileInfo(library).isFile())
            continue;
        paths.onnxLibrary = library;
        paths.libraryDirectories << ort.absolutePath();
        const QString torchLib = packages.filePath(QStringLiteral("torch/lib"));
        if (QFileInfo(torchLib).isDir())
            paths.libraryDirectories << torchLib;
        const QDir nvidia(packages.filePath(QStringLiteral("nvidia")));
        for (const QString &component : nvidia.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
        {
            const QString directory = nvidia.filePath(component + QStringLiteral("/lib"));
            if (QFileInfo(directory).isDir())
                paths.libraryDirectories << directory;
        }
        break;
    }
    paths.libraryDirectories.removeDuplicates();
    if (paths.onnxLibrary.isEmpty())
    {
        paths.error = QStringLiteral("GPU 环境缺少 ONNX Runtime 1.23.2 动态库，请重新准备 GPU 支持。");
        return paths;
    }
    paths.prepared = true;
    return paths;
}
} // namespace vision
