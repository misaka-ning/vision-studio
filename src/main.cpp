#include "ui/mainwindow.h"
#include "core/gpuruntime.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFontDatabase>
#include <QTimer>
int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication app(argc, argv);
    app.setApplicationName("Vision Studio");
    app.setOrganizationName("VisionStudio");
    app.setApplicationVersion("1.6.0");
    app.setWindowIcon(QIcon(":/app-icon.svg"));
    app.setStyle("Fusion");
    app.setFont(QFont("Noto Sans CJK SC", 10));
    QCommandLineParser parser;
    parser.setApplicationDescription("Vision Studio · 本地 YOLO 视觉推理工作台");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOption({"screenshot", "保存工作台截图并退出", "path"});
    parser.addOption({"smoke", "执行真实示例 ONNX 推理、导出和 UI 自检后退出", "directory"});
    parser.addOption({"demo", "启动后运行内置真实 YOLO 示例"});
    parser.addOption({"pt-demo", "启动后运行 PyTorch YOLOv8n .pt 示例"});
    parser.addOption({"smoke-pt", "执行真实 .pt 推理、导出和 UI 自检后退出", "directory"});
    parser.addOption({"smoke-model", "执行真实模型解析、三种视图与缓存自检后退出", "directory"});
    parser.addOption({"display-model", "指定结构显示的模型文件", "path"});
    parser.addOption({"device", "推理设备：auto、cpu 或 cuda", "mode"});
    parser.addOption({"gpu-index", "NVIDIA GPU 索引（默认 0）", "index", "0"});
    parser.process(app);
    const QString device = parser.value("device").toLower();
    bool validIndex = false;
    const int gpuIndex = parser.value("gpu-index").toInt(&validIndex);
    if (!validIndex || gpuIndex < 0 || gpuIndex > 63 ||
        (parser.isSet("device") && device != "auto" && device != "cpu" && device != "cuda"))
    {
        qCritical("Invalid device: use --device auto|cpu|cuda and --gpu-index 0..63");
        return 2;
    }
    MainWindow window;
    if (parser.isSet("device"))
        window.setComputeDevice(device == "cuda" ? vision::ComputeDevice::CUDA
                               : device == "cpu" ? vision::ComputeDevice::CPU
                                                 : vision::ComputeDevice::Auto,
                                gpuIndex);
    window.show();
    if (parser.isSet("screenshot"))
        QTimer::singleShot(300, &window,
                           [&]
                           {
                               window.saveScreenshot(parser.value("screenshot"));
                               app.quit();
                           });
    else if (parser.isSet("smoke-model"))
    {
        QTimer::singleShot(200, &window,
                           [&] { window.runModelSmoke(parser.value("smoke-model"), parser.value("display-model")); });
        QTimer::singleShot(60000, &app, [&] { qCritical("Model display smoke test timed out"); app.exit(3); });
    }
    else if (parser.isSet("smoke") || parser.isSet("smoke-pt"))
    {
        QTimer::singleShot(200, &window,
                           [&]
                           {
                               if (parser.isSet("smoke-pt"))
                                   window.runPtSmoke(parser.value("smoke-pt"));
                               else
                                   window.runSmoke(parser.value("smoke"));
                           });
        QTimer::singleShot(150000, &app,
                           [&]
                           {
                               qCritical("Smoke test timed out");
                               app.exit(3);
                           });
    }
    else if (parser.isSet("demo"))
        QTimer::singleShot(250, &window, &MainWindow::runDemo);
    else if (parser.isSet("pt-demo"))
        QTimer::singleShot(250, &window, &MainWindow::runPtDemo);
    else if (parser.isSet("display-model"))
        QTimer::singleShot(250, &window,
                           [&] { window.showModelStructure(parser.value("display-model")); });
    return app.exec();
}
