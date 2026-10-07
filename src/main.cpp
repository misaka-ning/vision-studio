#include "ui/mainwindow.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFontDatabase>
#include <QTimer>
int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setApplicationName("Vision Studio");
    app.setOrganizationName("VisionStudio");
    app.setApplicationVersion("1.2.0");
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
    parser.process(app);
    MainWindow window;
    window.show();
    if (parser.isSet("screenshot"))
        QTimer::singleShot(300, &window,
                           [&]
                           {
                               window.saveScreenshot(parser.value("screenshot"));
                               app.quit();
                           });
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
    return app.exec();
}
