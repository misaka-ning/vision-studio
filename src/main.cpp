#include "ui/mainwindow.h"
#include "core/gpuruntime.h"
#include "core/visionengine.h"
#include "ui/modelconversionpage.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFontDatabase>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QTimer>
#include <stdexcept>
#ifdef Q_OS_WIN
#include <cstdlib>
#endif
int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication app(argc, argv);
    app.setApplicationName("Vision Studio");
    app.setOrganizationName("VisionStudio");
    app.setApplicationVersion(QStringLiteral(VISION_STUDIO_APP_VERSION));
#ifdef Q_OS_WIN
    // Packaged builds find their own runtime after installation or relocation.
    const QDir package(QCoreApplication::applicationDirPath() + "/..");
    if (QFileInfo::exists(package.filePath("windows-install.json")))
    {
        if (!qEnvironmentVariableIsSet("VISION_STUDIO_HOME"))
            _wputenv_s(L"VISION_STUDIO_HOME", reinterpret_cast<const wchar_t *>(package.absolutePath().utf16()));
        qputenv("VISION_STUDIO_SYSTEM_INSTALL", "1");
        if (!qEnvironmentVariableIsSet("VISION_STUDIO_PYTHON"))
            _wputenv_s(L"VISION_STUDIO_PYTHON", reinterpret_cast<const wchar_t *>(package.filePath("runtime/python.exe").utf16()));
        qputenv("PYTHONNOUSERSITE", "1");
        qputenv("PYTHONDONTWRITEBYTECODE", "1");
        qputenv("PYTHONUTF8", "1");
        qputenv("PYTHONIOENCODING", "utf-8");
    }
#endif
    app.setWindowIcon(QIcon(":/app-icon.svg"));
    app.setStyle("Fusion");
#ifdef Q_OS_WIN
    app.setFont(QFont("Microsoft YaHei UI", 10));
#else
    app.setFont(QFont("Noto Sans CJK SC", 10));
#endif
    QCommandLineParser parser;
    parser.setApplicationDescription("Vision Studio · 本地 YOLO 视觉推理工作台");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOption({"screenshot", "保存工作台截图并退出", "path"});
    parser.addOption({"smoke", "执行真实示例 ONNX 推理、导出和 UI 自检后退出", "directory"});
    parser.addOption({"smoke-pt", "执行真实 .pt 推理、导出和 UI 自检后退出", "directory"});
    parser.addOption({"smoke-model", "执行真实模型解析、三种视图与缓存自检后退出", "directory"});
    parser.addOption({"smoke-conversion", "执行模型转换界面、导出与实际推理自检后退出", "directory"});
    parser.addOption({"smoke-evaluation", "执行数据集评估、阈值分析、结果导出和界面自检后退出", "directory"});
    parser.addOption({"smoke-dataset-session", "恢复上次数据集并检查跨进程载入，不执行推理", "directory"});
    parser.addOption({"evaluation-request", "评估自检 JSON 请求文件（需配合 --smoke-evaluation）", "path"});
    parser.addOption({"conversion-source", "转换自检的本地 PT 模型", "path"});
    parser.addOption({"conversion-format", "转换自检格式：onnx 或 torchscript", "format", "onnx"});
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
    else if (parser.isSet("smoke-dataset-session"))
        QTimer::singleShot(200, &window, [&] { window.runDatasetSessionSmoke(parser.value("smoke-dataset-session")); });
    else if (parser.isSet("smoke-evaluation"))
    {
        QTimer::singleShot(200, &window, [&] {
            window.runEvaluationSmoke(parser.value("smoke-evaluation"), parser.value("evaluation-request"));
        });
    }
    else if (parser.isSet("smoke-conversion"))
    {
        QTimer::singleShot(200, &window, [&] {
            const QString directory = QDir(parser.value("smoke-conversion")).absolutePath();
            const QString source = parser.value("conversion-source").isEmpty()
                ? qEnvironmentVariable("VISION_STUDIO_HOME") + "/models/yolov8n.pt"
                : QFileInfo(parser.value("conversion-source")).absoluteFilePath();
            const QString format = parser.value("conversion-format");
            if (!QDir().mkpath(directory) || (format != "onnx" && format != "torchscript")) {
                qCritical("Invalid conversion smoke directory/format"); app.exit(2); return;
            }
            window.showModelStructure(source);
            for (auto *button : window.findChildren<QPushButton *>())
                if (button->text() == QStringLiteral("模型转换")) { button->click(); break; }
            auto *page = window.findChild<ModelConversionPage *>();
            page->findChild<QLineEdit *>("conversionOutputDirectory")->setText(directory);
            page->findChild<QLineEdit *>("conversionOutputFilename")->setText("converted." +
                (format == "onnx" ? QString("onnx") : QString("torchscript")));
            page->findChild<QComboBox *>("conversionFormat")->setCurrentIndex(format == "onnx" ? 0 : 1);
            // Format selection refreshes the default name; set the smoke name last.
            page->findChild<QLineEdit *>("conversionOutputFilename")->setText("converted." +
                (format == "onnx" ? QString("onnx") : QString("torchscript")));
            QObject::connect(page, &ModelConversionPage::busyChanged, &window,
                [&, page, directory, source](bool active) {
                    if (active) return;
                    QJsonObject report{{"version", app.applicationVersion()}, {"success", false},
                                       {"source", source}, {"conversion", page->completedMetadata()}};
                    try {
                        const auto metadata = page->completedMetadata();
                        const QString output = metadata.value("output").toString();
                        if (!QFileInfo(output).isFile()) throw std::runtime_error("Conversion did not produce a verified artifact");
                        vision::ModelConfig config;
                        config.modelPath = output;
                        config.device = vision::ComputeDevice::CPU;
                        config.task = metadata.value("task").toString() == "classify" ? vision::ModelTask::Classification :
                            (metadata.value("layout").toString() == "v5" ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8);
                        const auto names = metadata.value("names").toObject();
                        for (int i = 0; i < names.size(); ++i) config.labels.append(names.value(QString::number(i)).toString());
                        config.inputSize = metadata.value("shape").toArray().at(2).toInt();
                        config.colorMode = metadata.value("input_channels").toInt() == 1
                            ? vision::InputColorMode::Grayscale : vision::InputColorMode::Color;
                        vision::VisionEngine engine;
                        engine.load(config);
                        const QString image = qEnvironmentVariable("VISION_STUDIO_HOME") + "/assets/bus.jpg";
                        const auto result = engine.infer(QImage(image), image);
                        report.insert("backend", engine.backendName());
                        report.insert("input_channels", engine.config().inputChannels);
                        report.insert("prediction_count", result.predictions.size());
                        QJsonArray classes;
                        for (const auto &prediction : result.predictions) classes.append(prediction.label);
                        report.insert("classes", classes);
                        report.insert("actual_inference", true);
                        report.insert("success", true);
                    } catch (const std::exception &error) { report.insert("error", QString::fromUtf8(error.what())); }
                    window.saveScreenshot(directory + "/model-conversion.png");
                    QFile file(directory + "/conversion-smoke-report.json");
                    const bool saved = file.open(QIODevice::WriteOnly) && file.write(QJsonDocument(report).toJson()) > 0;
                    app.exit(saved && report.value("success").toBool() ? 0 : 1);
                });
            page->findChild<QPushButton *>("conversionStartButton")->click();
            if (!page->isBusy()) { qCritical("Conversion smoke could not start"); app.exit(1); }
        });
        QTimer::singleShot(150000, &app, [&] { qCritical("Conversion smoke timed out"); app.exit(3); });
    }
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
    else if (parser.isSet("display-model"))
        QTimer::singleShot(250, &window,
                           [&] { window.showModelStructure(parser.value("display-model")); });
    else
        window.restoreDatasetSession();
    return app.exec();
}
