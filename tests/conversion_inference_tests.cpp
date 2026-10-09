#include "core/visionengine.h"
#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"
#include "ui/modelconversionpage.h"

#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QSet>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <cmath>
#include <memory>

namespace
{
QPushButton *namedButton(QWidget *parent, const QString &name)
{
    for (auto *button : parent->findChildren<QPushButton *>())
        if (button->text() == name)
            return button;
    return nullptr;
}
QStringList names(const QJsonObject &metadata)
{
    QStringList result;
    const auto object = metadata.value("names").toObject();
    for (int index = 0; index < object.size(); ++index)
        result.append(object.value(QString::number(index)).toString());
    return result;
}
}

class ConversionInferenceTests final : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        project_ = QStringLiteral(VISION_PROJECT_DIR);
        runtime_ = qEnvironmentVariable("VISION_STUDIO_CONVERSION_PYTHON",
                                        project_ + "/build-2.1.0/release-runtime/bin/python3");
        configuredGpuRuntime_ = qEnvironmentVariable("VISION_STUDIO_GPU_RUNTIME_DIR",
                                                     "/home/misaka/.local/share/vision-studio/gpu-runtime");
        QVERIFY2(QFileInfo(runtime_).isExecutable(), "The prepared conversion runtime is required.");
        fixtures_ = std::make_unique<QTemporaryDir>();
        QVERIFY(fixtures_->isValid());
        QProcess create;
        auto environment = QProcessEnvironment::systemEnvironment();
        for (const QString &key : {QString("YOLO_CONFIG_DIR"), QString("YOLOV5_CONFIG_DIR"),
                                   QString("MPLCONFIGDIR"), QString("TORCH_HOME")})
        {
            const QString directory = fixtures_->path() + '/' + key;
            QVERIFY(QDir().mkpath(directory));
            environment.insert(key, directory);
        }
        environment.insert("PYTHONDONTWRITEBYTECODE", "1");
        environment.insert("YOLO_AUTOINSTALL", "false");
        environment.insert("YOLOv5_AUTOINSTALL", "false");
        environment.insert("CUDA_VISIBLE_DEVICES", "");
        create.setProcessEnvironment(environment);
        create.start(runtime_, {"-c",
            "import runpy,sys; ns=runpy.run_path(sys.argv[1]); "
            "sys.argv=[sys.argv[1],sys.argv[2],sys.argv[3]]; "
            "exec(ns['FIXTURE_SCRIPT'],{'__name__':'__main__'})",
            project_ + "/tests/model_conversion_tests.py", project_, fixtures_->path()});
        QVERIFY2(create.waitForFinished(90000), qPrintable(create.errorString()));
        QVERIFY2(create.exitCode() == 0, create.readAllStandardError().constData());
        QApplication::setStyle("Fusion");
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        for (const QByteArray &key : environmentKeys())
        {
            if (qEnvironmentVariableIsSet(key.constData()))
                present_.insert(key);
            previous_.insert(key, qgetenv(key.constData()));
        }
    }
    void cleanupTestCase()
    {
        for (const QByteArray &key : environmentKeys())
            if (present_.contains(key))
                qputenv(key.constData(), previous_.value(key));
            else
                qunsetenv(key.constData());
        fixtures_.reset();
    }
    void init()
    {
        area_ = std::make_unique<QTemporaryDir>();
        QVERIFY(area_->isValid());
        const QString home = area_->path();
        QVERIFY(QDir().mkpath(home + "/models"));
        QVERIFY(QDir().mkpath(home + "/assets"));
        QVERIFY(QDir().mkpath(home + "/scripts"));
        QVERIFY(QFile::copy(project_ + "/models/yolov5n.onnx", home + "/models/yolov5n.onnx"));
        QVERIFY(QFile::copy(project_ + "/scripts/pt_worker.py", home + "/scripts/pt_worker.py"));
        QVERIFY(QFile::copy(project_ + "/scripts/netron_server.py", home + "/scripts/netron_server.py"));
        QVERIFY(QFile::link(project_ + "/vendor", home + "/vendor"));
        QFile probe(home + "/scripts/gpu_probe.py");
        QVERIFY(probe.open(QIODevice::WriteOnly));
        probe.write("import json; print(json.dumps({'ok':True,'prepared':False,'cuda_available':False,"
                    "'reason':'isolated CPU conversion test','devices':[]}))\n");
        probe.close();
        image_ = QImage(64, 64, QImage::Format_RGB888);
        image_.fill(QColor(240, 200, 160));
        QVERIFY(image_.save(home + "/assets/bus.jpg"));
        image_ = QImage(home + "/assets/bus.jpg");
        QVERIFY(!image_.isNull());
        qputenv("VISION_STUDIO_HOME", home.toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (home + "/output").toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (home + "/output/gpu-runtime").toUtf8());
        qputenv("VISION_STUDIO_PYTHON", runtime_.toUtf8());
        qputenv("VISION_STUDIO_CONVERSION_WORKER", (project_ + "/scripts/model_convert.py").toUtf8());
        for (const QByteArray &key : {QByteArray("YOLO_CONFIG_DIR"), QByteArray("YOLOV5_CONFIG_DIR"),
                                      QByteArray("MPLCONFIGDIR"), QByteArray("TORCH_HOME")})
        {
            const QString directory = home + '/' + QString::fromLatin1(key);
            QVERIFY(QDir().mkpath(directory));
            qputenv(key.constData(), directory.toUtf8());
        }
        QSettings settings(home + "/output/preferences.ini", QSettings::IniFormat);
        settings.setValue("computeDevice", "cpu");
        settings.sync();
        unexpected_.clear();
        guard_ = std::make_unique<QTimer>();
        connect(guard_.get(), &QTimer::timeout, this, [this] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible())
                {
                    unexpected_ = dialog->text();
                    dialog->accept();
                }
        });
        guard_->start(20);
        showWindow();
    }
    void cleanup()
    {
        window_.reset();
        guard_.reset();
        area_.reset();
    }
    void convertedModelsUseTheirOwnTaskClassesChannelsAndPixels_data()
    {
        QTest::addColumn<QString>("source");
        QTest::addColumn<QString>("format");
        QTest::addColumn<int>("channels");
        QTest::addColumn<bool>("preserveGray");
        QTest::addColumn<int>("imageSize");
        QTest::addColumn<bool>("fullModel");
        QTest::addColumn<bool>("cuda");
        QTest::newRow("modern-gray-onnx") << "modern-c1.pt" << "onnx" << 1 << false << 64 << false << false;
        QTest::newRow("legacy-focus-gray-onnx") << "focus-c1.pt" << "onnx" << 1 << false << 64 << false << false;
        QTest::newRow("script-gray-detection-onnx") << "script-detect-c1.pt" << "onnx" << 1 << false << 64 << false << false;
        QTest::newRow("script-gray-classification-onnx") << "script-classify-c1.pt" << "onnx" << 1 << false << 64 << false << false;
        QTest::newRow("modern-c3-classification-torchscript-preserves-gray") << "classify-c3.pt" << "torchscript" << 3 << true << 64 << false << false;
        QTest::newRow("script-color-detection-torchscript") << "script-detect-c3.pt" << "torchscript" << 3 << false << 64 << false << false;
        QTest::newRow("full-yolov8n-640-onnx-cpu") << "yolov8n.pt" << "onnx" << 3 << false << 640 << true << false;
        QTest::newRow("full-yolov5n-640-onnx-cpu") << "yolov5n.pt" << "onnx" << 3 << false << 640 << true << false;
        QTest::newRow("full-yolov8n-640-onnx-cuda") << "yolov8n.pt" << "onnx" << 3 << false << 640 << true << true;
        QTest::newRow("full-yolov5n-640-onnx-cuda") << "yolov5n.pt" << "onnx" << 3 << false << 640 << true << true;
        QTest::newRow("full-yolov8n-640-torchscript-cuda") << "yolov8n.pt" << "torchscript" << 3 << false << 640 << true << true;
        QTest::newRow("full-yolov5n-640-torchscript-cuda") << "yolov5n.pt" << "torchscript" << 3 << false << 640 << true << true;
    }
    void convertedModelsUseTheirOwnTaskClassesChannelsAndPixels()
    {
        QFETCH(QString, source);
        QFETCH(QString, format);
        QFETCH(int, channels);
        QFETCH(bool, preserveGray);
        QFETCH(int, imageSize);
        QFETCH(bool, fullModel);
        QFETCH(bool, cuda);
        if (cuda && qEnvironmentVariable("VISION_STUDIO_GPU_TESTS") != "1")
            QSKIP("Full converted model CUDA acceptance requires VISION_STUDIO_GPU_TESTS=1.");
        if (fullModel)
        {
            const QString bus = area_->path() + "/assets/bus.jpg";
            QVERIFY(QFile::remove(bus));
            QVERIFY(QFile::copy(project_ + "/assets/bus.jpg", bus));
            image_ = QImage(bus);
            QVERIFY(!image_.isNull());
            QVERIFY(image_.width() > 640 && image_.height() > 640);
        }
        if (cuda)
        {
            QVERIFY(QFileInfo(configuredGpuRuntime_ + "/ready.json").isFile());
            qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", configuredGpuRuntime_.toUtf8());
        }
        const QString original = (fullModel ? project_ + "/models" : fixtures_->path()) + '/' + source;
        window_->showModelStructure(original);
        if (preserveGray)
        {
            auto *color = window_->findChild<QComboBox *>("inputColorMode");
            color->setCurrentIndex(color->findData("grayscale"));
        }
        QTest::mouseClick(namedButton(window_.get(), "模型转换"), Qt::LeftButton);
        auto *page = window_->findChild<ModelConversionPage *>();
        QVERIFY(page);
        page->findChild<QSpinBox *>("conversionImageSize")->setValue(imageSize);
        auto *formats = page->findChild<QComboBox *>("conversionFormat");
        formats->setCurrentIndex(formats->findData(format));
        QTest::mouseClick(page->findChild<QPushButton *>("conversionStartButton"), Qt::LeftButton);
        QVERIFY(page->isBusy());
        QTRY_VERIFY_WITH_TIMEOUT(!page->isBusy(), 90000);
        QVERIFY2(page->findChild<QLabel *>("conversionStatus")->text() == "转换完成",
                  qPrintable(page->findChild<QLabel *>("conversionSummary")->text() + '\n' + unexpected_));
        const auto metadata = page->completedMetadata();
        QVERIFY(!metadata.isEmpty());
        const QString output = metadata.value("output").toString();
        QVERIFY(QFileInfo(output).isFile());
        QTest::mouseClick(page->findChild<QPushButton *>("conversionAddModelButton"), Qt::LeftButton);
        QCOMPARE(window_->findChild<QSpinBox *>("inputSize")->value(), imageSize);
        auto *color = window_->findChild<QComboBox *>("inputColorMode");
        QCOMPARE(color->currentData().toString(), channels == 1 || preserveGray ? QString("grayscale") : QString("rgb"));
        // Adding the same completed output again must refresh its preset rather
        // than return early merely because the path is already global.
        window_->findChild<QSpinBox *>("inputSize")->setValue(96);
        QTest::mouseClick(namedButton(window_.get(), "模型转换"), Qt::LeftButton);
        QTest::mouseClick(page->findChild<QPushButton *>("conversionAddModelButton"), Qt::LeftButton);
        QCOMPARE(window_->findChild<QSpinBox *>("inputSize")->value(), imageSize);
        QTest::mouseClick(namedButton(window_.get(), "检测工作台"), Qt::LeftButton);
        window_->findChild<QDoubleSpinBox *>("confidence")->setValue(fullModel ? 0.25 : 0.05);
        auto *device = window_->findChild<QComboBox *>("computeDevice");
        QVERIFY(device);
        device->setCurrentIndex(device->findData(cuda ? "cuda" : "cpu"));
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(namedButton(window_.get(), "开始检测"), Qt::LeftButton);
        QCOMPARE(starts.size(), 1);
        const auto request = starts[0][0].value<vision::JobRequest>();
        QCOMPARE(request.config.modelPath, output);
        QCOMPARE(request.config.inputChannels, channels);
        QCOMPARE(request.config.inputSize, imageSize);
        QCOMPARE(request.config.device, cuda ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU);
        QCOMPARE(request.config.task, metadata.value("task").toString() == "classify"
                                         ? vision::ModelTask::Classification
                                         : metadata.value("layout").toString() == "v5"
                                               ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8);
        QCOMPARE(request.config.labels, format == "onnx" ? names(metadata) : QStringList{});
        // The existing ONNX scale control stores eight decimal places.
        QVERIFY(std::abs(request.config.scale - 1.0 / 255.0) < 5e-9);
        auto *run = namedButton(window_.get(), "开始检测");
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 30000);
        QVERIFY2(unexpected_.isEmpty(), qPrintable(unexpected_));

        vision::ModelConfig sourceConfig = request.config;
        sourceConfig.modelPath = original;
        sourceConfig.labels.clear();
        // Compare the exported CPU/CUDA output to the original PT on CPU.
        // Real bus predictions expose anchor/coordinate errors hidden by empty
        // outputs or clipped boxes in small uniform-image fixtures.
        sourceConfig.device = vision::ComputeDevice::CPU;
        vision::VisionEngine originalEngine, convertedEngine;
        originalEngine.load(sourceConfig);
        convertedEngine.load(request.config);
        const auto before = originalEngine.infer(image_);
        const auto after = convertedEngine.infer(image_);
        QCOMPARE(convertedEngine.config().inputChannels, channels);
        QCOMPARE(convertedEngine.config().labels, names(metadata));
        QCOMPARE(after.requestedDevice, request.config.device);
        QCOMPARE(after.device, request.config.device);
        QCOMPARE(after.deviceIndex, cuda ? 0 : -1);
        QCOMPARE(after.backend, format == "onnx" ? (cuda ? QString("ONNX Runtime / CUDA") : QString("OpenCV DNN / CPU"))
                                               : (cuda ? QString("PyTorch / TorchScript / CUDA") : QString("PyTorch / TorchScript / CPU")));
        QVERIFY(after.deviceNotice.isEmpty());
        if (fullModel)
            QVERIFY(!before.predictions.isEmpty());
        QCOMPARE(after.predictions.size(), before.predictions.size());
        for (int index = 0; index < before.predictions.size(); ++index)
        {
            const auto expected = before.predictions[index], actual = after.predictions[index];
            QCOMPARE(actual.classId, expected.classId);
            QCOMPARE(actual.label, expected.label);
            QVERIFY(std::abs(actual.confidence - expected.confidence) < 0.002);
            QVERIFY(std::abs(actual.box.x() - expected.box.x()) < 0.05);
            QVERIFY(std::abs(actual.box.y() - expected.box.y()) < 0.05);
            QVERIFY(std::abs(actual.box.width() - expected.box.width()) < 0.05);
            QVERIFY(std::abs(actual.box.height() - expected.box.height()) < 0.05);
        }
        if (channels == 1 || preserveGray)
        {
            const QColor pixel = after.image.pixelColor(0, 0);
            QCOMPARE(pixel.red(), pixel.green());
            QCOMPARE(pixel.green(), pixel.blue());
        }
        QSettings settings(area_->path() + "/output/preferences.ini", QSettings::IniFormat);
        settings.sync();
        const auto stored = settings.value("convertedModelProfiles").toMap().value(output).toMap();
        QCOMPARE(stored.value("fileSize").toLongLong(), QFileInfo(output).size());
        QCOMPARE(stored.value("fileModifiedMs").toLongLong(), QFileInfo(output).lastModified().toMSecsSinceEpoch());
        QCOMPARE(stored.value("metadata").toMap().value("task").toString(), metadata.value("task").toString());
        if (source == "script-detect-c1.pt" && format == "onnx")
            verifyPersistenceAndManualLabels(output, metadata);
    }

  private:
    void verifyPersistenceAndManualLabels(const QString &output, const QJsonObject &metadata)
    {
        window_->close();
        window_.reset();
        showWindow();
        QCOMPARE(window_->findChild<QSpinBox *>("inputSize")->value(), 64);
        QCOMPARE(window_->findChild<QComboBox *>("inputColorMode")->currentData().toString(), QString("grayscale"));
        QVERIFY(namedButton(window_.get(), QString("类别标签 · 转换模型 %1 类").arg(names(metadata).size())));
        window_->close();
        window_.reset();
        QSettings settings(area_->path() + "/output/preferences.ini", QSettings::IniFormat);
        const QStringList manual = {"自定义 球", "自定义 其他"};
        settings.setValue("labels", manual);
        settings.setValue("labelsPath", area_->path() + "/manual.names");
        settings.sync();
        showWindow();
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(namedButton(window_.get(), "开始检测"), Qt::LeftButton);
        QCOMPARE(starts.size(), 1);
        QCOMPARE(starts[0][0].value<vision::JobRequest>().config.labels, manual);
        auto *run = namedButton(window_.get(), "开始检测");
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 30000);
        QVERIFY2(unexpected_.isEmpty(), qPrintable(unexpected_));
        window_->close();
        window_.reset();
        settings.setValue("labels", QStringList{});
        settings.setValue("labelsPath", QString());
        settings.sync();
        QFile file(output);
        QVERIFY(file.open(QIODevice::ReadWrite));
        QVERIFY(file.setFileTime(QFileInfo(output).lastModified().addSecs(2), QFileDevice::FileModificationTime));
        file.close();
        showWindow();
        // Same bytes with a different mtime must not receive the old class preset.
        QVERIFY(namedButton(window_.get(), "类别标签 · 默认 COCO 80"));
        window_->close();
        window_.reset();
        QVERIFY(file.open(QIODevice::Append));
        QCOMPARE(file.write("changed"), qint64(7));
        file.close();
        showWindow();
        QVERIFY(namedButton(window_.get(), "类别标签 · 默认 COCO 80"));
    }
    static QList<QByteArray> environmentKeys()
    {
        return {"VISION_STUDIO_HOME", "VISION_STUDIO_DATA_DIR", "VISION_STUDIO_GPU_RUNTIME_DIR",
                "VISION_STUDIO_PYTHON", "VISION_STUDIO_CONVERSION_WORKER", "YOLO_CONFIG_DIR",
                "YOLOV5_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"};
    }
    void showWindow()
    {
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        QTest::qWait(20);
    }
    QString project_, runtime_, configuredGpuRuntime_, unexpected_;
    QImage image_;
    QHash<QByteArray, QByteArray> previous_;
    QSet<QByteArray> present_;
    std::unique_ptr<QTemporaryDir> fixtures_, area_;
    std::unique_ptr<MainWindow> window_;
    std::unique_ptr<QTimer> guard_;
};

int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication application(argc, argv);
    ConversionInferenceTests tests;
    return QTest::qExec(&tests, argc, argv);
}
#include "conversion_inference_tests.moc"
