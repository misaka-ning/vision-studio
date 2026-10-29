// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/mainwindow.h"
#include "core/cpuruntime.h"
#include "ui/imagecanvas.h"
#include "ui/pages/workbenchpage.h"
#include "ui/pages/morepage.h"
#include "app/gpuenvironmentservice.h"
#include "app/asyncthreadretirement.h"
#include "app/recordingcontroller.h"
#include "ui/modelviewer.h"
#include "ui/modelconversionpage.h"
#include "ui/asyncimagepreview.h"
#include "ui/runtimesetup.h"
#include "ui_shutdown_helpers.h"
#include "thread_retirement_fixtures.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QImage>
#include <QJsonDocument>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QThread>
#include <QThreadPool>
#include <QTimer>
#include <QtTest>
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>

namespace
{
QByteArray contents(const QString &path)
{
    QFile file(path); return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
void writeFile(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size())
        throw std::runtime_error("Cannot create isolated shutdown fixture");
}
QByteArray number(quint64 value)
{
    QByteArray result;
    do { quint8 byte = value & 127; value >>= 7; result.append(char(byte | (value ? 128 : 0))); } while (value);
    return result;
}
QByteArray integer(int field, quint64 value) { return number(quint64(field) << 3) + number(value); }
QByteArray bytes(int field, const QByteArray &value) { return number((quint64(field) << 3) | 2) + number(value.size()) + value; }
QByteArray text(int field, const char *value) { return bytes(field, QByteArray(value)); }
QByteArray info(const char *name, const QVector<qint64> &dimensions)
{
    QByteArray shape;
    for (auto dimension : dimensions) shape += bytes(1, integer(1, dimension));
    return text(1, name) + bytes(2, bytes(1, integer(1, 1) + bytes(2, shape)));
}
template<class T> QByteArray tensor(const char *name, const QVector<qint64> &dimensions, int type, const QVector<T> &values)
{
    QByteArray data, result;
    for (const auto &value : values) {
        quint8 encoded[sizeof(T)]; std::memcpy(encoded, &value, sizeof(T));
#if Q_BYTE_ORDER == Q_BIG_ENDIAN
        std::reverse(encoded, encoded + sizeof(T));
#endif
        data.append(reinterpret_cast<const char *>(encoded), sizeof(T));
    }
    for (auto dimension : dimensions) result += integer(1, dimension);
    return result + integer(2, type) + text(8, name) + bytes(9, data);
}
QByteArray node(const char *op, const QVector<const char *> &inputs, const char *output, const QByteArray &attributes = {})
{
    QByteArray result;
    for (const auto input : inputs) result += text(1, input);
    return result + text(2, output) + text(4, op) + attributes;
}
void realOnnx(const QString &path)
{
    // Genuine OpenCV Conv -> GlobalAveragePool -> Reshape. One deterministic
    // class and a positive-area box; no inference or worker substitute is used.
    QVector<float> values(40, 0);
    values[0] = values[8] = 32; values[16] = values[24] = 20; values[32] = .8f;
    const auto kernel = bytes(5, text(1, "kernel_shape") + integer(8, 1) + integer(8, 1) + integer(20, 7));
    QByteArray graph;
    graph += bytes(1, node("Conv", {"images", "weights", "bias"}, "conv", kernel));
    graph += bytes(1, node("GlobalAveragePool", {"conv"}, "average"));
    graph += bytes(1, node("Reshape", {"average", "shape"}, "output"));
    graph += text(2, "vision_studio_window_shutdown_fixture");
    graph += bytes(5, tensor("weights", {40, 3, 1, 1}, 1, QVector<float>(120, 0)));
    graph += bytes(5, tensor("bias", {40}, 1, values));
    graph += bytes(5, tensor("shape", {3}, 7, QVector<qint64>{1, 5, 8}));
    graph += bytes(11, info("images", {1, 3, 64, 64}));
    graph += bytes(12, info("output", {1, 5, 8}));
    writeFile(path, integer(1, 7) + text(2, "Vision Studio lifecycle tests") + bytes(7, graph) + bytes(8, integer(2, 11)));
}
QImage texture(const QSize &size)
{
    QImage image(size, QImage::Format_RGB888);
    quint32 state = 0x7ac19e5U;
    for (int row = 0; row < image.height(); ++row)
        for (int column = 0; column < image.width() * 3; ++column) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            image.scanLine(row)[column] = uchar(state);
        }
    return image;
}
}

class WindowShutdownTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> area_;
    std::unique_ptr<MainWindow> window_;
    QHash<QByteArray, QByteArray> savedEnvironment_;
    QSet<QByteArray> missingEnvironment_;
    QString python_, model_, image_, failure_;
    QTimer dialogGuard_, heartbeat_;
    int beats_ = 0, beatsAtClose_ = 0;
    qint64 closeMilliseconds_ = -1;
    bool closeRequested_ = false;

    WorkbenchPage *page() const { return window_->findChild<WorkbenchPage *>(); }
    const WorkbenchPage::Controls &controls() const { return page()->controls(); }
    void environment(const QByteArray &key, const QByteArray &value)
    {
        if (!savedEnvironment_.contains(key)) {
            if (!qEnvironmentVariableIsSet(key.constData())) missingEnvironment_.insert(key);
            savedEnvironment_.insert(key, qgetenv(key.constData()));
        }
        qputenv(key.constData(), value);
    }
    bool waitUntil(const std::function<bool()> &condition, int timeout = 30000)
    {
        QElapsedTimer elapsed; elapsed.start();
        while (!condition() && failure_.isEmpty() && elapsed.elapsed() < timeout) QTest::qWait(5);
        return condition() && failure_.isEmpty();
    }
    void createWindow()
    {
        closeRequested_ = false; closeMilliseconds_ = -1;
        window_ = std::make_unique<MainWindow>(); window_->resize(1200, 820); window_->show();
    }
    void beginClose()
    {
        if (closeRequested_) return;
        closeRequested_ = true; beatsAtClose_ = beats_;
        QElapsedTimer elapsed; elapsed.start(); window_->close(); closeMilliseconds_ = elapsed.elapsed();
        const auto &controls = window_->findChild<WorkbenchPage *>()->controls();
        QVERIFY(!window_->isEnabled());
        QVERIFY(!controls.runButton->isEnabled());
        QVERIFY(!controls.computeDevice->isEnabled());
        QVERIFY(!controls.inputSize->isEnabled());
    }
    bool readyToDestroy()
    {
        return waitUntil([this] { return window_->isShutdownReady() && !window_->isVisible(); });
    }
    void destroyReadyWindow()
    {
        QVERIFY(window_ && window_->isShutdownReady());
        QElapsedTimer elapsed; elapsed.start(); window_.reset();
        QVERIFY2(elapsed.elapsed() < 500, "Destruction must not hide a synchronous wait after the explicit barrier.");
    }
    bool importImages(const QStringList &paths)
    {
        if (!window_->inputs()->requestImages(paths)) return false;
        return waitUntil([this, paths] {
            return !window_->isWorkbenchBusy() && window_->inputState().files == paths &&
                controls().queue->count() == paths.size() && !controls().canvas->result().image.isNull();
        });
    }
    QStringList copies(int count)
    {
        const auto folder = area_->filePath("images"); QDir().mkpath(folder);
        QStringList paths;
        for (int index = 0; index < count; ++index) {
            const auto path = folder + QString("/%1.png").arg(index, 5, 10, QChar('0'));
            if (!QFile::copy(image_, path)) return {};
            paths.append(path);
        }
        return paths;
    }

  private slots:
    void initTestCase()
    {
        QApplication::setStyle("Fusion"); QApplication::setFont(QFont("Noto Sans CJK SC", 10));
        QApplication::setQuitOnLastWindowClosed(false);
        QCoreApplication::setApplicationVersion("2.10.1");
        python_ = qEnvironmentVariable("VISION_STUDIO_EVALUATION_PYTHON");
        if (python_.isEmpty()) python_ = "/opt/VisionStudio/runtime/bin/python";
        QVERIFY2(QFileInfo(python_).isExecutable(), "A prepared local Python is required; lifecycle tests never download runtimes.");
        connect(&dialogGuard_, &QTimer::timeout, this, [this] {
            for (auto *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible()) {
                    failure_ = dialog->text(); dialog->reject();
                }
        });
        connect(&heartbeat_, &QTimer::timeout, this, [this] { ++beats_; });
    }
    void init()
    {
        failure_.clear(); beats_ = 0; beatsAtClose_ = 0;
        area_ = std::make_unique<QTemporaryDir>(); QVERIFY(area_->isValid());
        for (const auto &name : {"models", "assets", "scripts", "data", "config"}) QVERIFY(QDir().mkpath(area_->filePath(name)));
        const QString project = QStringLiteral(VISION_PROJECT_DIR);
        for (const auto &name : {"netron_server.py", "gpu_probe.py"}) {
            const auto source = project + "/scripts/" + name;
            if (QFileInfo(source).isFile()) QVERIFY(QFile::copy(source, area_->filePath(QString("scripts/") + name)));
        }
        model_ = area_->filePath("models/yolov5n.onnx"); realOnnx(model_);
        image_ = area_->filePath("assets/input.png"); QVERIFY(texture({96, 64}).save(image_));
        environment("VISION_STUDIO_HOME", area_->path().toUtf8());
        environment("VISION_STUDIO_DATA_DIR", area_->filePath("data").toUtf8());
        environment("VISION_STUDIO_PYTHON", python_.toUtf8());
        environment("VISION_STUDIO_GPU_RUNTIME_DIR", area_->filePath("no-gpu-runtime").toUtf8());
        environment("XDG_CONFIG_HOME", area_->filePath("config").toUtf8());
        environment("XDG_DATA_HOME", area_->filePath("data").toUtf8());
        environment("VISION_STUDIO_NETRON_WORKER", {});
        environment("VISION_STUDIO_CONVERSION_WORKER", {});
        QSettings settings(area_->filePath("data/preferences.ini"), QSettings::IniFormat);
        settings.setValue("computeDevice", "cpu"); settings.setValue("task", 1);
        settings.setValue("inputSize", 64); settings.setValue("confidence", .25);
        settings.setValue("restoreDatasetOnStartup", false);
        settings.setValue("exportDirectory", area_->filePath("data/results")); settings.sync();
        dialogGuard_.start(10); heartbeat_.start(1); createWindow();
        QVERIFY2(waitUntil([this] { return !window_->modelState().inspectionPending && window_->modelState().classCount == 1; }), qPrintable(failure_));
        QVERIFY(!vision::cpuRuntimeDownloadNeeded());
    }
    void cleanup()
    {
        bool ready = true;
        if (window_) {
            window_->close();
            ready = vision::test::shutdownWindow(window_.get(), 30000);
            if (ready) window_.reset(); else window_.release();
        }
        dialogGuard_.stop(); heartbeat_.stop();
        for (auto it = savedEnvironment_.cbegin(); it != savedEnvironment_.cend(); ++it)
            if (missingEnvironment_.contains(it.key())) qunsetenv(it.key().constData());
            else qputenv(it.key().constData(), it.value());
        savedEnvironment_.clear(); missingEnvironment_.clear();
        if (ready) area_.reset(); else area_.release();
        QVERIFY2(ready, "Lifecycle test cleanup did not reach the explicit barrier; owners and fixture retained rather than blocking in destructors.");
        QVERIFY2(failure_.isEmpty(), qPrintable(failure_));
    }

    void orphanNativeThreadTailRemainsInApplicationCloseBarrier()
    {
        // An accidentally destroyed resource owner must not hide its still
        // finishing native thread from the application's shutdown barrier.
        const auto tail = std::make_shared<vision::test::ThreadTailState>();
        vision::test::ThreadTailRelease release{tail};
        int callbacks = 0;
        auto owner = std::make_unique<QObject>();
        auto *thread = QThread::create([tail] { vision::test::establishThreadLocalTail(tail); });
        QPointer<QThread> guardedThread(thread);
        vision::app::AsyncThreadRetirement::watch(thread, owner.get(), [&callbacks] { ++callbacks; });
        thread->start();
        QVERIFY2(waitUntil([tail] { return tail->entered.load(); }), qPrintable(failure_));
        QVERIFY(thread->isFinished());
        QVERIFY(!tail->completed.load());
        QElapsedTimer elapsed; elapsed.start(); owner.reset();
        QVERIFY2(elapsed.elapsed() < 100, "Losing an owner must not join its thread on the GUI loop.");
        QVERIFY(guardedThread);
        QCOMPARE(callbacks, 0);

        beginClose();
        QVERIFY(closeMilliseconds_ < 500);
        QVERIFY(window_->isClosing());
        QVERIFY(!window_->isShutdownReady());
        // First settle every ordinary participant. Otherwise a slow model
        // helper could make the missing orphan-ticket gate appear to pass.
        auto *recordings = window_->findChild<vision::app::RecordingController *>(); QVERIFY(recordings);
        auto *gpu = window_->findChild<vision::app::GpuEnvironmentService *>(); QVERIFY(gpu);
        auto *viewer = window_->findChild<ModelViewer *>("modelStructureViewer"); QVERIFY(viewer);
        auto *conversion = window_->findChild<ModelConversionPage *>("modelConversionPage"); QVERIFY(conversion);
        QVERIFY2(waitUntil([this, thread, recordings, gpu, viewer, conversion] {
            const auto threads = vision::app::AsyncThreadRetirement::watchedThreads();
            const auto *thumbnails = qApp->findChild<QThreadPool *>("visionThumbnailPool");
            return window_->inputs()->isShutdownReady() && window_->models()->isShutdownReady() &&
                window_->evaluations()->isShutdownReady() && window_->detections()->isShutdownReady() &&
                recordings->isShutdownReady() && gpu->isShutdownReady() && viewer->isShutdownReady() &&
                conversion->isShutdownReady() && page()->isPreviewShutdownReady() &&
                ui::runtimeSetup()->isShutdownReady() &&
                AsyncImagePreview::applicationPool()->activeRequestCount() == 0 &&
                (!thumbnails || thumbnails->activeThreadCount() == 0) &&
                QThreadPool::globalInstance()->activeThreadCount() == 0 &&
                threads.size() == 1 && threads.first() == thread;
        }), qPrintable(failure_));
        const int before = beats_;
        QTest::qWait(60);
        QVERIFY2(beats_ > before, "The GUI loop must keep advancing while native TLS cleanup is pending.");
        QVERIFY2(!window_->isShutdownReady(), "An orphaned retirement ticket must still hold the application close barrier.");
        QVERIFY(guardedThread);
        QCOMPARE(callbacks, 0);

        tail->resume();
        QVERIFY2(readyToDestroy(), qPrintable(failure_));
        QTRY_VERIFY_WITH_TIMEOUT(guardedThread.isNull(), 3000);
        QVERIFY(tail->completed.load());
        QCOMPARE(callbacks, 0);
        destroyReadyWindow();
    }

    void runningRealOnnxTaskClosesWithoutBlockingGui()
    {
        const auto files = copies(40); QCOMPARE(files.size(), 40); QVERIFY(importImages(files));
        const auto before = QCryptographicHash::hash(contents(model_), QCryptographicHash::Sha256);
        const auto accepted = std::make_shared<vision::InferenceResult>();
        auto *detector = window_->findChild<vision::app::DetectionController *>(); QVERIFY(detector);
        connect(detector, &vision::app::DetectionController::imageResultReady, this,
            [this, accepted](const vision::InferenceResult &result, quint64) {
            if (closeRequested_) return; *accepted = result; beginClose();
        });
        controls().runButton->click();
        QVERIFY2(waitUntil([this] { return closeRequested_; }), qPrintable(failure_));
        QVERIFY(closeMilliseconds_ < 500); QVERIFY(window_->isClosing());
        QVERIFY(accepted->backend.contains("OpenCV")); QCOMPARE(accepted->device, vision::ComputeDevice::CPU);
        QVERIFY(!accepted->demonstration); QVERIFY(!accepted->predictions.isEmpty());
        QVERIFY2(readyToDestroy(), qPrintable(failure_)); QVERIFY(beats_ > beatsAtClose_);
        QVERIFY(window_->completedCount() >= 1); QVERIFY(window_->completedCount() < files.size());
        QCOMPARE(QCryptographicHash::hash(contents(model_), QCryptographicHash::Sha256), before);
        destroyReadyWindow();
    }

    void morePagePreparationIntentsAreRejectedAfterCloseStarts()
    {
        auto *more = window_->findChild<MorePage *>("morePage"); QVERIFY(more);
        auto *gpu = window_->findChild<vision::app::GpuEnvironmentService *>(); QVERIFY(gpu);
        QVERIFY2(waitUntil([gpu] { return !gpu->isChecking() && !gpu->isPreparing(); }), qPrintable(failure_));
        auto *runtime = ui::runtimeSetup(); QVERIFY(runtime->isShutdownReady());
        auto *status = window_->findChild<QLabel *>("statusText"); QVERIFY(status);
        QSignalSpy gpuChanges(gpu, &vision::app::GpuEnvironmentService::stateChanged);
        beginClose(); QVERIFY(window_->isClosing());
        QVERIFY(!more->controls().gpuCheckButton->isEnabled());
        QVERIFY(!more->controls().gpuPrepareButton->isEnabled());
        if (more->controls().cpuPrepareButton) QVERIFY(!more->controls().cpuPrepareButton->isEnabled());
        const auto statusAtClose = status->text();
        const auto changesAtClose = gpuChanges.size();
        const auto preparationsAtClose = runtime->findChildren<QObject *>("cpuRuntimePreparation").size();
        QSet<QProcess *> processesAtClose;
        for (auto *process : window_->findChildren<QProcess *>()) processesAtClose.insert(process);
        for (auto *process : runtime->findChildren<QProcess *>()) processesAtClose.insert(process);
        // Directly invoke the page intents: disabling a button alone must not
        // bypass the application permission guard through an already queued edit.
        QVERIFY(QMetaObject::invokeMethod(more, "cpuPrepareRequested", Qt::DirectConnection));
        QVERIFY(QMetaObject::invokeMethod(more, "gpuCheckRequested", Qt::DirectConnection));
        QVERIFY(QMetaObject::invokeMethod(more, "gpuPrepareRequested", Qt::DirectConnection));
        QCOMPARE(status->text(), statusAtClose); // CPU-ready callback must not run.
        QCOMPARE(gpuChanges.size(), changesAtClose);
        QCOMPARE(runtime->findChildren<QObject *>("cpuRuntimePreparation").size(), preparationsAtClose);
        QVERIFY(runtime->isShutdownReady()); QVERIFY(!gpu->isChecking()); QVERIFY(!gpu->isPreparing());
        for (auto *process : window_->findChildren<QProcess *>()) QVERIFY(processesAtClose.contains(process));
        for (auto *process : runtime->findChildren<QProcess *>()) QVERIFY(processesAtClose.contains(process));
        QVERIFY2(readyToDestroy(), qPrintable(failure_));
        // Linux proves the shared permission boundary and no new local helper;
        // it does not exercise Windows-only download, extraction or Job APIs.
        QVERIFY(runtime->isShutdownReady()); destroyReadyWindow();
    }

    void highResolutionAcceptedAutomaticSaveDrainsBeforeDestruction()
    {
        const auto large = area_->filePath("assets/high-resolution.png"); QVERIFY(texture({3072, 2048}).save(large));
        const auto sourceHash = QCryptographicHash::hash(contents(large), QCryptographicHash::Sha256);
        QVERIFY(importImages({large})); controls().autoExport->setChecked(true);
        const auto pendingAtClose = std::make_shared<int>(0);
        auto *detector = window_->findChild<vision::app::DetectionController *>(); QVERIFY(detector);
        connect(detector, &vision::app::DetectionController::imageResultReady, this,
            [this, pendingAtClose](const vision::InferenceResult &, quint64) {
            if (closeRequested_) return; *pendingAtClose = window_->pendingResultWrites(); beginClose();
        });
        controls().runButton->click();
        QVERIFY2(waitUntil([this] { return closeRequested_; }), qPrintable(failure_));
        QVERIFY(*pendingAtClose > 0); QVERIFY(closeMilliseconds_ < 500);
        QVERIFY2(readyToDestroy(), qPrintable(failure_)); QVERIFY(beats_ > beatsAtClose_);
        QCOMPARE(window_->pendingResultWrites(), 0);
        const QDir output(area_->filePath("data/results"));
        const auto jsonFiles = output.entryList({"*.json"}, QDir::Files);
        const auto pngFiles = output.entryList({"*.png"}, QDir::Files);
        const auto csvFiles = output.entryList({"*.csv"}, QDir::Files);
        QCOMPARE(jsonFiles.size(), 1); QCOMPARE(pngFiles.size(), 1); QCOMPARE(csvFiles.size(), 1);
        const auto metadata = QJsonDocument::fromJson(contents(output.filePath(jsonFiles.first()))).object();
        QCOMPARE(metadata.value("source").toString(), large); QVERIFY(!metadata.value("predictions").toArray().isEmpty());
        QCOMPARE(QImage(output.filePath(pngFiles.first())).size(), QSize(3072, 2048));
        QVERIFY(contents(output.filePath(csvFiles.first())).count('\n') >= 2);
        QCOMPARE(QCryptographicHash::hash(contents(large), QCryptographicHash::Sha256), sourceHash);
        const auto history = QJsonDocument::fromJson(contents(area_->filePath("data/history.json"))).array();
        QCOMPARE(history.size(), 1); QCOMPARE(history.first().toObject().value("source").toString(), large);
        destroyReadyWindow();
    }

    void importAndQueueProjectionCanCloseImmediately_data()
    {
        QTest::addColumn<bool>("duringProjection");
        QTest::newRow("background-file-validation") << false;
        QTest::newRow("partially-projected-large-queue") << true;
    }
    void importAndQueueProjectionCanCloseImmediately()
    {
        QFETCH(bool, duringProjection);
        const auto files = copies(1300); QCOMPARE(files.size(), 1300);
        const auto projectedAtClose = std::make_shared<int>(0);
        if (duringProjection)
            connect(page(), &WorkbenchPage::queueProjectionFinished, this, [this, projectedAtClose](quint64, int) {
                if (closeRequested_) return;
                *projectedAtClose = controls().queue->count(); beginClose();
            });
        QVERIFY(window_->inputs()->requestImages(files));
        if (!duringProjection) {
            QCOMPARE(window_->inputState().phase, vision::app::InputPhase::CheckingImages); beginClose();
        }
        QVERIFY2(waitUntil([this] { return closeRequested_; }), qPrintable(failure_));
        QVERIFY(closeMilliseconds_ < 500);
        if (duringProjection) QVERIFY(*projectedAtClose > 0 && *projectedAtClose < files.size());
        QVERIFY2(readyToDestroy(), qPrintable(failure_)); QVERIFY(beats_ > beatsAtClose_);
        QVERIFY(!QFileInfo::exists(window_->datasetSessionFilePath()));
        destroyReadyWindow();
    }

    void modelViewerLoadingCloseReapsHelperBeforeDestroyingWindow()
    {
        // Explicit lifecycle helper, never evidence for model parsing or inference.
        const auto helper = area_->filePath("viewer-lifecycle.py");
        writeFile(helper, "import signal,sys,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nwith open(sys.argv[-1]+'.started','w') as f: f.write('started')\ntime.sleep(30)\n");
        environment("VISION_STUDIO_NETRON_WORKER", helper.toUtf8());
        auto *viewer = window_->findChild<ModelViewer *>("modelStructureViewer"); QVERIFY(viewer);
        const auto loadingModel = area_->filePath("viewer-loading.onnx"); QVERIFY(QFile::copy(model_, loadingModel));
        viewer->stop(); window_->showModelStructure(loadingModel);
        QVERIFY2(waitUntil([&] { return QFileInfo::exists(loadingModel + ".started"); }), qPrintable(failure_));
        QCOMPARE(viewer->state(), ModelViewer::State::Loading);
        QPointer<QProcess> process;
        for (auto *candidate : viewer->findChildren<QProcess *>())
            if (candidate->arguments().contains(helper)) { process = candidate; break; }
        QVERIFY(process); QVERIFY(process->state() != QProcess::NotRunning);
        beginClose(); QVERIFY(closeMilliseconds_ < 500); QVERIFY(!window_->isShutdownReady());
        QVERIFY2(readyToDestroy(), qPrintable(failure_)); QVERIFY(beats_ > beatsAtClose_);
        QVERIFY(viewer->isShutdownReady()); QVERIFY(process.isNull()); destroyReadyWindow();
    }

    void conversionOutputSurvivesCloseWhileHelperCleanupFinishes()
    {
        // The helper exercises process/result/cleanup lifecycle only; its bytes
        // are deliberately not advertised as a converted model.
        const auto source = area_->filePath("lifecycle-source.pt"); writeFile(source, "original lifecycle fixture");
        const auto helper = area_->filePath("conversion-lifecycle.py");
        writeFile(helper, R"PY(import argparse,json,signal,time
p=argparse.ArgumentParser()
for name in ['source','format','output','image-size','opset','work-dir']: p.add_argument('--'+name)
a=p.parse_args()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
def emit(**e): print(json.dumps(dict(protocol=1,**e)),flush=True)
emit(type='status',message='prepared',job_temp=a.work_dir)
with open(a.output,'xb') as f: f.write(b'lifecycle fixture artifact')
emit(type='result',message='done',output=a.output,format=a.format,input_channels=3)
time.sleep(30)
)PY");
        environment("VISION_STUDIO_CONVERSION_WORKER", helper.toUtf8());
        auto *conversion = window_->findChild<ModelConversionPage *>("modelConversionPage"); QVERIFY(conversion);
        conversion->setSourceModel(source, "生命周期 fixture");
        window_->findChild<QPushButton *>("navigation-6")->click();
        conversion->findChild<QPushButton *>("conversionStartButton")->click();
        auto *summary = conversion->findChild<QLabel *>("conversionSummary"); QVERIFY(summary);
        QVERIFY2(waitUntil([&] { return summary->text().contains("FP32"); }), qPrintable(failure_));
        QVERIFY(conversion->isBusy());
        const auto directory = conversion->findChild<QLineEdit *>("conversionOutputDirectory")->text();
        const auto output = QDir(directory).filePath(conversion->findChild<QLineEdit *>("conversionOutputFilename")->text());
        const QPointer<QProcess> process(conversion->findChild<QProcess *>()); QVERIFY(process);
        QVERIFY(QFileInfo::exists(output));
        beginClose(); QVERIFY(closeMilliseconds_ < 500); QVERIFY(!window_->isShutdownReady());
        QVERIFY2(readyToDestroy(), qPrintable(failure_)); QVERIFY(beats_ > beatsAtClose_);
        QVERIFY(conversion->isShutdownReady()); QVERIFY(process.isNull());
        QCOMPARE(contents(output), QByteArray("lifecycle fixture artifact"));
        QCOMPARE(contents(source), QByteArray("original lifecycle fixture"));
        QVERIFY(QDir(directory).entryList({".vision-studio-convert-*"}, QDir::Dirs | QDir::Hidden | QDir::NoDotAndDotDot).isEmpty());
        destroyReadyWindow();
    }

    void sameApplicationCanRebuildWindowAndUsePreviewModelRuntimeAndInference()
    {
        QVERIFY(importImages({image_})); beginClose(); QVERIFY2(readyToDestroy(), qPrintable(failure_)); destroyReadyWindow();
        QVERIFY(!AsyncImagePreview::applicationPool()->shutdownRequested());
        createWindow();
        QVERIFY2(waitUntil([this] { return !window_->modelState().inspectionPending && window_->modelState().classCount == 1; }), qPrintable(failure_));
        QVERIFY(importImages({image_}));
        QVERIFY(!controls().canvas->result().image.isNull());
        QVERIFY(ui::ensureCpuRuntime(window_.get(), [] {}));
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        controls().runButton->click(); QCOMPARE(starts.size(), 1);
        QVERIFY2(waitUntil([this] { return !window_->isWorkbenchBusy() && window_->completedCount() == 1 && window_->pendingResultWrites() == 0; }), qPrintable(failure_));
        QVERIFY(!window_->detectionState().failed); QVERIFY(window_->displayedResult().backend.contains("OpenCV"));
        QVERIFY(!window_->displayedResult().predictions.isEmpty());
        window_->showModelStructure(model_);
        auto *viewer = window_->findChild<ModelViewer *>("modelStructureViewer"); QVERIFY(viewer);
        QVERIFY2(waitUntil([viewer] { return viewer->state() == ModelViewer::State::Ready; }), qPrintable(viewer->errorString()));
        beginClose(); QVERIFY2(readyToDestroy(), qPrintable(failure_)); destroyReadyWindow();
    }
};

int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication application(argc, argv); WindowShutdownTests tests;
    return QTest::qExec(&tests, argc, argv);
}
#include "window_shutdown_tests.moc"
