// SPDX-License-Identifier: AGPL-3.0-only
#include "ui/asyncimagepreview.h"
#include "ui/modelconversionpage.h"
#include "ui/modelviewer.h"

#include <QApplication>
#include <QBuffer>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QLabel>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QRunnable>
#include <QScopeGuard>
#include <QSemaphore>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QThreadPool>
#include <QTimer>
#include <QtTest>
#include <atomic>
#include <memory>
#include <thread>
#ifndef Q_OS_WIN
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class ComponentShutdownTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir directory_;
    QString python_, viewerHelper_, converterHelper_, source_;
    QList<std::pair<QByteArray, QByteArray>> savedEnvironment_;
    QSet<QByteArray> missingEnvironment_;

    void write(const QString &path, const QByteArray &content)
    {
        QFile file(path); QVERIFY(file.open(QIODevice::WriteOnly)); QCOMPARE(file.write(content), content.size());
    }
    void overrideEnvironment(const QByteArray &key, const QByteArray &value)
    {
        if (!qEnvironmentVariableIsSet(key.constData())) missingEnvironment_.insert(key);
        savedEnvironment_.append({key, qgetenv(key.constData())});
        qputenv(key.constData(), value);
    }
    QStringList temporaryConversionDirectories() const
    {
        const QDir output(directory_.filePath("converted-models"));
        return output.entryList({".vision-studio-convert-*"}, QDir::Dirs | QDir::Hidden | QDir::NoDotAndDotDot);
    }
    void converter(const QByteArray &afterResult)
    {
        write(converterHelper_, R"PY(import argparse,json,signal,time
p=argparse.ArgumentParser()
for name in ['source','format','output','image-size','opset','work-dir']: p.add_argument('--'+name)
a=p.parse_args()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
def emit(**e): print(json.dumps(dict(protocol=1,**e)),flush=True)
emit(type='status',message='prepared',job_temp=a.work_dir)
with open(a.output,'xb') as f: f.write(b'accepted conversion artifact')
emit(type='result',message='done',output=a.output,format=a.format,input_channels=3)
)PY" + afterResult);
    }

#ifndef Q_OS_WIN
    struct FifoFixture
    {
        QString path;
        QByteArray image;
        std::thread writer;
        std::atomic_bool done{false};
        explicit FifoFixture(QString value) : path(std::move(value))
        {
            ::mkfifo(QFile::encodeName(path).constData(), 0600);
            QImage pixels(16, 16, QImage::Format_RGB888); pixels.fill(Qt::green);
            QBuffer buffer(&image); buffer.open(QIODevice::WriteOnly); pixels.save(&buffer, "PNG");
        }
        void release()
        {
            if (writer.joinable()) return;
            writer = std::thread([this] {
                for (int attempt = 0; attempt < 1500; ++attempt) {
                    const int fd = ::open(QFile::encodeName(path).constData(), O_WRONLY | O_NONBLOCK);
                    if (fd >= 0) { (void)::write(fd, image.constData(), size_t(image.size())); ::close(fd); break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                done.store(true);
            });
        }
        ~FifoFixture() { release(); writer.join(); }
    };
#endif

  private slots:
    void init()
    {
        QVERIFY(directory_.isValid());
        python_ = QStandardPaths::findExecutable("python3");
        if (python_.isEmpty()) python_ = QStandardPaths::findExecutable("python");
        QVERIFY(!python_.isEmpty());
        viewerHelper_ = directory_.filePath("viewer.py");
        converterHelper_ = directory_.filePath("converter.py");
        source_ = directory_.filePath("source.pt");
        write(source_, "fixture source, never modified");
        write(viewerHelper_, "import signal,sys,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
             "open(sys.argv[-1]+'.started','w').write('started')\ntime.sleep(30)\n");
        overrideEnvironment("VISION_STUDIO_PYTHON", python_.toUtf8());
        overrideEnvironment("VISION_STUDIO_NETRON_WORKER", viewerHelper_.toUtf8());
        overrideEnvironment("VISION_STUDIO_CONVERSION_WORKER", converterHelper_.toUtf8());
    }
    void cleanup()
    {
        for (const auto &entry : savedEnvironment_) {
            if (missingEnvironment_.contains(entry.first)) qunsetenv(entry.first.constData());
            else qputenv(entry.first.constData(), entry.second);
        }
        savedEnvironment_.clear(); missingEnvironment_.clear();
        QFile::remove(source_ + ".started");
        QDir(directory_.filePath("converted-models")).removeRecursively();
    }

    void viewerShutdownIncludesRetiredAndActiveHelpers()
    {
        ModelViewer viewer;
        viewer.openModel(source_);
        QTRY_VERIFY_WITH_TIMEOUT(QFileInfo::exists(source_ + ".started"), 3000);
        const QPointer<QProcess> first(viewer.findChild<QProcess *>()); QVERIFY(first);
        viewer.stop();
        QVERIFY(first && first->state() != QProcess::NotRunning);
        const auto secondSource = directory_.filePath("second.pt"); write(secondSource, "another fixture");
        viewer.openModel(secondSource);
        QTRY_VERIFY_WITH_TIMEOUT(QFileInfo::exists(secondSource + ".started"), 3000);
        QCOMPARE(viewer.findChildren<QProcess *>().size(), 2);
        QSignalSpy ready(&viewer, &ModelViewer::shutdownReady);
        int ticks = 0; QTimer heartbeat;
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++ticks; }); heartbeat.start(10);
        QElapsedTimer elapsed; elapsed.start(); viewer.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100); QVERIFY(!viewer.isShutdownReady()); QCOMPARE(ready.size(), 0);
        const auto selected = viewer.modelPath(); viewer.openModel(source_); QCOMPARE(viewer.modelPath(), selected);
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 5000);
        QVERIFY(viewer.isShutdownReady()); QVERIFY(first.isNull()); QVERIFY(ticks >= 3);
        QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        viewer.requestShutdown(); QTest::qWait(20); QCOMPARE(ready.size(), 1);
    }

    void viewerDirectDestructionLeavesAnAsynchronousReaper()
    {
        auto viewer = std::make_unique<ModelViewer>(); viewer->openModel(source_);
        QTRY_VERIFY_WITH_TIMEOUT(QFileInfo::exists(source_ + ".started"), 3000);
        const QPointer<QProcess> process(viewer->findChild<QProcess *>()); QVERIFY(process);
        QElapsedTimer elapsed; elapsed.start(); viewer.reset(); QVERIFY(elapsed.elapsed() < 100);
        QTRY_VERIFY_WITH_TIMEOUT(process.isNull(), 5000);
    }

    void conversionShutdownRetainsAcceptedOutputAndAwaitsCleanup()
    {
        converter("time.sleep(30)\n");
        ModelConversionPage page(directory_.path()); page.setSourceModel(source_);
        auto *start = page.findChild<QPushButton *>("conversionStartButton"); QVERIFY(start);
        QTest::mouseClick(start, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(page.findChild<QLabel *>("conversionSummary")->text().contains("FP32"), 3000);
        QVERIFY(page.isBusy()); QVERIFY(!temporaryConversionDirectories().isEmpty());
        QSignalSpy ready(&page, &ModelConversionPage::shutdownReady);
        QElapsedTimer elapsed; elapsed.start(); page.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100); QVERIFY(!page.isShutdownReady());
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 5000);
        QVERIFY(temporaryConversionDirectories().isEmpty()); QVERIFY(!start->isEnabled());
        QFile artifact(directory_.filePath("converted-models/source-640.onnx"));
        QVERIFY(artifact.open(QIODevice::ReadOnly)); QCOMPARE(artifact.readAll(), QByteArray("accepted conversion artifact"));
        QFile source(source_); QVERIFY(source.open(QIODevice::ReadOnly)); QCOMPARE(source.readAll(), QByteArray("fixture source, never modified"));
    }

    void conversionShutdownDoesNotOutrunQueuedGlobalCleanup()
    {
        converter("pass\n");
        auto *pool = QThreadPool::globalInstance(); const int previousMaximum = pool->maxThreadCount();
        const auto release = std::make_shared<QSemaphore>(); const auto started = std::make_shared<std::atomic_bool>(false);
        pool->setMaxThreadCount(1);
        const auto restore = qScopeGuard([&] { release->release(); pool->setMaxThreadCount(previousMaximum); });
        pool->start(QRunnable::create([release, started] { started->store(true); release->acquire(); }));
        QTRY_VERIFY_WITH_TIMEOUT(started->load(), 2000);
        ModelConversionPage page(directory_.path()); page.setSourceModel(source_);
        QTest::mouseClick(page.findChild<QPushButton *>("conversionStartButton"), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!page.isBusy(), 3000);
        QVERIFY(!temporaryConversionDirectories().isEmpty());
        QSignalSpy ready(&page, &ModelConversionPage::shutdownReady); page.requestShutdown();
        QTest::qWait(50); QCOMPARE(ready.size(), 0); QVERIFY(!page.isShutdownReady());
        release->release();
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 3000);
        QVERIFY(temporaryConversionDirectories().isEmpty());
    }

    void conversionOrphanIsReapedBeforeAnotherPageCanReportReady()
    {
        converter("time.sleep(30)\n");
        auto page = std::make_unique<ModelConversionPage>(directory_.path()); page->setSourceModel(source_);
        QTest::mouseClick(page->findChild<QPushButton *>("conversionStartButton"), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(page->findChild<QLabel *>("conversionSummary")->text().contains("FP32"), 3000);
        const QPointer<QProcess> process(page->findChild<QProcess *>()); QVERIFY(process);
        QElapsedTimer elapsed; elapsed.start(); page.reset(); QVERIFY(elapsed.elapsed() < 100);
        ModelConversionPage observer(directory_.path());
        QSignalSpy ready(&observer, &ModelConversionPage::shutdownReady); observer.requestShutdown();
        QVERIFY(!observer.isShutdownReady());
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 5000);
        QVERIFY(process.isNull()); QVERIFY(temporaryConversionDirectories().isEmpty());
    }

    void previewShutdownRejectsPendingAndFutureRequests()
    {
        AsyncImagePreview preview; QSignalSpy images(&preview, &AsyncImagePreview::ready);
        QSignalSpy ready(&preview, &AsyncImagePreview::shutdownReady);
        preview.request(1, source_, {}, vision::InputColorMode::Color); preview.requestShutdown();
        preview.request(2, source_, {}, vision::InputColorMode::Color);
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 2000);
        QVERIFY(preview.isShutdownReady()); QCOMPARE(images.size(), 0); QVERIFY(!preview.hasPendingRequest());
    }

    void previewShutdownWaitsForRealDiskReadWithoutBlocking()
    {
#ifdef Q_OS_WIN
        QSKIP("POSIX FIFO exercises an actually blocked image read; Windows pipe acceptance is separate.");
#else
        FifoFixture fifo(directory_.filePath("blocked-preview.fifo"));
        AsyncImagePreview preview; QSignalSpy ready(&preview, &AsyncImagePreview::shutdownReady);
        QSignalSpy images(&preview, &AsyncImagePreview::ready);
        preview.request(1, fifo.path, {}, vision::InputColorMode::Color);
        QTRY_COMPARE_WITH_TIMEOUT(preview.activeRequestCount(), 1, 2000); QTest::qWait(30);
        QElapsedTimer elapsed; elapsed.start(); preview.requestShutdown(); QVERIFY(elapsed.elapsed() < 100);
        QTest::qWait(30); QCOMPARE(ready.size(), 0); QVERIFY(!preview.isShutdownReady());
        fifo.release();
        QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 4000);
        QTRY_VERIFY_WITH_TIMEOUT(fifo.done.load(), 1000); QCOMPARE(images.size(), 0);
#endif
    }

    // Application shutdown is intentionally last: the global gate cannot reopen.
    void finalApplicationPreviewGateIncludesDestroyedInstances()
    {
        auto *pool = AsyncImagePreview::applicationPool();
        QSignalSpy ready(pool, &AsyncImagePreviewPool::shutdownReady);
#ifdef Q_OS_WIN
        pool->requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 2000);
#else
        FifoFixture fifo(directory_.filePath("orphan-preview.fifo"));
        auto preview = std::make_unique<AsyncImagePreview>();
        preview->request(1, fifo.path, {}, vision::InputColorMode::Color);
        QTRY_COMPARE_WITH_TIMEOUT(preview->activeRequestCount(), 1, 2000); QTest::qWait(30);
        preview.reset();
        pool->requestShutdown();
        QVERIFY(!pool->isShutdownReady()); QTest::qWait(30); QCOMPARE(ready.size(), 0);
        fifo.release(); QTRY_COMPARE_WITH_TIMEOUT(ready.size(), 1, 4000);
        QTRY_VERIFY_WITH_TIMEOUT(fifo.done.load(), 1000);
#endif
        QVERIFY(pool->isShutdownReady());
        AsyncImagePreview future; QSignalSpy images(&future, &AsyncImagePreview::ready);
        future.request(1, source_, {}, vision::InputColorMode::Color);
        QTRY_VERIFY_WITH_TIMEOUT(future.isShutdownReady(), 2000); QCOMPARE(images.size(), 0);
    }
};

QTEST_MAIN(ComponentShutdownTests)
#include "component_shutdown_tests.moc"
