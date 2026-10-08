#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"
#include "ui/modelviewer.h"

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QImageWriter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPointer>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <algorithm>
#include <memory>

namespace
{
QPushButton *findButton(QWidget *parent, const QString &title)
{
    for (auto *button : parent->findChildren<QPushButton *>())
        if (button->text() == title)
            return button;
    return nullptr;
}

bool chooseFolder(QPushButton *button, const QString &path)
{
    if (!button)
        return false;
    bool selected = false, nameSet = false;
    QPointer<QFileDialog> active;
    QElapsedTimer elapsed;
    elapsed.start();
    QTimer selection;
    QObject::connect(&selection, &QTimer::timeout, button,
        [&]
        {
            auto *dialog = qobject_cast<QFileDialog *>(QApplication::activeModalWidget());
            if (!dialog)
                return;
            if (active != dialog)
            {
                active = dialog;
                dialog->setOption(QFileDialog::DontUseNativeDialog);
                dialog->setDirectory(QFileInfo(path).absolutePath());
                QObject::connect(dialog, &QDialog::finished, &selection,
                    [&](int result) { selected = result == QDialog::Accepted; selection.stop(); });
            }
            if (elapsed.elapsed() > 5000)
            {
                dialog->reject();
                return;
            }
            if (elapsed.elapsed() < 500)
                return;
            auto *name = dialog->findChild<QLineEdit *>(QStringLiteral("fileNameEdit"));
            if (!name)
                return;
            if (!nameSet)
            {
                name->setText(path);
                nameSet = true;
            }
            QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection);
        });
    selection.start(50);
    QTest::mouseClick(button, Qt::LeftButton);
    return selected;
}

QImage texturedImage(const QSize &size)
{
    QImage image(size, QImage::Format_RGB888);
    quint32 state = 0xA53189F1;
    for (int y = 0; y < size.height(); ++y)
        for (int x = 0; x < size.width() * 3; ++x)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            image.scanLine(y)[x] = uchar(state);
        }
    return image;
}

QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
} // namespace

class BatchResponsivenessTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> fixture_;
    std::unique_ptr<MainWindow> window_;
    std::unique_ptr<QTimer> dialogGuard_;
    QHash<QByteArray, QByteArray> oldEnvironment_;
    QHash<QByteArray, bool> hadEnvironment_;
    QString unexpectedDialog_;

    QStringList prepareFolder(const QSize &size)
    {
        const QString folder = fixture_->path() + QStringLiteral("/batch");
        if (!QDir().mkpath(folder))
            return {};
        const QString first = folder + QStringLiteral("/01.png");
        QImageWriter writer(first, "PNG");
        writer.setCompression(11);
        if (!writer.write(texturedImage(size)))
            return {};
        writer.device()->close();
        QStringList files{first};
        for (int row = 2; row <= 3; ++row)
        {
            const QString file = folder + QStringLiteral("/%1.png").arg(row, 2, 10, QChar('0'));
            if (!QFile::copy(first, file))
                return {};
            files.append(file);
        }
        if (!chooseFolder(findButton(window_.get(), QStringLiteral("文件夹")), folder))
            return {};
        return files;
    }

    QListWidget *queue() const
    {
        for (auto *list : window_->findChildren<QListWidget *>())
            if (list->viewMode() == QListView::IconMode)
                return list;
        return nullptr;
    }

    ImageCanvas *canvas() const
    {
        auto *page = window_->findChild<QWidget *>(QStringLiteral("workbenchPage"));
        return page ? page->findChild<ImageCanvas *>() : nullptr;
    }

  private slots:
    void initTestCase()
    {
        QApplication::setStyle(QStringLiteral("Fusion"));
        QApplication::setFont(QFont(QStringLiteral("Noto Sans CJK SC"), 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
        QCoreApplication::setApplicationVersion(QStringLiteral("2.0.0"));
    }

    void init()
    {
        for (const QByteArray &name : {QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"),
                                       QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR")})
        {
            oldEnvironment_.insert(name, qgetenv(name.constData()));
            hadEnvironment_.insert(name, qEnvironmentVariableIsSet(name.constData()));
        }
        fixture_ = std::make_unique<QTemporaryDir>();
        QVERIFY(fixture_->isValid());
        for (const QString &directory : {QStringLiteral("models"), QStringLiteral("assets"),
                                         QStringLiteral("scripts"), QStringLiteral("output")})
            QVERIFY(QDir().mkpath(fixture_->path() + "/" + directory));
        const QString project = QStringLiteral(VISION_PROJECT_DIR);
        for (const QString &path : {QStringLiteral("models/yolov5n.onnx"), QStringLiteral("assets/bus.jpg"),
                                    QStringLiteral("scripts/netron_server.py")})
            QVERIFY(QFile::copy(project + "/" + path, fixture_->path() + "/" + path));
        qputenv("VISION_STUDIO_HOME", fixture_->path().toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (fixture_->path() + QStringLiteral("/output")).toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (fixture_->path() + QStringLiteral("/absent-gpu")).toUtf8());
        QSettings settings(fixture_->path() + QStringLiteral("/output/preferences.ini"), QSettings::IniFormat);
        settings.setValue(QStringLiteral("computeDevice"), QStringLiteral("cpu"));
        settings.setValue(QStringLiteral("confidence"), 1.0);
        settings.setValue(QStringLiteral("autoExport"), true);
        settings.sync();
        unexpectedDialog_.clear();
        dialogGuard_ = std::make_unique<QTimer>();
        connect(dialogGuard_.get(), &QTimer::timeout, this,
            [this]
            {
                for (QWidget *widget : QApplication::topLevelWidgets())
                    if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible())
                    {
                        unexpectedDialog_ = dialog->text();
                        dialog->accept();
                    }
            });
        dialogGuard_->start(20);
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        auto *viewer = window_->findChild<ModelViewer *>(QStringLiteral("modelStructureViewer"));
        QVERIFY(viewer);
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, 30000);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
    }

    void cleanup()
    {
        window_.reset();
        dialogGuard_.reset();
        fixture_.reset();
        for (auto iterator = oldEnvironment_.cbegin(); iterator != oldEnvironment_.cend(); ++iterator)
            if (hadEnvironment_.value(iterator.key()))
                qputenv(iterator.key().constData(), iterator.value());
            else
                qunsetenv(iterator.key().constData());
        oldEnvironment_.clear();
        hadEnvironment_.clear();
    }

    void highResolutionAutomaticExportCanStopWithoutBlockingUi_data()
    {
        QTest::addColumn<QSize>("imageSize");
        QTest::newRow("1920x1080-rgb-random") << QSize(1920, 1080);
        QTest::newRow("3840x2160-rgb-random") << QSize(3840, 2160);
    }

    void highResolutionAutomaticExportCanStopWithoutBlockingUi()
    {
        QFETCH(QSize, imageSize);
        const auto files = prepareFolder(imageSize);
        QCOMPARE(files.size(), 3);
        auto *images = queue();
        auto *view = canvas();
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *stop = findButton(window_.get(), QStringLiteral("停止运行"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QVERIFY(images && view && run && stop && exportButton);
        images->setCurrentRow(2);
        QVector<vision::InferenceResult> accepted;
        bool scheduled = false, stopped = false;
        int acceptedAtStop = -1, ticks = 0, ticksAtStop = 0;
        qint64 maxGap = 0, lastTick = 0, dispatchDelay = -1;
        QElapsedTimer heartbeatTime, deliveryTime, stopTime;
        QTimer heartbeat;
        heartbeatTime.start();
        connect(&heartbeat, &QTimer::timeout, this,
            [&] { const qint64 now = heartbeatTime.elapsed(); maxGap = std::max(maxGap, now - lastTick);
                  lastTick = now; ++ticks; });
        heartbeat.start(10);
        QObject observer;
        const auto observation = connect(images, &QListWidget::currentRowChanged, &observer,
            [&](int row)
            {
                if (row < 0 || row >= files.size() || view->result().inferenceMs <= 0 ||
                    view->result().source != files[row])
                    return;
                accepted.append(view->result());
                if (!scheduled)
                {
                    scheduled = true;
                    deliveryTime.start();
                    QTimer::singleShot(30, &observer,
                        [&]
                        {
                            dispatchDelay = deliveryTime.elapsed();
                            acceptedAtStop = accepted.size();
                            ticksAtStop = ticks;
                            stopTime.start();
                            stopped = true;
                            QTest::mouseClick(stop, Qt::LeftButton);
                        });
                }
            });
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(stopped && run->isVisible(), 45000);
        const qint64 observedStopElapsed = stopTime.elapsed();
        QVERIFY(observedStopElapsed < 2000);
        QVERIFY(dispatchDelay < 500);
        QVERIFY(maxGap < 500);
        QVERIFY(accepted.size() > 0 && accepted.size() < files.size());
        QCOMPARE(accepted.size(), acceptedAtStop);
        QCOMPARE(starts.size(), 1);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        // Saving the accepted frame continues in the background after cancellation.
        QTRY_VERIFY_WITH_TIMEOUT(run->isEnabled(), 15000);
        QVERIFY(ticks > ticksAtStop);
        heartbeat.stop();
        disconnect(observation);
        const QByteArray savedHistory = contents(fixture_->path() + QStringLiteral("/output/history.json"));
        QCOMPARE(QJsonDocument::fromJson(savedHistory).array().size(), accepted.size());
        const QDir exported(fixture_->path() + QStringLiteral("/output/results"));
        QCOMPARE(exported.entryList({QStringLiteral("*.json")}, QDir::Files).size(), accepted.size());
        QCOMPARE(exported.entryList({QStringLiteral("*.png")}, QDir::Files).size(), accepted.size());
        QCOMPARE(exported.entryList({QStringLiteral("*.csv")}, QDir::Files).size(), accepted.size());
        for (const auto &result : accepted)
        {
            const auto metadataFiles = exported.entryList({QFileInfo(result.source).completeBaseName() +
                                                            QStringLiteral("-*.json")}, QDir::Files);
            QCOMPARE(metadataFiles.size(), 1);
            const QString base = metadataFiles[0].left(metadataFiles[0].size() - 5);
            const auto metadata = QJsonDocument::fromJson(contents(exported.filePath(metadataFiles[0]))).object();
            QCOMPARE(metadata.value(QStringLiteral("source")).toString(), result.source);
            QCOMPARE(metadata.value(QStringLiteral("actual_device")).toString(), QStringLiteral("cpu"));
            QCOMPARE(metadata.value(QStringLiteral("version")).toString(), QStringLiteral("2.0.0"));
            QCOMPARE(QImage(exported.filePath(base + QStringLiteral(".png"))).size(), imageSize);
            QVERIFY(!contents(exported.filePath(base + QStringLiteral(".csv"))).isEmpty());
            QVERIFY(QFile::remove(result.source));
        }
        images->setCurrentRow(2); // This image was not processed.
        QCOMPARE(view->result().inferenceMs, 0.0);
        QVERIFY(!exportButton->isEnabled());
        for (int row = 0; row < accepted.size(); ++row)
        {
            images->setCurrentRow(row);
            QCOMPARE(view->result().source, accepted[row].source);
            QCOMPARE(view->result().image, accepted[row].image);
            QCOMPARE(view->result().originalImage, accepted[row].originalImage);
            QCOMPARE(view->result().inferenceMs, accepted[row].inferenceMs);
            QVERIFY(exportButton->isEnabled());
        }
        QCOMPARE(contents(fixture_->path() + QStringLiteral("/output/history.json")), savedHistory);
        const QJsonObject evidence{{"width", imageSize.width()}, {"height", imageSize.height()},
            {"backend", accepted.first().backend}, {"device", "cpu"}, {"automatic_export", true},
            {"heartbeat_ticks", ticks}, {"maximum_heartbeat_gap_ms", maxGap},
            {"scheduled_stop_dispatch_ms", dispatchDelay}, {"observed_stop_elapsed_ms", observedStopElapsed},
            {"accepted_at_stop", acceptedAtStop}, {"accepted_after_stop", accepted.size()}, {"total_images", files.size()},
            {"cache_survives_removed_sources", true}, {"complete_exports", true}};
        qInfo().noquote() << "batch_responsiveness_evidence:" << QJsonDocument(evidence).toJson(QJsonDocument::Compact);
    }

    void automaticExportDiskFailureStopsWithoutDeadlockAndBatchCanRecover()
    {
        const auto files = prepareFolder(QSize(320, 240));
        QCOMPARE(files.size(), 3);
        auto *images = queue();
        auto *view = canvas();
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        auto *status = window_->findChild<QLabel *>(QStringLiteral("statusText"));
        QVERIFY(images && view && run && exportButton && status);
        images->setCurrentRow(2);
        bool blocked = false;
        vision::InferenceResult accepted;
        QObject observer;
        const QString destination = fixture_->path() + QStringLiteral("/output/results");
        const auto observation = connect(images, &QListWidget::currentRowChanged, &observer,
            [&](int row)
            {
                if (row != 0 || blocked || view->result().inferenceMs <= 0)
                    return;
                accepted = view->result();
                // Start passes the directory check; replace it only after the
                // actual first result, immediately before its async export begins.
                if (!QDir().rmdir(destination))
                    return;
                QFile blocker(destination);
                blocked = blocker.open(QIODevice::WriteOnly) && blocker.write("owned-test-blocker") == 18;
                blocker.close();
            });
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 15000);
        disconnect(observation);
        QVERIFY(blocked);
        QCOMPARE(accepted.source, files.first());
        QVERIFY(status->toolTip().contains(QStringLiteral("自动导出失败")));
        QCOMPARE(QJsonDocument::fromJson(contents(fixture_->path() + QStringLiteral("/output/history.json"))).array().size(), 1);
        QCOMPARE(contents(destination), QByteArray("owned-test-blocker"));
        images->setCurrentRow(2);
        QVERIFY(!exportButton->isEnabled());
        images->setCurrentRow(0);
        QCOMPARE(view->result().image, accepted.image);
        QVERIFY(exportButton->isEnabled());
        QCheckBox *automatic = nullptr;
        for (auto *check : window_->findChildren<QCheckBox *>())
            if (check->text().contains(QStringLiteral("自动保存")))
                automatic = check;
        QVERIFY(automatic);
        automatic->setChecked(false);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(run->isVisible() && run->isEnabled(), 15000);
        QCOMPARE(starts.size(), 2);
        QCOMPARE(QJsonDocument::fromJson(contents(fixture_->path() + QStringLiteral("/output/history.json"))).array().size(), 4);
        for (int row : {0, 1, 2})
        {
            images->setCurrentRow(row);
            QCOMPARE(view->result().source, files[row]);
            QVERIFY(view->result().inferenceMs > 0);
            QVERIFY(exportButton->isEnabled());
        }
        QCOMPARE(contents(destination), QByteArray("owned-test-blocker"));
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        qInfo().noquote() << "batch_disk_failure_evidence: export_error_reported=true no_deadlock=true completed_result_retained=true recovered_images=3";
    }
};

QTEST_MAIN(BatchResponsivenessTests)
#include "batch_responsiveness_tests.moc"
