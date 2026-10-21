#include "ui/mainwindow.h"
#include "core/gpuruntime.h"
#include "ui/imagecanvas.h"
#include "ui/modelviewer.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDateTime>
#include <QFileInfo>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QScrollArea>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#include <memory>

class ScopeFilterUiTests final : public QObject
{
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> area_;
    std::unique_ptr<MainWindow> window_;
    QHash<QByteArray, QByteArray> originalEnvironment_;
    QHash<QByteArray, bool> hadEnvironment_;
    QString unexpectedDialog_;
    QTimer dialogGuard_;
    QStringList files_;

    QStringList createQueue(int count)
    {
        window_->clearImageResults();
        window_->files_.clear();
        window_->queue_->clear();
        const QString folder = area_->path() + "/images";
        if (!QDir().mkpath(folder)) return {};
        QImage image(QStringLiteral(VISION_PROJECT_DIR) + "/assets/bus.jpg");
        if (image.isNull()) return {};
        image = image.scaledToWidth(320, Qt::SmoothTransformation);
        QStringList paths;
        for (int i = 0; i < count; ++i) {
            const QString path = folder + QString("/%1.png").arg(i + 1, 3, 10, QChar('0'));
            if (!image.save(path)) return {};
            paths.append(path);
        }
        window_->addFiles(paths);
        files_ = paths;
        return paths;
    }

    bool waitForCompleted(int expected, int timeout = 60000)
    {
        QElapsedTimer timer;
        timer.start();
        while (timer.elapsed() < timeout && unexpectedDialog_.isEmpty()) {
            QApplication::processEvents();
            if (!window_->busy_ && window_->completed_ == expected && !window_->resultIoPending_)
                return true;
            QTest::qWait(10);
        }
        return false;
    }

    bool waitForReview(const QString &source, bool processed, int timeout = 10000)
    {
        QElapsedTimer timer; timer.start();
        while (timer.elapsed() < timeout && unexpectedDialog_.isEmpty()) {
            QApplication::processEvents();
            const auto &result = window_->canvas_->result();
            if (result.source == source && !result.image.isNull() &&
                (result.inferenceMs > 0) == processed && window_->exportButton_->isEnabled() == processed)
                return true;
            QTest::qWait(5);
        }
        const auto &result = window_->canvas_->result();
        qWarning().noquote() << "async_review_wait_failure" << "expected_source" << source << "processed" << processed
                            << "actual_source" << result.source << "image_null" << result.image.isNull()
                            << "inference_ms" << result.inferenceMs << "export_enabled" << window_->exportButton_->isEnabled();
        return false;
    }

    void evidence(const QString &kind, const QJsonObject &details)
    {
        QJsonObject record = details;
        record.insert("kind", kind);
        record.insert("backend", window_->lastResult_.backend);
        record.insert("actual_device", vision::computeDeviceKey(window_->lastResult_.device));
        qInfo().noquote() << "detection_scope_evidence:" << QJsonDocument(record).toJson(QJsonDocument::Compact);
    }

  private slots:
    void initTestCase()
    {
        QApplication::setStyle("Fusion");
        QApplication::setFont(QFont("Noto Sans CJK SC", 10));
        QCoreApplication::setApplicationVersion("2.2.0");
        connect(&dialogGuard_, &QTimer::timeout, this, [this] {
            for (QWidget *widget : QApplication::topLevelWidgets())
                if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible()) {
                    unexpectedDialog_ = dialog->text();
                    dialog->accept();
                }
        });
        dialogGuard_.setInterval(20);
    }

    void init()
    {
        for (const QByteArray &name : {QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"),
                                      QByteArray("VISION_STUDIO_GPU_RUNTIME_DIR")}) {
            originalEnvironment_[name] = qgetenv(name.constData());
            hadEnvironment_[name] = qEnvironmentVariableIsSet(name.constData());
        }
        area_ = std::make_unique<QTemporaryDir>();
        QVERIFY(area_->isValid());
        qputenv("VISION_STUDIO_HOME", QStringLiteral(VISION_PROJECT_DIR).toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", area_->path().toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", (area_->path() + "/absent-gpu").toUtf8());
        QSettings settings(area_->path() + "/preferences.ini", QSettings::IniFormat);
        settings.setValue("computeDevice", "cpu");
        settings.setValue("autoExport", false);
        settings.setValue("confidence", .25);
        settings.sync();
        unexpectedDialog_.clear();
        dialogGuard_.start();
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        window_->modelViewer_->stop();
        QCOMPARE(window_->detectionRange_->currentIndex(), 0);
        QVERIFY(window_->classFilter_->text().isEmpty());
    }

    void cleanup()
    {
        window_.reset();
        dialogGuard_.stop();
        area_.reset();
        for (auto it = originalEnvironment_.cbegin(); it != originalEnvironment_.cend(); ++it)
            if (hadEnvironment_.value(it.key())) qputenv(it.key().constData(), it.value());
            else qunsetenv(it.key().constData());
        originalEnvironment_.clear();
        hadEnvironment_.clear();
    }

    void fixedRangeAndProgress_data()
    {
        QTest::addColumn<int>("count");
        QTest::addColumn<int>("mode");
        QTest::addColumn<int>("selection");
        for (int mode = 0; mode < 3; ++mode)
            for (int row = 0; row < 3; ++row)
                QTest::newRow(qPrintable(QString("mode-%1-row-%2").arg(mode).arg(row))) << 3 << mode << row;
        for (int mode = 0; mode < 3; ++mode) {
            QTest::newRow(qPrintable(QString("single-mode-%1").arg(mode))) << 1 << mode << 0;
            QTest::newRow(qPrintable(QString("sixty-one-mode-%1").arg(mode))) << 61 << mode << 50;
        }
    }

    void fixedRangeAndProgress()
    {
        QFETCH(int, count); QFETCH(int, mode); QFETCH(int, selection);
        QCOMPARE(createQueue(count).size(), count);
        window_->queue_->setCurrentRow(selection);
        window_->detectionRange_->setCurrentIndex(mode);
        const auto expected = mode == 0 ? files_ : mode == 1 ? files_.mid(selection) : QStringList{files_[selection]};
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QVERIFY(window_->runButton_->isEnabled());
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QCOMPARE(starts.size(), 1);
        const auto request = qvariant_cast<vision::JobRequest>(starts[0][0]);
        QCOMPARE(request.files, expected);
        QVERIFY(!window_->detectionRange_->isEnabled());
        QVERIFY(!window_->queue_->isEnabled());
        // A non-UI caller must not mutate the submitted input either.
        window_->addFiles({QStringLiteral(VISION_PROJECT_DIR) + "/assets/bus.jpg"});
        QCOMPARE(window_->files_, files_);
        QVERIFY2(waitForCompleted(expected.size()), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->progress_->maximum(), expected.size());
        QCOMPARE(window_->progress_->value(), expected.size());
        QCOMPARE(window_->imageResults_.size(), expected.size());
        for (int row = 0; row < count; ++row) {
            window_->queue_->setCurrentRow(row);
            const bool processed = expected.contains(files_[row]);
            QVERIFY2(waitForReview(files_[row], processed), qPrintable(unexpectedDialog_));
            QCOMPARE(window_->canvas_->result().source, files_[row]);
            QCOMPARE(window_->canvas_->result().inferenceMs > 0, processed);
            QCOMPARE(window_->exportButton_->isEnabled(), processed);
            QCOMPARE(window_->imageResults_.contains(files_[row]), processed);
        }
        evidence("range", {{"queue_count", count}, {"mode", mode}, {"selected_row", selection},
                             {"submitted_count", expected.size()}, {"completed", window_->completed_},
                             {"first_queue_row", files_.indexOf(expected.first())},
                             {"last_queue_row", files_.indexOf(expected.last())},
                             {"range_outside_unprocessed", true}, {"review_matches_selection", true}});
    }

    void filtersAndScope_data()
    {
        QTest::addColumn<int>("mode"); QTest::addColumn<QString>("filter");
        for (int mode = 0; mode < 3; ++mode)
            for (const QString &filter : {QString(), QString("32"), QString("0,5,32"), QString("5，0 5，32")})
                QTest::newRow(qPrintable(QString("mode-%1-filter-%2").arg(mode).arg(filter.isEmpty() ? "all" : filter))) << mode << filter;
    }

    void filtersAndScope()
    {
        QFETCH(int, mode); QFETCH(QString, filter);
        QCOMPARE(createQueue(3).size(), 3);
        window_->queue_->setCurrentRow(1);
        window_->detectionRange_->setCurrentIndex(mode);
        window_->classFilter_->setText(filter);
        QTRY_VERIFY_WITH_TIMEOUT(window_->runButton_->isEnabled(), 15000);
        QVector<int> ids; QVERIFY(vision::parseClassFilter(filter, &ids));
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        const int expected = mode == 0 ? 3 : mode == 1 ? 2 : 1;
        QVERIFY2(waitForCompleted(expected), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->lastConfig_.classFilter, ids);
        QCOMPARE(window_->lastConfig_.modelClassCount, 80);
        QCOMPARE(window_->imageResults_.size(), expected);
        for (const QString &path : window_->taskRangeFiles_) {
            window_->queue_->setCurrentRow(files_.indexOf(path));
            QVERIFY2(waitForReview(path, true), qPrintable(unexpectedDialog_));
            const auto result = window_->canvas_->result();
            for (const auto &prediction : result.predictions)
                QVERIFY(ids.isEmpty() || ids.contains(prediction.classId));
            if (ids == QVector<int>{32}) QVERIFY(result.predictions.isEmpty());
            else QVERIFY(!result.predictions.isEmpty());
            QCOMPARE(window_->predictionTable_->rowCount(), result.predictions.size());
            const QString target = area_->path() + "/manual";
            QVERIFY(QDir().mkpath(target));
            QString error; QVERIFY2(window_->writeResult(result, target, &error), qPrintable(error));
            const auto jsonFiles = QDir(target).entryList({"*.json"}, QDir::Files);
            QVERIFY(!jsonFiles.isEmpty());
            QFile file(target + '/' + jsonFiles.last()); QVERIFY(file.open(QIODevice::ReadOnly));
            const auto exported = QJsonDocument::fromJson(file.readAll()).object();
            QCOMPARE(exported["source"].toString(), result.source);
            QCOMPARE(exported["predictions"].toArray().size(), result.predictions.size());
            QCOMPARE(exported["config"].toObject()["class_filter"].toArray().size(), ids.size());
            for (const auto &value : exported["predictions"].toArray())
                QVERIFY(ids.isEmpty() || ids.contains(value.toObject()["class_id"].toInt()));
        }
        evidence("scope-and-filter", {{"mode", mode}, {"filter_text", filter}, {"completed", expected},
                                      {"original_ids_retained", true}, {"export_matches_preview", true},
                                      {"zero_match_is_completed", ids == QVector<int>{32}}});
    }

    void selectionRequiredAndStreamingVisibility()
    {
        QCOMPARE(createQueue(3).size(), 3);
        window_->queue_->setCurrentRow(-1);
        for (int mode : {1, 2}) {
            window_->detectionRange_->setCurrentIndex(mode);
            QVERIFY(!window_->runButton_->isEnabled());
            QVERIFY(window_->rangeHint_->text().contains("选中"));
            QSignalSpy starts(window_.get(), &MainWindow::startRequested);
            window_->startInference();
            QCOMPARE(starts.size(), 0);
        }
        window_->detectionRange_->setCurrentIndex(0);
        QVERIFY(window_->runButton_->isEnabled());
        window_->sourceKind_ = vision::SourceKind::Video;
        window_->updateSourceUi();
        QVERIFY(window_->detectionRangePanel_->isHidden());
        window_->sourceKind_ = vision::SourceKind::Camera;
        window_->updateSourceUi();
        QVERIFY(window_->detectionRangePanel_->isHidden());
        window_->sourceKind_ = vision::SourceKind::Images;
        window_->updateSourceUi();
        QVERIFY(!window_->detectionRangePanel_->isHidden());
        evidence("selection-and-stream", {{"missing_selection_blocks_tail_and_current", true},
                                         {"whole_queue_works_without_selection", true}, {"stream_range_hidden", true}});
    }

    void invalidFilterAndModelChange()
    {
        QCOMPARE(createQueue(1).size(), 1);
        QTRY_VERIFY_WITH_TIMEOUT(!window_->modelInspectionPending_, 15000);
        QCOMPARE(window_->modelClassCount_, 80);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        for (const QString &invalid : {QString("-1"), QString("1.2"), QString("person"), QString("80"), QString("99999999999999")}) {
            window_->classFilter_->setText(invalid);
            QVERIFY(!window_->runButton_->isEnabled());
            QVERIFY(!window_->classFilterError_.isEmpty());
            window_->startInference();
            QCOMPARE(starts.size(), 0);
        }
        window_->classFilter_->setText("32");
        QVERIFY(window_->runButton_->isEnabled());
        const QString tiny = area_->path() + "/one-class.onnx";
        QProcess create;
        const QString python = qEnvironmentVariable("VISION_STUDIO_PYTHON", "/opt/VisionStudio/runtime/bin/python");
        create.start(python, {"-c", R"PY(
import sys, numpy as np, onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
w=np.zeros((5,3,1,1),np.float32); b=np.array([320,320,120,120,.9],np.float32)
g=h.make_graph([h.make_node('Conv',['images','w','b'],['conv'],kernel_shape=[1,1]),h.make_node('GlobalAveragePool',['conv'],['pool']),h.make_node('Reshape',['pool','shape'],['output'])],'one-class',[h.make_tensor_value_info('images',T.FLOAT,[1,3,32,32])],[h.make_tensor_value_info('output',T.FLOAT,[1,5,1])],[n.from_array(w,'w'),n.from_array(b,'b'),n.from_array(np.array([1,5,1],np.int64),'shape')])
m=h.make_model(g,opset_imports=[h.make_opsetid('',12)],ir_version=7)
onnx.save(m,sys.argv[1])
)PY", tiny});
        QVERIFY2(create.waitForFinished(15000), qPrintable(create.errorString()));
        QVERIFY2(create.exitCode() == 0, create.readAllStandardError().constData());
        window_->setModel(tiny);
        window_->modelViewer_->stop();
        window_->taskBox_->setCurrentIndex(int(vision::ModelTask::YoloV8));
        QTRY_VERIFY_WITH_TIMEOUT(!window_->modelInspectionPending_, 15000);
        QCOMPARE(window_->modelClassCount_, 1);
        QVERIFY(!window_->runButton_->isEnabled());
        QVERIFY(window_->classFilterHint_->text().contains("0～0"));
        window_->classFilter_->setText("0");
        QVERIFY(window_->runButton_->isEnabled());
        window_->taskBox_->setCurrentIndex(int(vision::ModelTask::Classification));
        QVERIFY(!window_->classFilter_->isEnabled());
        QVERIFY(window_->classFilterHint_->text().contains("分类"));
        QVERIFY(window_->currentConfig().classFilter.isEmpty());
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        evidence("validation", {{"invalid_inputs_block_start", true}, {"coco_classes", 80},
                                {"replacement_model_classes", 1}, {"classification_filter_disabled", true}});
    }

    void validCacheOutsideRangeAndChangedParameters()
    {
        QCOMPARE(createQueue(3).size(), 3);
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(3), qPrintable(unexpectedDialog_));
        const auto original = window_->imageResults_[files_[0]].result;
        window_->queue_->setCurrentRow(1);
        window_->detectionRange_->setCurrentIndex(2);
        const auto stopBeforeFirst = connect(window_.get(), &MainWindow::startRequested, this,
                                             [this] { window_->stopInference(); });
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(0), qPrintable(unexpectedDialog_));
        disconnect(stopBeforeFirst);
        QCOMPARE(window_->imageResults_.size(), 3);
        window_->queue_->setCurrentRow(0);
        QVERIFY2(waitForReview(files_[0], true), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->canvas_->result().inferenceMs, original.inferenceMs);
        QVERIFY(window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(1);
        QVERIFY2(waitForReview(files_[1], true), qPrintable(unexpectedDialog_));
        QVERIFY(window_->canvas_->result().inferenceMs > 0);
        QVERIFY(window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(2);
        QVERIFY2(waitForReview(files_[2], true), qPrintable(unexpectedDialog_));
        QVERIFY(window_->canvas_->result().inferenceMs > 0);
        window_->confidence_->setValue(.65);
        QVERIFY(window_->imageResults_.isEmpty());
        for (int row : {0, 2}) {
            window_->queue_->setCurrentRow(row);
            QVERIFY2(waitForReview(files_[row], false), qPrintable(unexpectedDialog_));
            QCOMPARE(window_->canvas_->result().inferenceMs, 0.0);
            QVERIFY(!window_->exportButton_->isEnabled());
        }
        evidence("cache-range", {{"outside_range_kept_after_cancel", true}, {"same_configuration_cancel_keeps_accepted_results", true},
                                  {"changed_parameters_invalidate", true}});
    }

    void stoppedRangePreservesCompletedResultsAndExport()
    {
        QCOMPARE(createQueue(61).size(), 61);
        window_->queue_->setCurrentRow(50);
        window_->detectionRange_->setCurrentIndex(1);
        window_->classFilter_->setText("0,5");
        QTRY_VERIFY_WITH_TIMEOUT(window_->runButton_->isEnabled(), 15000);
        window_->autoExport_->setChecked(true);
        int observed = 0;
        const auto stopAfterFirst = connect(window_->queue_, &QListWidget::currentRowChanged, this, [this, &observed](int row) {
            if (window_->busy_ && window_->lastResult_.inferenceMs > 0 && row >= 50) {
                ++observed;
                QTimer::singleShot(0, window_.get(), [this] { window_->stopInference(); });
            }
        });
        // Select a different row before running so the first accepted result emits a row change.
        // The selected range must still start at row 50 at click time.
        window_->queue_->setCurrentRow(49);
        window_->queue_->setCurrentRow(50);
        QElapsedTimer elapsed;
        elapsed.start();
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(window_->completed_ > 0, 45000);
        if (window_->busy_) window_->stopInference();
        QTRY_VERIFY_WITH_TIMEOUT(!window_->busy_, 2000);
        disconnect(stopAfterFirst);
        QVERIFY(window_->completed_ > 0 && window_->completed_ < 11);
        const int done = window_->completed_;
        QTRY_VERIFY_WITH_TIMEOUT(!window_->resultIoPending_, 15000);
        QTest::qWait(150);
        QCOMPARE(window_->completed_, done);
        QCOMPARE(window_->progress_->maximum(), 11);
        QCOMPARE(window_->progress_->value(), done);
        const QDir exported(area_->path() + "/results");
        QCOMPARE(exported.entryList({"*.json"}, QDir::Files).size(), done);
        for (int row = 0; row < 61; ++row) {
            window_->queue_->setCurrentRow(row);
            const bool processed = row >= 50 && row < 50 + done;
            QVERIFY2(waitForReview(files_[row], processed), qPrintable(unexpectedDialog_));
            QCOMPARE(window_->canvas_->result().inferenceMs > 0, processed);
            QCOMPARE(window_->exportButton_->isEnabled(), processed);
        }
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        evidence("stopped-range", {{"queue_count", 61}, {"range_count", 11}, {"completed", done},
                                    {"no_followup_after_stop", true}, {"auto_export_count", done},
                                    {"review_correct", true}, {"total_observed_ms", elapsed.elapsed()}});
    }

    void dynamicClassMetadataResolvedByActualInference()
    {
        QCOMPARE(createQueue(1).size(), 1);
        const QString model = area_->path() + "/dynamic-classes-yolov8.onnx";
        QProcess create;
        create.start(qEnvironmentVariable("VISION_STUDIO_PYTHON", "/opt/VisionStudio/runtime/bin/python"),
            {"-c", R"PY(
import sys, numpy as np, onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
w=np.zeros((5,3,1,1),np.float32); b=np.array([16,16,12,12,.9],np.float32)
g=h.make_graph([h.make_node('Conv',['images','w','b'],['conv'],kernel_shape=[1,1]),h.make_node('GlobalAveragePool',['conv'],['pool']),h.make_node('Reshape',['pool','shape'],['output'])],'dynamic-class',[h.make_tensor_value_info('images',T.FLOAT,[1,3,32,32])],[h.make_tensor_value_info('output',T.FLOAT,[1,'classes_plus_four',1])],[n.from_array(w,'w'),n.from_array(b,'b'),n.from_array(np.array([1,5,1],np.int64),'shape')])
onnx.save(h.make_model(g,opset_imports=[h.make_opsetid('',12)],ir_version=7),sys.argv[1])
)PY", model});
        QVERIFY(create.waitForFinished(15000));
        QVERIFY2(create.exitCode() == 0, create.readAllStandardError().constData());
        window_->setModel(model);
        window_->modelViewer_->stop();
        window_->inputSize_->setValue(32);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        // Intentionally start before the queued metadata completion is handled.
        // Empty filtering remains valid while the output's class axis is unknown.
        window_->startInference();
        QCOMPARE(starts.size(), 1);
        QCOMPARE(qvariant_cast<vision::JobRequest>(starts[0][0]).config.modelClassCount, 0);
        QVERIFY2(waitForCompleted(1), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->lastResult_.modelClassCount, 1);
        QCOMPARE(window_->modelClassCount_, 1);
        QCOMPARE(window_->lastConfig_.modelClassCount, 1);
        window_->classFilter_->setText("0");
        QVERIFY(window_->runButton_->isEnabled());
        window_->classFilter_->setText("1");
        QVERIFY(!window_->runButton_->isEnabled());
        evidence("dynamic-class-metadata", {{"submitted_class_count", 0}, {"actual_class_count", 1},
                                            {"ui_and_export_configuration_synchronized", true}});
    }

    void workbenchScopeFilterVisualArtifact()
    {
        window_->resize(1580, 960);
        QCOMPARE(createQueue(64).size(), 64);
        window_->queue_->setCurrentRow(50);
        window_->detectionRange_->setCurrentIndex(2);
        window_->classFilter_->setText("0,5");
        QTRY_VERIFY_WITH_TIMEOUT(window_->runButton_->isEnabled(), 15000);
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(1), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->canvas_->result().source, files_[50]);
        QVERIFY(!window_->canvas_->result().predictions.isEmpty());
        const QString artifactDirectory = qEnvironmentVariable("VISION_STUDIO_QA_ARTIFACTS");
        QString screenshot;
        if (!artifactDirectory.isEmpty()) {
            QVERIFY2(QDir::isAbsolutePath(artifactDirectory), "QA artifact directory must be explicitly absolute.");
            QVERIFY(QDir().mkpath(artifactDirectory));
            screenshot = QDir(artifactDirectory).filePath("scope-filter-workbench.png");
            QVERIFY2(!QFileInfo::exists(screenshot), "Refusing to overwrite an existing QA screenshot.");
            if (auto *scroll = window_->findChild<QScrollArea *>("configurationScrollArea"))
                scroll->ensureWidgetVisible(window_->classFilter_);
            QApplication::processEvents();
            QTest::qWait(150);
            window_->saveScreenshot(screenshot);
            QVERIFY(QFileInfo(screenshot).size() > 10000);
        }
        evidence("workbench-visual", {{"queue_count", 64}, {"selected_row", 50}, {"mode", 2},
                                     {"filter", "0,5"}, {"actual_inference", true},
                                     {"optional_screenshot", screenshot}});
    }

    void externalModelReplacementDuringTaskCannotRelabelCache()
    {
        QCOMPARE(createQueue(3).size(), 3);
        const QString mutableModel = area_->path() + "/mutable-yolov5n.onnx";
        QVERIFY(QFile::copy(QStringLiteral(VISION_PROJECT_DIR) + "/models/yolov5n.onnx", mutableModel));
        window_->setModel(mutableModel);
        window_->modelViewer_->stop();
        window_->queue_->setCurrentRow(2);
        const QString originalIdentity = window_->modelFileIdentity();
        QString pinnedFingerprint;
        bool replaced = false, verifiedLaterFrame = false;
        const auto replacement = connect(window_->queue_, &QListWidget::currentRowChanged, this,
            [this, &replaced, &verifiedLaterFrame, &pinnedFingerprint, mutableModel](int row) {
                if (!window_->busy_ || window_->lastResult_.inferenceMs <= 0) return;
                if (!replaced && row == 0) {
                    pinnedFingerprint = window_->imageConfigFingerprint_;
                    QFile file(mutableModel);
                    if (!file.open(QIODevice::ReadWrite)) return;
                    replaced = file.setFileTime(QDateTime::currentDateTime().addSecs(17), QFileDevice::FileModificationTime);
                } else if (replaced && row > 0) {
                    verifiedLaterFrame = window_->imageConfigFingerprint_ == pinnedFingerprint;
                }
            });
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(3), qPrintable(unexpectedDialog_));
        disconnect(replacement);
        QVERIFY(replaced);
        QVERIFY(verifiedLaterFrame);
        QVERIFY(originalIdentity != window_->modelFileIdentity());
        QCOMPARE(window_->activeModelIdentity_, originalIdentity);
        // The current file now represents another model identity. Accepted
        // in-memory results from the previous identity must not be reused.
        QVERIFY(window_->imageResults_.isEmpty());
        for (int row : {0, 1, 2}) {
            window_->queue_->setCurrentRow(row);
            QVERIFY2(waitForReview(files_[row], false), qPrintable(unexpectedDialog_));
            QCOMPARE(window_->canvas_->result().inferenceMs, 0.0);
            QVERIFY(!window_->exportButton_->isEnabled());
        }
        evidence("model-identity-during-task", {{"identity_changed", true}, {"job_identity_frozen", true},
                                              {"later_results_not_relabelled", true}, {"stale_cache_unavailable", true}});
    }

    void submittedSourceIdentityIsNotReplacedByLaterDiskVersion()
    {
        QCOMPARE(createQueue(2).size(), 2);
        QString submittedIdentity;
        const auto replacement = connect(window_.get(), &MainWindow::startRequested, this, [this, &submittedIdentity] {
            submittedIdentity = window_->taskSourceIdentities_.value(files_[0]);
            QImage replacement(79, 61, QImage::Format_RGB888); replacement.fill(Qt::magenta);
            QVERIFY(replacement.save(files_[0]));
        });
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(2), qPrintable(unexpectedDialog_));
        disconnect(replacement);
        QVERIFY(!submittedIdentity.isEmpty());
        QCOMPARE(window_->imageResults_[files_[0]].sourceIdentity, submittedIdentity);
        window_->queue_->setCurrentRow(0);
        QVERIFY2(waitForReview(files_[0], false), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->canvas_->result().image.size(), QSize(79, 61));
        QCOMPARE(window_->canvas_->result().image.pixelColor(0, 0), QColor(Qt::magenta));
        QCOMPARE(window_->canvas_->result().inferenceMs, 0.0);
        QVERIFY(!window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(1);
        QVERIFY2(waitForReview(files_[1], true), qPrintable(unexpectedDialog_));
        QVERIFY(window_->canvas_->result().inferenceMs > 0);
        evidence("source-identity-during-task", {{"submitted_identity_frozen", true},
                                               {"changed_source_cache_unavailable", true},
                                               {"unchanged_source_review_available", true}});
    }

    void equivalentFiltersAndChangedSourceFile()
    {
        QCOMPARE(createQueue(2).size(), 2);
        window_->classFilter_->setText("0,5");
        QTRY_VERIFY_WITH_TIMEOUT(window_->runButton_->isEnabled(), 15000);
        QTest::mouseClick(window_->runButton_, Qt::LeftButton);
        QVERIFY2(waitForCompleted(2), qPrintable(unexpectedDialog_));
        window_->classFilter_->setText("5，0 5");
        QCOMPARE(window_->imageResults_.size(), 2);
        window_->queue_->setCurrentRow(0);
        QVERIFY2(waitForReview(files_[0], true), qPrintable(unexpectedDialog_));
        QVERIFY(window_->canvas_->result().inferenceMs > 0);
        QImage replacement(61, 79, QImage::Format_RGB888); replacement.fill(Qt::cyan);
        QVERIFY(replacement.save(files_[0]));
        window_->queue_->setCurrentRow(1);
        window_->queue_->setCurrentRow(0);
        QVERIFY2(waitForReview(files_[0], false), qPrintable(unexpectedDialog_));
        QCOMPARE(window_->canvas_->result().image.size(), replacement.size());
        QCOMPARE(window_->canvas_->result().image.convertToFormat(QImage::Format_RGB888), replacement);
        QCOMPARE(window_->canvas_->result().inferenceMs, 0.0);
        QVERIFY(!window_->exportButton_->isEnabled());
        window_->queue_->setCurrentRow(1);
        QVERIFY2(waitForReview(files_[1], true), qPrintable(unexpectedDialog_));
        QVERIFY(window_->canvas_->result().inferenceMs > 0);
        window_->classFilter_->setText("32");
        QVERIFY(window_->imageResults_.isEmpty());
        evidence("cache-identity", {{"equivalent_filter_keeps_cache", true},
                                    {"changed_source_rejects_old_result", true}, {"changed_filter_invalidates", true}});
    }
};

QTEST_MAIN(ScopeFilterUiTests)
#include "scope_filter_ui_tests.moc"
