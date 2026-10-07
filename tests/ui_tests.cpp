#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"
#include "ui/modelviewer.h"

#include <QAbstractButton>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardItemModel>
#include <QTableView>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QTreeView>
#include <QUrl>
#include <QWebEnginePage>
#include <QWebEngineView>
#include <algorithm>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace
{
QPushButton *findButton(QWidget *parent, const QString &text)
{
    for (auto *button : parent->findChildren<QPushButton *>())
        if (button->text() == text)
            return button;
    return nullptr;
}

ImageCanvas *workbenchCanvas(QWidget *parent)
{
    auto *page = parent->findChild<QWidget *>(QStringLiteral("workbenchPage"));
    return page ? page->findChild<ImageCanvas *>() : nullptr;
}

QTableWidget *predictionTable(QWidget *parent)
{
    for (auto *table : parent->findChildren<QTableWidget *>())
        if (table->columnCount() == 3 && table->horizontalHeaderItem(0) &&
            table->horizontalHeaderItem(0)->text() == QStringLiteral("类别"))
            return table;
    return nullptr;
}

QListWidget *inputQueue(QWidget *parent)
{
    for (auto *list : parent->findChildren<QListWidget *>())
        if (list->viewMode() == QListView::IconMode)
            return list;
    return nullptr;
}

QImage renderCanvas(ImageCanvas *canvas)
{
    QPixmap image(canvas->size());
    canvas->render(&image);
    return image.toImage();
}

QPoint predictionCenter(ImageCanvas *canvas, int index)
{
    const auto &result = canvas->result();
    const qreal scale = std::min(qreal(canvas->width() - 64) / result.image.width(),
                                 qreal(canvas->height() - 64) / result.image.height());
    const QPointF origin((canvas->width() - result.image.width() * scale) / 2,
                         (canvas->height() - result.image.height() * scale) / 2);
    return (origin + result.predictions[index].box.center() * scale).toPoint();
}

bool dropFile(ImageCanvas *canvas, const QString &file)
{
    QMimeData mime;
    mime.setUrls({QUrl::fromLocalFile(file)});
    const QPoint point = canvas->rect().center();
    QDragEnterEvent enter(point, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(canvas, &enter);
    if (!enter.isAccepted())
        return false;
    QDropEvent drop(point, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(canvas, &drop);
    return drop.isAccepted();
}

QCheckBox *autoExportControl(QWidget *parent)
{
    for (auto *check : parent->findChildren<QCheckBox *>())
        if (check->text() == QStringLiteral("自动保存每张图片的结果"))
            return check;
    return nullptr;
}

bool activateModel(MainWindow *window, const QString &fileName)
{
    auto *nav = findButton(window, QStringLiteral("模型库"));
    auto *activate = findButton(window, QStringLiteral("前往检测工作台"));
    if (!nav || !activate)
        return false;
    QTest::mouseClick(nav, Qt::LeftButton);
    for (auto *list : window->findChildren<QListWidget *>())
        for (int row = 0; row < list->count(); ++row)
            if (QFileInfo(list->item(row)->data(Qt::UserRole).toString()).fileName() == fileName)
            {
                list->setCurrentRow(row);
                QTest::mouseClick(activate, Qt::LeftButton);
                return true;
            }
    return false;
}

bool chooseDialogFile(QPushButton *button, const QString &path)
{
    if (!button)
        return false;
    bool selected = false;
    bool fileNameSet = false;
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
                                              [&](int result)
                                              {
                                                  selected = result == QDialog::Accepted;
                                                  selection.stop();
                                              });
                         }
                         if (elapsed.elapsed() > 5000)
                         {
                             dialog->reject();
                             return;
                         }
                         // Let the asynchronous file-system model settle, then enter the absolute
                         // path exactly as a user would. Repeated selectFile calls can reset it.
                         if (elapsed.elapsed() < 500)
                             return;
                         auto *fileName = dialog->findChild<QLineEdit *>(QStringLiteral("fileNameEdit"));
                         if (!fileName)
                             return;
                         if (!fileNameSet)
                         {
                             fileName->setText(QFileInfo(path).absoluteFilePath());
                             fileNameSet = true;
                         }
                         QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection);
                     });
    selection.start(50);
    QTest::mouseClick(button, Qt::LeftButton);
    return selected;
}

bool selectVideo(MainWindow *window, const QString &path)
{
    return chooseDialogFile(findButton(window, QStringLiteral("视频")), path);
}

QJsonObject exportedMetadata(const QString &home)
{
    const QDir results(home + QStringLiteral("/output/results"));
    const auto files = results.entryList({QStringLiteral("*.json")}, QDir::Files, QDir::Name);
    if (files.isEmpty())
        return {};
    QFile file(results.filePath(files.last()));
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
} // namespace

class UiTests final : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        originalHome_ = qgetenv("VISION_STUDIO_HOME");
        hadHome_ = qEnvironmentVariableIsSet("VISION_STUDIO_HOME");
        originalData_ = qgetenv("VISION_STUDIO_DATA_DIR");
        hadData_ = qEnvironmentVariableIsSet("VISION_STUDIO_DATA_DIR");
        originalPython_ = qgetenv("VISION_STUDIO_PYTHON");
        hadPython_ = qEnvironmentVariableIsSet("VISION_STUDIO_PYTHON");
        originalGpuRuntime_ = qgetenv("VISION_STUDIO_GPU_RUNTIME_DIR");
        hadGpuRuntime_ = qEnvironmentVariableIsSet("VISION_STUDIO_GPU_RUNTIME_DIR");
        QApplication::setStyle(QStringLiteral("Fusion"));
        QApplication::setFont(QFont(QStringLiteral("Noto Sans CJK SC"), 10));
        QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    }

    void init()
    {
        fixture_ = std::make_unique<QTemporaryDir>();
        QVERIFY(fixture_->isValid());
        QVERIFY(QDir().mkpath(fixture_->path() + QStringLiteral("/models")));
        QVERIFY(QDir().mkpath(fixture_->path() + QStringLiteral("/assets")));
        QVERIFY(QDir().mkpath(fixture_->path() + QStringLiteral("/scripts")));
        const QString project = QStringLiteral(VISION_PROJECT_DIR);
        QVERIFY2(QFile::copy(project + QStringLiteral("/models/yolov5n.onnx"),
                             fixture_->path() + QStringLiteral("/models/yolov5n.onnx")),
                 "The real YOLOv5n fixture is required for GUI integration tests.");
        QVERIFY(QFile::copy(project + QStringLiteral("/assets/bus.jpg"),
                            fixture_->path() + QStringLiteral("/assets/bus.jpg")));
        QVERIFY2(QFile::copy(project + QStringLiteral("/models/yolov8n.pt"),
                             fixture_->path() + QStringLiteral("/models/yolov8n.pt")),
                 "The official YOLOv8n checkpoint is required for PT GUI integration.");
        QVERIFY(QFile::copy(project + QStringLiteral("/scripts/pt_worker.py"),
                            fixture_->path() + QStringLiteral("/scripts/pt_worker.py")));
        QVERIFY(QFile::copy(project + QStringLiteral("/scripts/netron_server.py"),
                            fixture_->path() + QStringLiteral("/scripts/netron_server.py")));
        qputenv("VISION_STUDIO_HOME", fixture_->path().toUtf8());
        qputenv("VISION_STUDIO_DATA_DIR", (fixture_->path() + QStringLiteral("/output")).toUtf8());
        qputenv("VISION_STUDIO_GPU_RUNTIME_DIR",
                (fixture_->path() + QStringLiteral("/output/gpu-runtime")).toUtf8());
        prepareGpuUiFixtures();
        QVERIFY(QDir().mkpath(fixture_->path() + QStringLiteral("/output")));
        QSettings settings(fixture_->path() + QStringLiteral("/output/preferences.ini"),
                           QSettings::IniFormat);
        settings.setValue(QStringLiteral("computeDevice"), QStringLiteral("cpu"));
        settings.sync();
        unexpectedDialog_.clear();
        dialogGuard_ = std::make_unique<QTimer>();
        connect(dialogGuard_.get(), &QTimer::timeout, this,
                [this]
                {
                    for (QWidget *widget : QApplication::topLevelWidgets())
                    {
                        auto *dialog = qobject_cast<QMessageBox *>(widget);
                        if (dialog && dialog->isVisible())
                        {
                            unexpectedDialog_ = dialog->text();
                            dialog->accept();
                        }
                    }
                });
        dialogGuard_->start(50);
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        window_->activateWindow();
        QTest::qWait(100);
        QVERIFY(window_->isVisible());
    }

    void cleanup()
    {
        window_.reset();
        dialogGuard_.reset();
        fixture_.reset();
        if (hadHome_)
            qputenv("VISION_STUDIO_HOME", originalHome_);
        else
            qunsetenv("VISION_STUDIO_HOME");
        if (hadData_)
            qputenv("VISION_STUDIO_DATA_DIR", originalData_);
        else
            qunsetenv("VISION_STUDIO_DATA_DIR");
        if (hadPython_)
            qputenv("VISION_STUDIO_PYTHON", originalPython_);
        else
            qunsetenv("VISION_STUDIO_PYTHON");
        if (hadGpuRuntime_)
            qputenv("VISION_STUDIO_GPU_RUNTIME_DIR", originalGpuRuntime_);
        else
            qunsetenv("VISION_STUDIO_GPU_RUNTIME_DIR");
    }

    void realInferenceSelectionAndNavigation()
    {
        auto *device = window_->findChild<QComboBox *>(QStringLiteral("computeDevice"));
        QVERIFY(device);
        QCOMPARE(device->currentData().toString(), QStringLiteral("cpu"));
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        auto *canvas = workbenchCanvas(window_.get());
        auto *table = predictionTable(window_.get());
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QVERIFY(canvas);
        QVERIFY(table);
        QVERIFY(exportButton);
        window_->runDemo();
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QVERIFY(!canvas->result().demonstration);
        QVERIFY(canvas->result().inferenceMs > 0);
        QCOMPARE(starts.size(), 1);
        QCOMPARE(qvariant_cast<vision::JobRequest>(starts.first().at(0)).config.device,
                 vision::ComputeDevice::CPU);
        QCOMPARE(canvas->result().requestedDevice, vision::ComputeDevice::CPU);
        QCOMPARE(canvas->result().device, vision::ComputeDevice::CPU);
        QCOMPARE(canvas->result().deviceIndex, -1);
        auto *actual = window_->findChild<QLabel *>(QStringLiteral("actualDeviceBadge"));
        QVERIFY(actual);
        QVERIFY(actual->text().contains(QStringLiteral("CPU")));
        QVERIFY(actual->text().contains(QStringLiteral("OpenCV")));
        auto *footer = window_->findChild<QLabel *>(QStringLiteral("actualDeviceFooter"));
        QVERIFY(footer && footer->isVisible());
        QVERIFY(footer->height() >= footer->fontMetrics().height());
        QVERIFY(footer->text().contains(QStringLiteral("OpenCV")));
        QVERIFY(window_->rect().contains(QRect(footer->mapTo(window_.get(), QPoint()), footer->size())));
        QCOMPARE(canvas->result().image.size(),
                 QImage(fixture_->path() + QStringLiteral("/assets/bus.jpg")).size());
        QCOMPARE(table->rowCount(), canvas->result().predictions.size());

        canvas->fitToView();
        table->clearSelection();
        const QImage unselected = renderCanvas(canvas);
        table->setCurrentCell(0, 0);
        table->selectRow(0);
        QCOMPARE(table->selectedItems().size(), table->columnCount());
        QVERIFY(renderCanvas(canvas) != unselected);

        int smallestIndex = 0;
        qreal smallestArea = std::numeric_limits<qreal>::max();
        for (int i = 0; i < canvas->result().predictions.size(); ++i)
        {
            const QRectF box = canvas->result().predictions[i].box;
            if (box.width() * box.height() < smallestArea)
            {
                smallestArea = box.width() * box.height();
                smallestIndex = i;
            }
        }
        QSignalSpy selected(canvas, &ImageCanvas::predictionSelected);
        QTest::mouseClick(canvas, Qt::LeftButton, Qt::NoModifier, predictionCenter(canvas, smallestIndex));
        QCOMPARE(selected.size(), 1);
        QCOMPARE(selected.first().at(0).toInt(), smallestIndex);
        QVERIFY(!table->selectedItems().isEmpty());
        QCOMPARE(table->selectedItems().first()->row(), smallestIndex);
        QCOMPARE(table->currentRow(), smallestIndex);
        QTest::mouseClick(canvas, Qt::LeftButton, Qt::NoModifier, QPoint(8, 8));
        QCOMPARE(selected.last().at(0).toInt(), -1);
        QVERIFY(table->selectedItems().isEmpty());

        auto *pages = window_->findChild<QStackedWidget *>(QStringLiteral("workspacePages"));
        auto *title = window_->findChild<QLabel *>(QStringLiteral("pageTitle"));
        auto *demoButton = findButton(window_.get(), QStringLiteral("运行示例"));
        QVERIFY(pages);
        QVERIFY(title);
        QVERIFY(demoButton);
        QCOMPARE(pages->count(), 7);
        QCOMPARE(pages->widget(0)->objectName(), QStringLiteral("workbenchPage"));
        QCOMPARE(pages->widget(2)->objectName(), QStringLiteral("modelDisplayPage"));
        QCOMPARE(pages->widget(6)->objectName(), QStringLiteral("morePage"));
        QVERIFY(pages->widget(6)->isAncestorOf(demoButton));
        QVERIFY(pages->widget(6)->isAncestorOf(exportButton));
        QVERIFY(!demoButton->isVisible());
        QVERIFY(!exportButton->isVisible());
        for (const auto &route :
             {qMakePair(QStringLiteral("模型库"), 1), qMakePair(QStringLiteral("模型显示"), 2),
              qMakePair(QStringLiteral("运行记录"), 3), qMakePair(QStringLiteral("使用指南"), 4),
              qMakePair(QStringLiteral("录制视频"), 5), qMakePair(QStringLiteral("更多"), 6),
              qMakePair(QStringLiteral("检测工作台"), 0)})
        {
            auto *nav = findButton(window_.get(), route.first);
            QVERIFY(nav);
            QTest::mouseClick(nav, Qt::LeftButton);
            QCOMPARE(pages->currentIndex(), route.second);
            QCOMPARE(title->text(), route.first);
            QVERIFY(nav->isChecked());
            QCOMPARE(demoButton->isVisible(), route.second == 6);
            QCOMPARE(exportButton->isVisible(), route.second == 6);
            if (route.second == 6)
                window_->saveScreenshot(QStringLiteral(VISION_PROJECT_DIR) +
                                        QStringLiteral("/output/test-screenshots/1.4-more.png"));
        }
        QCOMPARE(window_->size(), QSize(1260, 820));
        const QRect canvasBounds(canvas->mapTo(window_.get(), QPoint()), canvas->size());
        QVERIFY(window_->rect().contains(canvasBounds));
        QVERIFY(canvas->width() >= 300 && canvas->height() >= 280);
        QVERIFY(findButton(window_.get(), QStringLiteral("开始检测"))->isVisible());
        const QString screenshot = qEnvironmentVariable("VISION_UI_TEST_SCREENSHOT");
        if (!screenshot.isEmpty())
            window_->saveScreenshot(screenshot);
    }

    void globalModelSelectionPreloadsAndCachesStructure()
    {
        auto *pages = window_->findChild<QStackedWidget *>(QStringLiteral("workspacePages"));
        auto *hint = window_->findChild<QLabel *>(QStringLiteral("structureOnnxHint"));
        auto *models = window_->findChild<QListWidget *>(QStringLiteral("globalModelList"));
        auto *name = window_->findChild<QLabel *>(QStringLiteral("structureModelName"));
        auto *active = window_->findChild<QLabel *>(QStringLiteral("activeModelName"));
        auto *viewer = window_->findChild<ModelViewer *>(QStringLiteral("modelStructureViewer"));
        QVERIFY(pages && hint && models && name && active && viewer);
        QVERIFY(!window_->findChild<QComboBox *>(QStringLiteral("structureModelCombo")));
        QVERIFY(!window_->findChild<QPushButton *>(QStringLiteral("chooseStructureModelButton")));
        QCOMPARE(hint->text(), QStringLiteral("想看完整结构，建使用导出的 ONNX。"));
        QCOMPARE(pages->currentIndex(), 0);
        QVERIFY(!viewer->isVisible());
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, 30000);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
        QCOMPARE(QFileInfo(viewer->modelPath()).fileName(), QStringLiteral("yolov5n.onnx"));
        QVERIFY(viewer->hierarchyItemCount() > 0);
        QVERIFY(viewer->parameterCount() > 0);
        QVERIFY(hint->isHidden());

        QTest::mouseClick(findButton(window_.get(), QStringLiteral("模型库")), Qt::LeftButton);
        int ptRow = -1;
        for (int row = 0; row < models->count(); ++row)
            if (QFileInfo(models->item(row)->data(Qt::UserRole).toString()).fileName() == "yolov8n.pt")
                ptRow = row;
        QVERIFY(ptRow >= 0);
        QSignalSpy loaded(viewer, &ModelViewer::modelLoaded);
        models->setCurrentRow(ptRow);
        QCOMPARE(pages->currentIndex(), 1);
        QCOMPARE(name->text(), QStringLiteral("yolov8n.pt"));
        QCOMPARE(active->text(), name->text());
        QSettings settings(fixture_->path() + "/output/preferences.ini", QSettings::IniFormat);
        QCOMPARE(QFileInfo(settings.value("activeModel").toString()).fileName(),
                 QStringLiteral("yolov8n.pt"));
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, 30000);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
        QVERIFY(viewer->graphNodeCount() > 0);
        QVERIFY(viewer->hierarchyItemCount() > 0);
        QVERIFY(viewer->parameterCount() > 0);
        QCOMPARE(loaded.count(), 1);
        auto *browser = viewer->findChild<QWebEngineView *>(QStringLiteral("netronWebView"));
        auto *service = viewer->findChild<QProcess *>();
        QVERIFY(browser && service);
        const qint64 pid = service->processId();
        QSignalSpy pageLoads(browser->page(), &QWebEnginePage::loadStarted);
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("模型显示")), Qt::LeftButton);
        auto *tree = viewer->findChild<QTreeView *>(QStringLiteral("modelHierarchyTree"));
        auto *parameters = viewer->findChild<QTableView *>(QStringLiteral("modelParameterTable"));
        auto *search = viewer->findChild<QLineEdit *>(QStringLiteral("modelStructureSearch"));
        QVERIFY(tree && parameters && search);
        for (const auto &choice :
             {qMakePair(ModelViewer::DisplayMode::Hierarchy, "modelHierarchyModeButton"),
              qMakePair(ModelViewer::DisplayMode::Parameters, "modelParametersModeButton"),
              qMakePair(ModelViewer::DisplayMode::Graph, "modelGraphModeButton"),
              qMakePair(ModelViewer::DisplayMode::Parameters, "modelParametersModeButton")})
        {
            auto *button = viewer->findChild<QAbstractButton *>(choice.second);
            QVERIFY(button && button->isVisible());
            QTest::mouseClick(button, Qt::LeftButton);
            QCOMPARE(viewer->displayMode(), choice.first);
            QCOMPARE(viewer->state(), ModelViewer::State::Ready);
        }
        QVERIFY(parameters->isVisible());
        search->setFocus();
        QTest::keyClicks(search, "model.0.conv.weight");
        QTRY_COMPARE_WITH_TIMEOUT(parameters->model()->rowCount(), 1, 3000);
        const QModelIndex firstWeight = parameters->model()->index(0, 0);
        QVERIFY(
            firstWeight.data(Qt::UserRole + 1).toString().contains(QStringLiteral("model.0.conv.weight")));
        QCOMPARE(firstWeight.data(Qt::UserRole + 3).toString(), QStringLiteral("[16,3,3,3]"));
        for (const QString &route : {QStringLiteral("检测工作台"), QStringLiteral("模型显示")})
            QTest::mouseClick(findButton(window_.get(), route), Qt::LeftButton);
        QTest::qWait(150);
        QCOMPARE(viewer->displayMode(), ModelViewer::DisplayMode::Parameters);
        QCOMPARE(search->text(), QStringLiteral("model.0.conv.weight"));
        QCOMPARE(parameters->model()->rowCount(), 1);
        QCOMPARE(viewer->findChild<QWebEngineView *>(QStringLiteral("netronWebView")), browser);
        QCOMPARE(service->processId(), pid);
        QCOMPARE(pageLoads.count(), 0);
        QCOMPARE(loaded.count(), 1);
        QVERIFY(hint->isVisible());
        QCOMPARE(window_->size(), QSize(1260, 820));
        window_->saveScreenshot(QStringLiteral(VISION_PROJECT_DIR) +
                                QStringLiteral("/output/test-screenshots/1.5-parameters.png"));
        // Clear the global selection through the real model-library action.
        // Removing list entries preserves the original files and empties all
        // three cached views without retaining the old search or details.
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("模型库")), Qt::LeftButton);
        auto *remove = findButton(window_.get(), QStringLiteral("从列表移除"));
        QVERIFY(remove);
        while (models->count())
        {
            const int before = models->count();
            models->setCurrentRow(0);
            QTest::mouseClick(remove, Qt::LeftButton);
            QCOMPARE(models->count(), before - 1);
        }
        QTRY_COMPARE_WITH_TIMEOUT(viewer->state(), ModelViewer::State::Empty, 3000);
        QVERIFY(viewer->modelPath().isEmpty());
        QCOMPARE(viewer->graphNodeCount(), 0);
        QCOMPARE(viewer->hierarchyItemCount(), 0);
        QCOMPARE(viewer->parameterCount(), 0);
        QCOMPARE(tree->model()->rowCount(), 0);
        QCOMPARE(parameters->model()->rowCount(), 0);
        QVERIFY(search->text().isEmpty());
        QVERIFY(viewer->findChildren<QWebEngineView *>().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(viewer->findChildren<QProcess *>().isEmpty(), 3000);
        QVERIFY(QFileInfo(fixture_->path() + "/models/yolov5n.onnx").isFile());
        QVERIFY(QFileInfo(fixture_->path() + "/models/yolov8n.pt").isFile());
    }

    void shortcutsCancelAndRestartFromFocusedControls()
    {
        auto *canvas = workbenchCanvas(window_.get());
        auto *stop = findButton(window_.get(), QStringLiteral("停止运行"));
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QSpinBox *inputSize = nullptr;
        for (auto *spin : window_->findChildren<QSpinBox *>())
            if (spin->suffix() == QStringLiteral(" px"))
                inputSize = spin;
        QVERIFY(canvas);
        QVERIFY(stop);
        QVERIFY(run);
        QVERIFY(exportButton);
        QVERIFY(inputSize);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        canvas->setFocus();
        QTest::keyClick(canvas, Qt::Key_R, Qt::ControlModifier);
        QTRY_COMPARE_WITH_TIMEOUT(starts.size(), 1, 2000);
        QVERIFY(stop->isVisible());
        QVERIFY(!exportButton->isEnabled());
        QTest::keyClick(canvas, Qt::Key_Escape);
        QTRY_VERIFY_WITH_TIMEOUT(!stop->isVisible() && run->isVisible(), 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));

        inputSize->setFocus();
        QTest::keyClick(inputSize, Qt::Key_R, Qt::ControlModifier);
        QTRY_COMPARE_WITH_TIMEOUT(starts.size(), 2, 2000);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QVERIFY(inputSize->isEnabled());
        QVERIFY(run->isVisible());
        QVERIFY(!stop->isVisible());
    }

    void directPtInferenceReadsNamesAndExportsNativeBackend()
    {
        auto *canvas = workbenchCanvas(window_.get());
        auto *table = predictionTable(window_.get());
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QCheckBox *autoExport = nullptr;
        for (auto *check : window_->findChildren<QCheckBox *>())
            if (check->text() == QStringLiteral("自动保存每张图片的结果"))
                autoExport = check;
        QVERIFY(canvas);
        QVERIFY(table);
        QVERIFY(exportButton);
        QVERIFY(autoExport);
        autoExport->setChecked(true);
        window_->runPtDemo();
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 90000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        const auto &result = canvas->result();
        QVERIFY(!result.demonstration);
        QCOMPARE(result.modelName, QStringLiteral("yolov8n.pt"));
        QVERIFY(result.backend.contains(QStringLiteral("PyTorch")));
        QVERIFY(result.inferenceMs > 0);
        QCOMPARE(table->rowCount(), result.predictions.size());
        QVERIFY(findButton(window_.get(), QStringLiteral("类别标签 · 内置 80 类")));
        bool hasPerson = false;
        bool hasBus = false;
        for (const auto &prediction : result.predictions)
        {
            hasPerson |= prediction.classId == 0 && prediction.label == QStringLiteral("person");
            hasBus |= prediction.classId == 5 && prediction.label == QStringLiteral("bus");
        }
        QVERIFY(hasPerson);
        QVERIFY(hasBus);

        const QDir exported(fixture_->path() + QStringLiteral("/output/results"));
        const auto jsonFiles = exported.entryList({QStringLiteral("*.json")}, QDir::Files);
        const auto pngFiles = exported.entryList({QStringLiteral("*.png")}, QDir::Files);
        const auto csvFiles = exported.entryList({QStringLiteral("*.csv")}, QDir::Files);
        QCOMPARE(jsonFiles.size(), 1);
        QCOMPARE(pngFiles.size(), 1);
        QCOMPARE(csvFiles.size(), 1);
        QFile json(exported.filePath(jsonFiles.first()));
        QVERIFY(json.open(QIODevice::ReadOnly));
        const QJsonObject metadata = QJsonDocument::fromJson(json.readAll()).object();
        QVERIFY(metadata.value(QStringLiteral("backend")).toString().contains(QStringLiteral("PyTorch")));
        QCOMPARE(metadata.value(QStringLiteral("requested_device")).toString(), QStringLiteral("cpu"));
        QCOMPARE(metadata.value(QStringLiteral("actual_device")).toString(), QStringLiteral("cpu"));
        QCOMPARE(metadata.value(QStringLiteral("device_index")).toInt(), -1);
        QVERIFY(metadata.contains(QStringLiteral("device_name")) &&
                metadata.contains(QStringLiteral("device_notice")));
        QCOMPARE(metadata.value(QStringLiteral("model")).toString(), QStringLiteral("yolov8n.pt"));
        QCOMPARE(metadata.value(QStringLiteral("predictions")).toArray().size(), result.predictions.size());
        const QJsonObject config = metadata.value(QStringLiteral("config")).toObject();
        QCOMPARE(config.value(QStringLiteral("input_size")).toInt(), 640);
        QCOMPARE(
            config.value(QStringLiteral("preprocess")).toObject().value(QStringLiteral("mode")).toString(),
            QStringLiteral("model_native"));
        QCOMPARE(QImage(exported.filePath(pngFiles.first())).size(), result.image.size());
        QFile csv(exported.filePath(csvFiles.first()));
        QVERIFY(csv.open(QIODevice::ReadOnly));
        QCOMPARE(csv.readAll().count('\n'), result.predictions.size() + 1);
    }

    void queueChangeAndClearInvalidateOldResults()
    {
        auto *canvas = workbenchCanvas(window_.get());
        auto *table = predictionTable(window_.get());
        auto *queue = inputQueue(window_.get());
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QVERIFY(canvas);
        QVERIFY(table);
        QVERIFY(queue);
        QVERIFY(exportButton);
        window_->runDemo();
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        const QString second = fixture_->path() + QStringLiteral("/assets/second.jpg");
        QVERIFY(QFile::copy(fixture_->path() + QStringLiteral("/assets/bus.jpg"), second));
        QVERIFY(dropFile(canvas, second));
        QCOMPARE(queue->count(), 2);
        QCOMPARE(queue->currentRow(), 1);
        QCOMPARE(canvas->result().source, second);
        QVERIFY(!canvas->result().image.isNull());
        QVERIFY(canvas->result().predictions.isEmpty());
        QCOMPARE(table->rowCount(), 0);
        QVERIFY(!exportButton->isEnabled());
        const auto metrics = window_->findChildren<QLabel *>(QStringLiteral("metricValue"));
        QCOMPARE(metrics.size(), 4);
        for (int i = 0; i < 3; ++i)
            QCOMPARE(metrics[i]->text(), QStringLiteral("—"));
        QVERIFY(metrics[3]->text().contains(QStringLiteral("×")));

        auto *clear = findButton(window_.get(), QStringLiteral("清空"));
        QVERIFY(clear);
        QTest::mouseClick(clear, Qt::LeftButton);
        QCOMPARE(queue->count(), 0);
        QVERIFY(canvas->result().image.isNull());
        QVERIFY(canvas->result().predictions.isEmpty());
        QCOMPARE(table->rowCount(), 0);
        QVERIFY(!exportButton->isEnabled());
        for (auto *metric : metrics)
            QCOMPARE(metric->text(), QStringLiteral("—"));
    }

    void grayscalePreviewAndColorSwitchInvalidatePredictions()
    {
        auto *canvas = workbenchCanvas(window_.get());
        auto *table = predictionTable(window_.get());
        auto *color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        auto *autoExport = autoExportControl(window_.get());
        QVERIFY(canvas && table && color && run && exportButton && autoExport);
        autoExport->setChecked(false);
        window_->runDemo();
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QVERIFY(!canvas->result().image.isGrayscale());

        color->setCurrentIndex(color->findData(QStringLiteral("grayscale")));
        QVERIFY(canvas->result().predictions.isEmpty());
        QCOMPARE(table->rowCount(), 0);
        QVERIFY(!exportButton->isEnabled());
        autoExport->setChecked(true);
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        const vision::InferenceResult grayResult = canvas->result();
        QCOMPARE(grayResult.image.format(), QImage::Format_Grayscale8);
        QVERIFY(grayResult.image.isGrayscale());
        QVERIFY(!grayResult.originalImage.isGrayscale());
        QCOMPARE(grayResult.image.size(), grayResult.originalImage.size());
        QCOMPARE(table->rowCount(), grayResult.predictions.size());
        const QJsonObject metadata = exportedMetadata(fixture_->path());
        QCOMPARE(metadata.value(QStringLiteral("config"))
                     .toObject()
                     .value(QStringLiteral("color_mode"))
                     .toString(),
                 QStringLiteral("grayscale"));

        color->setCurrentIndex(color->findData(QStringLiteral("rgb")));
        QCOMPARE(canvas->result().image.convertToFormat(QImage::Format_RGB888),
                 grayResult.originalImage.convertToFormat(QImage::Format_RGB888));
        QVERIFY(!canvas->result().image.isGrayscale());
        QVERIFY(canvas->result().predictions.isEmpty());
        QCOMPARE(table->rowCount(), 0);
        QVERIFY(!exportButton->isEnabled());
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().predictions.isEmpty()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QVERIFY(!canvas->result().image.isGrayscale());
    }

    void legacyColorPreferenceMigration_data()
    {
        QTest::addColumn<bool>("swapRb");
        QTest::addColumn<QString>("expectedMode");
        QTest::newRow("legacy-rgb") << true << QStringLiteral("rgb");
        QTest::newRow("legacy-bgr") << false << QStringLiteral("bgr");
    }

    void legacyColorPreferenceMigration()
    {
        QFETCH(bool, swapRb);
        QFETCH(QString, expectedMode);
        window_->close();
        window_.reset();
        const QString preferenceFile = fixture_->path() + QStringLiteral("/output/preferences.ini");
        {
            QSettings settings(preferenceFile, QSettings::IniFormat);
            settings.remove(QStringLiteral("colorMode"));
            settings.setValue(QStringLiteral("swapRB"), swapRb);
            settings.sync();
        }
        showNewWindow();
        auto *color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        QVERIFY(color);
        QCOMPARE(color->currentData().toString(), expectedMode);
        window_->close();
        QSettings migrated(preferenceFile, QSettings::IniFormat);
        QCOMPARE(migrated.value(QStringLiteral("colorMode")).toString(), expectedMode);
    }

    void grayscaleAndStereoPreferencesSurviveRestartAndPtSwitch()
    {
        auto *color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        auto *stereo = window_->findChild<QComboBox *>(QStringLiteral("stereoView"));
        auto *camera = window_->findChild<QSpinBox *>(QStringLiteral("cameraIndex"));
        auto *meanR = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanR"));
        auto *meanG = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanG"));
        auto *meanB = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanB"));
        QVERIFY(color && stereo && camera && meanR && meanG && meanB);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        meanR->setValue(7.5);
        meanG->setValue(19);
        meanB->setValue(33);
        color->setCurrentIndex(color->findData(QStringLiteral("grayscale")));
        QVERIFY(meanR->isEnabled());
        QVERIFY(!meanG->isEnabled());
        QVERIFY(!meanB->isEnabled());
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("摄像头")), Qt::LeftButton);
        camera->setValue(3);
        stereo->setCurrentIndex(stereo->findData(QStringLiteral("left")));
        QVERIFY(stereo->isVisible() && stereo->isEnabled());
        QVERIFY(color->isVisible() && color->isEnabled());
        QCOMPARE(window_->size(), QSize(1260, 820));
        for (QWidget *control :
             {static_cast<QWidget *>(color), static_cast<QWidget *>(stereo), static_cast<QWidget *>(camera),
              static_cast<QWidget *>(findButton(window_.get(), QStringLiteral("开始检测")))})
        {
            QVERIFY(control);
            QVERIFY(control->isVisible());
            QVERIFY(
                window_->rect().contains(QRect(control->mapTo(window_.get(), QPoint()), control->size())));
        }
        const QString screenshot = qEnvironmentVariable(
            "VISION_UI_CAMERA_SCREENSHOT",
            QStringLiteral(VISION_PROJECT_DIR) + QStringLiteral("/output/test-screenshots/1.4-camera.png"));
        window_->saveScreenshot(screenshot);
        QCOMPARE(QImage(screenshot).size(), window_->size());
        QCOMPARE(starts.size(), 0); // Configuring a camera must not connect to a device.

        window_->close();
        window_.reset();
        showNewWindow();
        color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        stereo = window_->findChild<QComboBox *>(QStringLiteral("stereoView"));
        camera = window_->findChild<QSpinBox *>(QStringLiteral("cameraIndex"));
        meanR = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanR"));
        meanG = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanG"));
        meanB = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanB"));
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        QCOMPARE(stereo->currentData().toString(), QStringLiteral("left"));
        QCOMPARE(camera->value(), 3);
        QCOMPARE(meanR->value(), 7.5);
        QVERIFY(!meanG->isEnabled() && !meanB->isEnabled());
        QVERIFY(!stereo->isVisible() && !stereo->isEnabled()); // Images are always full frame.
        QVERIFY(activateModel(window_.get(), QStringLiteral("yolov8n.pt")));
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        QVERIFY(!meanR->isEnabled());
        auto *items = qobject_cast<QStandardItemModel *>(color->model());
        QVERIFY(items);
        QVERIFY(!items->item(color->findData(QStringLiteral("bgr")))->isEnabled());
        QVERIFY(activateModel(window_.get(), QStringLiteral("yolov5n.onnx")));
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        QVERIFY(meanR->isEnabled());
        QVERIFY(!meanG->isEnabled() && !meanB->isEnabled());
        QVERIFY(items->item(color->findData(QStringLiteral("bgr")))->isEnabled());
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("摄像头")), Qt::LeftButton);
        QVERIFY(stereo->isVisible());
        QCOMPARE(stereo->currentData().toString(), QStringLiteral("left"));
        QCOMPARE(camera->value(), 3);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
    }

    void stereoVideoRequestExportAndImageIsolation()
    {
        const QString video = fixture_->path() + QStringLiteral("/assets/stereo.avi");
        cv::VideoWriter writer(video.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                               cv::Size(320, 96));
        QVERIFY2(writer.isOpened(), "The local OpenCV MJPEG encoder is required for stream integration.");
        cv::Mat frame(96, 320, CV_8UC3);
        frame(cv::Rect(0, 0, 160, 96)).setTo(cv::Scalar(0, 0, 200));
        frame(cv::Rect(160, 0, 160, 96)).setTo(cv::Scalar(200, 0, 0));
        writer.write(frame);
        writer.write(frame);
        writer.release();

        auto *canvas = workbenchCanvas(window_.get());
        auto *color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        auto *stereo = window_->findChild<QComboBox *>(QStringLiteral("stereoView"));
        auto *meanR = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanR"));
        auto *meanG = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanG"));
        auto *meanB = window_->findChild<QDoubleSpinBox *>(QStringLiteral("meanB"));
        auto *autoExport = autoExportControl(window_.get());
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *exportButton = findButton(window_.get(), QStringLiteral("导出结果"));
        QVERIFY(canvas && color && stereo && meanR && meanG && meanB && autoExport && run && exportButton);
        QVERIFY(selectVideo(window_.get(), video));
        QVERIFY(stereo->isVisible() && stereo->isEnabled());
        stereo->setCurrentIndex(stereo->findData(QStringLiteral("left")));
        meanR->setValue(9);
        meanG->setValue(27);
        meanB->setValue(42);
        color->setCurrentIndex(color->findData(QStringLiteral("grayscale")));
        autoExport->setChecked(true);
        QSignalSpy starts(window_.get(), &MainWindow::startRequested);
        QTest::mouseClick(run, Qt::LeftButton);
        QCOMPARE(starts.size(), 1);
        const auto videoRequest = qvariant_cast<vision::JobRequest>(starts.first().at(0));
        QCOMPARE(videoRequest.sourceKind, vision::SourceKind::Video);
        QCOMPARE(videoRequest.stereoView, vision::StereoView::Left);
        QCOMPARE(videoRequest.config.colorMode, vision::InputColorMode::Grayscale);
        QCOMPARE(videoRequest.config.device, vision::ComputeDevice::CPU);
        QCOMPARE(videoRequest.config.meanR, 9.0);
        QCOMPARE(videoRequest.config.meanG, 9.0);
        QCOMPARE(videoRequest.config.meanB, 9.0);
        QVERIFY(!stereo->isEnabled() && !color->isEnabled());
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().image.isNull()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QCOMPARE(canvas->result().stereoView, vision::StereoView::Left);
        QCOMPARE(canvas->result().sourceFrameSize, QSize(320, 96));
        QCOMPARE(canvas->result().image.size(), QSize(160, 96));
        const QColor pixel = canvas->result().image.pixelColor(80, 48);
        QCOMPARE(pixel.red(), pixel.green());
        QCOMPARE(pixel.green(), pixel.blue());
        QCOMPARE(canvas->result().originalImage.size(), QSize(160, 96));
        const QColor originalPixel = canvas->result().originalImage.pixelColor(80, 48);
        QVERIFY(originalPixel.red() > originalPixel.blue() + 100);
        auto *modelMeta = window_->findChild<QLabel *>(QStringLiteral("modelMetadata"));
        QVERIFY(modelMeta && modelMeta->text().contains(QStringLiteral("输入 3 通道")));
        const QJsonObject metadata = exportedMetadata(fixture_->path());
        QVERIFY(!metadata.isEmpty());
        QCOMPARE(metadata.value(QStringLiteral("stereo_view")).toString(), QStringLiteral("left"));
        QCOMPARE(metadata.value(QStringLiteral("width")).toInt(), 160);
        QCOMPARE(metadata.value(QStringLiteral("height")).toInt(), 96);
        const QJsonObject sourceSize = metadata.value(QStringLiteral("source_frame_size")).toObject();
        QCOMPARE(sourceSize.value(QStringLiteral("width")).toInt(), 320);
        QCOMPARE(sourceSize.value(QStringLiteral("height")).toInt(), 96);
        const QJsonObject config = metadata.value(QStringLiteral("config")).toObject();
        QCOMPARE(config.value(QStringLiteral("color_mode")).toString(), QStringLiteral("grayscale"));
        QCOMPARE(config.value(QStringLiteral("input_channels")).toInt(), 3);
        QCOMPARE(
            config.value(QStringLiteral("preprocess")).toObject().value(QStringLiteral("mean_rgb")).toArray(),
            QJsonArray({9, 9, 9}));
        QFile history(fixture_->path() + QStringLiteral("/output/history.json"));
        QVERIFY(history.open(QIODevice::ReadOnly));
        QCOMPARE(QJsonDocument::fromJson(history.readAll())
                     .array()
                     .first()
                     .toObject()
                     .value(QStringLiteral("stereo_view"))
                     .toString(),
                 QStringLiteral("left"));

        stereo->setCurrentIndex(stereo->findData(QStringLiteral("right")));
        QVERIFY(canvas->result().image.isNull());
        QVERIFY(!exportButton->isEnabled()); // Do not export an old result after switching eyes.
        QVERIFY(dropFile(canvas, fixture_->path() + QStringLiteral("/assets/bus.jpg")));
        QVERIFY(!stereo->isVisible() && !stereo->isEnabled());
        QCOMPARE(stereo->currentData().toString(), QStringLiteral("right"));
        QTest::mouseClick(run, Qt::LeftButton);
        QCOMPARE(starts.size(), 2);
        const auto imageRequest = qvariant_cast<vision::JobRequest>(starts.last().at(0));
        QCOMPARE(imageRequest.sourceKind, vision::SourceKind::Images);
        QCOMPARE(imageRequest.stereoView, vision::StereoView::Full);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() ||
                                     (exportButton->isEnabled() && !canvas->result().image.isNull()),
                                 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QCOMPARE(canvas->result().stereoView, vision::StereoView::Full);
        QCOMPARE(canvas->result().image.size(),
                 QImage(fixture_->path() + QStringLiteral("/assets/bus.jpg")).size());
    }

    void recordingStartStopListsAndPlaysSelectedGrayscaleVideo()
    {
        const QString video = fixture_->path() + QStringLiteral("/assets/record-source.avi");
        cv::VideoWriter writer(video.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                               cv::Size(320, 96));
        QVERIFY2(writer.isOpened(), "The local MJPEG encoder is required for recording UI integration.");
        cv::Mat frame(96, 320, CV_8UC3);
        frame(cv::Rect(0, 0, 160, 96)).setTo(cv::Scalar(0, 0, 200));
        frame(cv::Rect(160, 0, 160, 96)).setTo(cv::Scalar(220, 0, 0));
        for (int i = 0; i < 30; ++i)
            writer.write(frame);
        writer.release();

        auto *canvas = workbenchCanvas(window_.get());
        auto *color = window_->findChild<QComboBox *>(QStringLiteral("inputColorMode"));
        auto *stereo = window_->findChild<QComboBox *>(QStringLiteral("stereoView"));
        auto *run = findButton(window_.get(), QStringLiteral("开始检测"));
        auto *record = window_->findChild<QPushButton *>(QStringLiteral("recordButton"));
        auto *recordings = window_->findChild<QTableWidget *>(QStringLiteral("recordingsTable"));
        auto *play = window_->findChild<QPushButton *>(QStringLiteral("recordingPlayButton"));
        auto *recordingCanvas = window_->findChild<ImageCanvas *>(QStringLiteral("recordingCanvas"));
        QVERIFY(canvas && color && stereo && run && record && recordings && play && recordingCanvas);
        QVERIFY(!record->isEnabled()); // Image batches cannot be recorded.
        QVERIFY(selectVideo(window_.get(), video));
        stereo->setCurrentIndex(stereo->findData(QStringLiteral("left")));
        color->setCurrentIndex(color->findData(QStringLiteral("grayscale")));
        QVERIFY(!record->isEnabled()); // Start the inference stream before recording.
        QTest::mouseClick(run, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() || record->isEnabled(), 45000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QTest::mouseClick(record, Qt::LeftButton);
        QTRY_COMPARE_WITH_TIMEOUT(record->text(), QStringLiteral("结束录制"), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!unexpectedDialog_.isEmpty() || canvas->result().frameNumber >= 4, 20000);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
        QTest::mouseClick(record, Qt::LeftButton);

        const QDir directory(fixture_->path() + QStringLiteral("/output/recordings"));
        QTRY_COMPARE_WITH_TIMEOUT(directory.entryList({QStringLiteral("*.avi")}, QDir::Files).size(), 1,
                                  20000);
        QTRY_COMPARE_WITH_TIMEOUT(directory.entryList({QStringLiteral("*.json")}, QDir::Files).size(), 1,
                                  20000);
        QTRY_COMPARE_WITH_TIMEOUT(record->text(), QStringLiteral("开始录制"), 20000);
        const QString fileName = directory.entryList({QStringLiteral("*.avi")}, QDir::Files).first();
        const QString path = directory.filePath(fileName);
        cv::VideoCapture capture(path.toStdString());
        QVERIFY(capture.isOpened());
        cv::Mat recorded;
        QVERIFY(capture.read(recorded));
        QCOMPARE(recorded.cols, 160);
        QCOMPARE(recorded.rows, 96);
        QCOMPARE(recorded.channels(), 3);
        // MJPEG encodes grayscale as BGR. Most of the known flat image must remain
        // neutral; colored prediction overlays may occupy a small part of it.
        int neutralPixels = 0;
        for (int y = 0; y < recorded.rows; ++y)
            for (int x = 0; x < recorded.cols; ++x)
            {
                const auto pixel = recorded.at<cv::Vec3b>(y, x);
                if (std::abs(int(pixel[0]) - int(pixel[1])) <= 4 &&
                    std::abs(int(pixel[1]) - int(pixel[2])) <= 4)
                    ++neutralPixels;
            }
        QVERIFY(neutralPixels > recorded.cols * recorded.rows / 2);
        QFile sidecar(directory.filePath(QFileInfo(fileName).completeBaseName() + QStringLiteral(".json")));
        QVERIFY(sidecar.open(QIODevice::ReadOnly));
        const QByteArray originalMetadata = sidecar.readAll();
        const QJsonObject metadata = QJsonDocument::fromJson(originalMetadata).object();
        QVERIFY(metadata.value(QStringLiteral("frames")).toInt() > 0);
        QVERIFY(metadata.value(QStringLiteral("fps")).toDouble() > 0);
        QVERIFY(metadata.value(QStringLiteral("duration_seconds")).toDouble() > 0);
        QCOMPARE(metadata.value(QStringLiteral("content_width")).toInt(), 160);
        QCOMPARE(metadata.value(QStringLiteral("content_height")).toInt(), 96);
        QCOMPARE(metadata.value(QStringLiteral("color_mode")).toString(), QStringLiteral("grayscale"));
        QCOMPARE(metadata.value(QStringLiteral("stereo_view")).toString(), QStringLiteral("left"));
        QVERIFY(directory.entryList({QStringLiteral("*.partial.avi")}, QDir::Files | QDir::Hidden).isEmpty());

        auto *nav = findButton(window_.get(), QStringLiteral("录制视频"));
        QVERIFY(nav);
        QTest::mouseClick(nav, Qt::LeftButton);
        QTRY_COMPARE_WITH_TIMEOUT(recordings->rowCount(), 1, 5000);
        bool listed = false;
        for (int column = 0; column < recordings->columnCount(); ++column)
            if (auto *item = recordings->item(0, column))
                listed |= item->text().contains(QFileInfo(fileName).completeBaseName()) ||
                          item->toolTip().contains(path) || item->data(Qt::UserRole).toString() == path;
        QVERIFY(listed);
        recordings->setCurrentCell(0, 0);
        recordings->selectRow(0);
        QVERIFY(play->isEnabled());
        QTest::mouseClick(play, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!recordingCanvas->result().image.isNull(), 5000);
        const QImage preview = recordingCanvas->result().image;
        QCOMPARE(preview.size(), QSize(160, 96));
        int neutralPreviewPixels = 0;
        for (int y = 0; y < preview.height(); ++y)
            for (int x = 0; x < preview.width(); ++x)
            {
                const QColor pixel = preview.pixelColor(x, y);
                if (std::abs(pixel.red() - pixel.green()) <= 4 && std::abs(pixel.green() - pixel.blue()) <= 4)
                    ++neutralPreviewPixels;
            }
        QVERIFY(neutralPreviewPixels > preview.width() * preview.height() / 2);
        window_->saveScreenshot(QStringLiteral(VISION_PROJECT_DIR) +
                                QStringLiteral("/output/test-screenshots/1.4-recordings.png"));
        if (play->text() == QStringLiteral("暂停播放"))
            QTest::mouseClick(play, Qt::LeftButton);

        auto *exportRecording = window_->findChild<QPushButton *>(QStringLiteral("exportRecordingButton"));
        QVERIFY(exportRecording && exportRecording->isEnabled());
        const QString exportedDirectory = fixture_->path() + QStringLiteral("/exported-recording");
        QVERIFY(QDir().mkpath(exportedDirectory));
        const QString exportedVideo = exportedDirectory + QStringLiteral("/recording-copy.avi");
        QVERIFY(chooseDialogFile(exportRecording, exportedVideo));
        QFile originalVideo(path);
        QFile copiedVideo(exportedVideo);
        QVERIFY(originalVideo.open(QIODevice::ReadOnly));
        QVERIFY(copiedVideo.open(QIODevice::ReadOnly));
        const QByteArray originalBytes = originalVideo.readAll();
        QCOMPARE(copiedVideo.readAll(), originalBytes);
        QFile copiedMetadata(exportedDirectory + QStringLiteral("/recording-copy.json"));
        QVERIFY(copiedMetadata.open(QIODevice::ReadOnly));
        QCOMPARE(copiedMetadata.readAll(), originalMetadata);

        const QString invalidDestination = exportedDirectory + QStringLiteral("/invalid.json");
        QVERIFY(chooseDialogFile(exportRecording, invalidDestination));
        QVERIFY(!QFileInfo::exists(invalidDestination));
        QVERIFY(!QFileInfo::exists(invalidDestination + QStringLiteral(".avi")));
        QCOMPARE(QDir(exportedDirectory).entryList(QDir::Files).size(), 2);
        QVERIFY(originalVideo.seek(0));
        QCOMPARE(originalVideo.readAll(), originalBytes);
        QVERIFY(sidecar.seek(0));
        QCOMPARE(sidecar.readAll(), originalMetadata);
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
    }

    void explicitCudaSmokeFailureDoesNotClaimCpuInference()
    {
        const QDir executableDirectory(QCoreApplication::applicationDirPath());
        QString executable = executableDirectory.filePath("bin/vision-studio");
        if (!QFileInfo(executable).isExecutable())
            executable = executableDirectory.filePath("vision-studio");
        QVERIFY2(QFileInfo(executable).isExecutable(),
                 "The application must be built for the CLI integration test.");
        const QString smokeDirectory = fixture_->path() + "/cuda-failure-smoke";
        const QString childDataDirectory = fixture_->path() + "/cuda-failure-data";
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        environment.insert("VISION_STUDIO_DATA_DIR", childDataDirectory);
        environment.insert("VISION_STUDIO_GPU_RUNTIME_DIR", fixture_->path() + "/missing-gpu-runtime");
        environment.insert("QT_QPA_PLATFORM", "offscreen");
        QProcess smoke;
        smoke.setProcessEnvironment(environment);
        smoke.start(executable, {"--device", "cuda", "--gpu-index", "2", "--smoke", smokeDirectory});
        QVERIFY(smoke.waitForStarted(3000));
        QTRY_VERIFY_WITH_TIMEOUT(smoke.state() == QProcess::NotRunning, 20000);
        QCOMPARE(smoke.exitStatus(), QProcess::NormalExit);
        QCOMPARE(smoke.exitCode(), 2);
        QFile reportFile(smokeDirectory + "/smoke-report.json");
        QVERIFY2(reportFile.open(QIODevice::ReadOnly), smoke.readAllStandardError().constData());
        const QJsonObject report = QJsonDocument::fromJson(reportFile.readAll()).object();
        QVERIFY(!report.value("success").toBool());
        QCOMPARE(report.value("requested_device").toString(), QStringLiteral("cuda"));
        QVERIFY(report.value("actual_device").isNull());
        QCOMPARE(report.value("device_index").toInt(), -1);
        QVERIFY(report.value("device_name").toString().isEmpty());
        QVERIFY(report.value("backend").toString().isEmpty());
        QVERIFY(!report.value("error").toString().isEmpty());
        QFile historyFile(childDataDirectory + "/history.json");
        if (historyFile.open(QIODevice::ReadOnly))
            QVERIFY(QJsonDocument::fromJson(historyFile.readAll()).array().isEmpty());
        QVERIFY(QDir(smokeDirectory).entryList({"*.csv", "*.json"}, QDir::Files).size() == 1);
    }

    void automaticDeviceDefaultAndPreferencesPreserveInputModes()
    {
        window_->close();
        window_.reset();
        {
            QSettings settings(fixture_->path() + "/output/preferences.ini", QSettings::IniFormat);
            settings.remove("computeDevice");
            settings.remove("deviceIndex");
            settings.sync();
        }
        showNewWindow();
        auto *device = window_->findChild<QComboBox *>("computeDevice");
        auto *index = window_->findChild<QComboBox *>("gpuDeviceIndex");
        auto *color = window_->findChild<QComboBox *>("inputColorMode");
        auto *stereo = window_->findChild<QComboBox *>("stereoView");
        auto *actual = window_->findChild<QLabel *>("actualDeviceBadge");
        QVERIFY(device && index && color && stereo && actual);
        QCOMPARE(device->currentData().toString(), QStringLiteral("auto"));
        QVERIFY(device->isVisible());
        QVERIFY(actual->text().contains(QStringLiteral("待运行")));
        window_->setComputeDevice(vision::ComputeDevice::CUDA, 2);
        color->setCurrentIndex(color->findData("grayscale"));
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("摄像头")), Qt::LeftButton);
        stereo->setCurrentIndex(stereo->findData("left"));
        QCOMPARE(index->currentData().toInt(), 2);
        QVERIFY(index->isVisible());
        QCOMPARE(window_->size(), QSize(1260, 820));
        QVERIFY(window_->rect().contains(QRect(device->mapTo(window_.get(), QPoint()), device->size())));
        QVERIFY(window_->rect().contains(QRect(index->mapTo(window_.get(), QPoint()), index->size())));
        window_->saveScreenshot(QStringLiteral(VISION_PROJECT_DIR) +
                                "/output/test-screenshots/1.6-device-camera.png");
        window_->close();
        window_.reset();
        showNewWindow();
        device = window_->findChild<QComboBox *>("computeDevice");
        index = window_->findChild<QComboBox *>("gpuDeviceIndex");
        color = window_->findChild<QComboBox *>("inputColorMode");
        stereo = window_->findChild<QComboBox *>("stereoView");
        QCOMPARE(device->currentData().toString(), QStringLiteral("cuda"));
        QCOMPARE(index->currentData().toInt(), 2);
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        QCOMPARE(stereo->currentData().toString(), QStringLiteral("left"));
        QVERIFY(activateModel(window_.get(), QStringLiteral("yolov8n.pt")));
        QCOMPARE(device->currentData().toString(), QStringLiteral("cuda"));
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        window_->setComputeDevice(vision::ComputeDevice::CPU);
        QCOMPARE(color->currentData().toString(), QStringLiteral("grayscale"));
        QVERIFY(!index->isVisible());
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
    }

    void gpuPreparationReportsProgressCancelsAndRefreshesWithoutBlocking()
    {
        QTest::mouseClick(findButton(window_.get(), QStringLiteral("更多")), Qt::LeftButton);
        auto *check = window_->findChild<QPushButton *>("gpuCheckButton");
        auto *prepare = window_->findChild<QPushButton *>("gpuPrepareButton");
        auto *cancel = window_->findChild<QPushButton *>("gpuCancelButton");
        auto *status = window_->findChild<QLabel *>("gpuEnvironmentStatus");
        auto *log = window_->findChild<QPlainTextEdit *>("gpuSetupLog");
        auto *progress = window_->findChild<QProgressBar *>("gpuSetupProgress");
        auto *device = window_->findChild<QComboBox *>("computeDevice");
        QVERIFY(check && prepare && cancel && status && log && progress && device);
        QTRY_VERIFY_WITH_TIMEOUT(check->isEnabled(), 5000);
        QVERIFY(status->text().contains(QStringLiteral("尚未就绪")));
        QVERIFY(status->text().contains(QStringLiteral("测试 NVIDIA GPU")));
        QFile mode(fixture_->path() + "/scripts/setup-mode.txt");
        QVERIFY(mode.open(QIODevice::WriteOnly));
        QCOMPARE(mode.write("wait"), qint64(4));
        mode.close();
        QTest::mouseClick(prepare, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(cancel->isVisible() && progress->isVisible(), 3000);
        QVERIFY(!prepare->isEnabled() && !check->isEnabled());
        QVERIFY(!device->isEnabled());
        QVERIFY(!findButton(window_.get(), QStringLiteral("开始检测"))->isEnabled());
        int ticks = 0;
        QTimer heartbeat;
        connect(&heartbeat, &QTimer::timeout, this, [&ticks] { ++ticks; });
        heartbeat.start(20);
        QTest::qWait(200);
        QVERIFY(ticks >= 3);
        QTRY_VERIFY_WITH_TIMEOUT(log->toPlainText().contains(QStringLiteral("正在准备测试环境")), 3000);
        QCOMPARE(progress->value(), 17);
        QPointer<QProcess> setup = window_->findChild<QProcess *>("gpuSetupProcess");
        QVERIFY(setup && setup->state() != QProcess::NotRunning);
        QTest::mouseClick(cancel, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!cancel->isVisible() && prepare->isEnabled(), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(setup.isNull(), 3000);
        QVERIFY(status->text().contains(QStringLiteral("已取消")));
        QVERIFY(!QFileInfo::exists(fixture_->path() + "/output/gpu-runtime/mock-ready"));
        QVERIFY(device->isEnabled());
        QVERIFY(QFile::remove(mode.fileName()));
        QTest::mouseClick(prepare, Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(status->text().contains(QStringLiteral("已就绪")) && check->isEnabled(),
                                 5000);
        QVERIFY(QFileInfo::exists(fixture_->path() + "/output/gpu-runtime/mock-ready"));
        QCOMPARE(device->currentData().toString(), QStringLiteral("cpu"));
        QVERIFY(window_->findChild<QLabel *>("actualDeviceBadge")->text().contains(QStringLiteral("待运行")));
        window_->saveScreenshot(QStringLiteral(VISION_PROJECT_DIR) +
                                "/output/test-screenshots/1.6-gpu-more.png");
        QCOMPARE(window_->size(), QSize(1260, 820));
        QVERIFY2(unexpectedDialog_.isEmpty(), qPrintable(unexpectedDialog_));
    }

  private:
    void prepareGpuUiFixtures()
    {
        const QString scripts = fixture_->path() + QStringLiteral("/scripts");
        auto write = [](const QString &path, const QByteArray &data)
        {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
            QCOMPARE(file.write(data), data.size());
        };
        write(scripts + "/gpu_probe.py", R"PY(import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime-dir',required=True);a=p.parse_args()
root=Path(a.runtime_dir);ready=(root/'mock-ready').is_file()
print(json.dumps({'ok':True,'prepared':ready,'cuda_available':ready,'reason':'' if ready else '测试环境未准备','devices':[{'index':0,'name':'测试 NVIDIA GPU','total_memory_mb':8192}],'torch_version':'mock','torch_cuda':'mock','ort_version':'mock','driver_version':'mock'}))
)PY");
        write(scripts + "/gpu_setup.py", R"PY(import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime-dir',required=True);a=p.parse_args()
root=Path(a.runtime_dir);root.mkdir(parents=True,exist_ok=True)
print(json.dumps({'event':'progress','message':'开始测试准备','progress':11}),flush=True)
print(json.dumps({'event':'progress','message':'正在准备测试环境','percent':17}),flush=True)
mode=Path(__file__).with_name('setup-mode.txt')
if mode.is_file() and mode.read_text().strip()=='wait':time.sleep(30)
else:time.sleep(.25)
(root/'mock-ready').write_text('UI test fixture; no real GPU dependencies')
print(json.dumps({'event':'ready','runtime_dir':str(root),'message':'测试环境准备完成','progress':100}),flush=True)
)PY");
        QString realPython = QString::fromUtf8(originalPython_);
        if (!QFileInfo(realPython).isExecutable())
        {
            realPython = QStringLiteral(VISION_PROJECT_DIR) + "/runtime/bin/python";
            if (!QFileInfo(realPython).isExecutable())
                realPython = QStringLiteral(VISION_PROJECT_DIR) + "/build/release-runtime/bin/python";
            if (!QFileInfo(realPython).isExecutable())
                realPython = QDir::homePath() + "/VisionStudio/runtime/bin/python";
        }
        QVERIFY(QFileInfo(realPython).isExecutable());
        const QString launcher = scripts + "/ui-python";
        const QByteArray wrapper =
            "#!/usr/bin/python3.10\nimport os,sys\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\nprobe=args[1:] if args and args[0]=='-u' else args\n"
            "name=Path(probe[0]).name if probe else ''\n"
            "if name in ('gpu_probe.py','gpu_setup.py'):\n"
            " os.execv('/usr/bin/python3.10',['/usr/bin/"
            "python3.10','-u',str(Path(__file__).with_name(name)),*probe[1:]])\n"
            "else:\n real=" +
            QJsonDocument(QJsonArray{realPython}).toJson(QJsonDocument::Compact).trimmed() +
            "[0]\n os.execv(real,[real,*args])\n";
        write(launcher, wrapper);
        QVERIFY(QFile::setPermissions(launcher, QFileDevice::ReadOwner | QFileDevice::WriteOwner |
                                                    QFileDevice::ExeOwner | QFileDevice::ReadGroup |
                                                    QFileDevice::ExeGroup));
        qputenv("VISION_STUDIO_PYTHON", launcher.toUtf8());
    }

    void showNewWindow()
    {
        window_ = std::make_unique<MainWindow>();
        window_->resize(1260, 820);
        window_->show();
        window_->activateWindow();
        QTest::qWait(50);
    }

    QByteArray originalHome_;
    QByteArray originalData_;
    QByteArray originalPython_, originalGpuRuntime_;
    bool hadHome_ = false;
    bool hadData_ = false;
    bool hadPython_ = false, hadGpuRuntime_ = false;
    QString unexpectedDialog_;
    std::unique_ptr<QTemporaryDir> fixture_;
    std::unique_ptr<QTimer> dialogGuard_;
    std::unique_ptr<MainWindow> window_;
};

int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication application(argc, argv);
    UiTests tests;
    return QTest::qExec(&tests, argc, argv);
}
#include "ui_tests.moc"
