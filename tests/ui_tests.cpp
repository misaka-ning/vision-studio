#include "ui/imagecanvas.h"
#include "ui/mainwindow.h"

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QPixmap>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QUrl>
#include <algorithm>
#include <limits>
#include <memory>

namespace
{
QPushButton *findButton(QWidget *parent, const QString &text)
{
    for (auto *button : parent->findChildren<QPushButton *>())
        if (button->text() == text)
            return button;
    return nullptr;
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
} // namespace

class UiTests final : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        originalHome_ = qgetenv("VISION_STUDIO_HOME");
        hadHome_ = qEnvironmentVariableIsSet("VISION_STUDIO_HOME");
        QApplication::setStyle(QStringLiteral("Fusion"));
        QApplication::setFont(QFont(QStringLiteral("Noto Sans CJK SC"), 10));
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
        qputenv("VISION_STUDIO_HOME", fixture_->path().toUtf8());
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
    }

    void realInferenceSelectionAndNavigation()
    {
        auto *canvas = window_->findChild<ImageCanvas *>();
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

        auto *pages = window_->findChild<QStackedWidget *>();
        auto *title = window_->findChild<QLabel *>(QStringLiteral("pageTitle"));
        QVERIFY(pages);
        QVERIFY(title);
        for (const auto &route :
             {qMakePair(QStringLiteral("模型库"), 1), qMakePair(QStringLiteral("运行记录"), 2),
              qMakePair(QStringLiteral("使用指南"), 3), qMakePair(QStringLiteral("检测工作台"), 0)})
        {
            auto *nav = findButton(window_.get(), route.first);
            QVERIFY(nav);
            QTest::mouseClick(nav, Qt::LeftButton);
            QCOMPARE(pages->currentIndex(), route.second);
            QCOMPARE(title->text(), route.first);
            QVERIFY(nav->isChecked());
        }
        QCOMPARE(window_->size(), QSize(1260, 820));
        const QRect canvasBounds(canvas->mapTo(window_.get(), QPoint()), canvas->size());
        QVERIFY(window_->rect().contains(canvasBounds));
        QVERIFY(canvas->width() >= 300 && canvas->height() >= 330);
        QVERIFY(findButton(window_.get(), QStringLiteral("开始检测"))->isVisible());
        const QString screenshot = qEnvironmentVariable("VISION_UI_TEST_SCREENSHOT");
        if (!screenshot.isEmpty())
            window_->saveScreenshot(screenshot);
    }

    void shortcutsCancelAndRestartFromFocusedControls()
    {
        auto *canvas = window_->findChild<ImageCanvas *>();
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
        auto *canvas = window_->findChild<ImageCanvas *>();
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
        auto *canvas = window_->findChild<ImageCanvas *>();
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

  private:
    QByteArray originalHome_;
    bool hadHome_ = false;
    QString unexpectedDialog_;
    std::unique_ptr<QTemporaryDir> fixture_;
    std::unique_ptr<QTimer> dialogGuard_;
    std::unique_ptr<MainWindow> window_;
};

QTEST_MAIN(UiTests)
#include "ui_tests.moc"
