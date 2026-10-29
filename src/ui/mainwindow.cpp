#include "runtimesetup.h"
#include "app/historyrepository.h"
#include "app/gpuenvironmentservice.h"
#include "core/cpuruntime.h"
#include "mainwindow.h"
#include "core/gpuruntime.h"
#include "core/inferenceworker.h"
#include "core/evaluationworker.h"
#include "evaluationpanel.h"
#include "core/visionengine.h"
#include "icons.h"
#include "imagecanvas.h"
#include "lazyimagequeue.h"
#include "asyncimagepreview.h"
#include "modelviewer.h"
#include "modelconversionpage.h"
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDataStream>
#include <QCryptographicHash>
#include <QDebug>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHeaderView>
#include <QImageReader>
#include <QImageWriter>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMouseEvent>
#include <QMessageBox>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QPalette>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QRunnable>
#include <QThreadPool>
#include <QWheelEvent>
#include <QSaveFile>
#include <QScrollArea>
#include <QSet>
#include <QSettings>
#include <QShortcut>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QUuid>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{
const QString modelLibraryDropPrompt = QStringLiteral("将 ONNX / PT / TorchScript 文件拖到此处，或点击导入；支持多选，原文件保留。");
enum class PreviewDropKind { Images, Dataset, Video };
bool classifyPreviewDrop(const QStringList &input, QStringList *paths, PreviewDropKind *kind, QString *error)
{
    paths->clear();
    if (input.isEmpty()) { *error = "请拖入本地文件。"; return false; }
    const QStringList videos{"mp4", "avi", "mkv", "mov", "webm", "m4v"};
    QSet<QString> imageFormats;
    for (const auto &format : QImageReader::supportedImageFormats()) imageFormats.insert(QString::fromLatin1(format).toLower());
    // Drag-move events must not stat thousands of files on the GUI thread.
    // Actual files, readability and canonical identities are checked atomically
    // by importDroppedImages before any queue change.
    if (input.size() > 96 && std::all_of(input.begin(), input.end(), [&imageFormats](const QString &path) {
            return imageFormats.contains(QFileInfo(path).suffix().toLower()); })) {
        *paths = input; *kind = PreviewDropKind::Images; return true;
    }
    QSet<QString> seen;
    int category = -1;
    for (const QString &path : input) {
        const QFileInfo info(path);
        if (!info.exists() || !info.isReadable() || (!info.isFile() && !info.isDir())) {
            *error = "无法读取拖入的路径：" + path; return false;
        }
        const QString suffix = info.suffix().toLower();
        const int next = info.isDir() || suffix == "yaml" || suffix == "yml" ? 1 :
            videos.contains(suffix) ? 2 : imageFormats.contains(suffix) ? 0 : -1;
        if (next < 0) { *error = "不支持此文件。请拖入图片、视频或 YOLO 数据集目录 / YAML：" + path; return false; }
        if (category >= 0 && category != next) {
            *error = "一次拖入请只包含图片，或一个数据集目录 / YAML，或一个视频；不能混合输入类型。"; return false;
        }
        category = next;
        const QString canonical = info.canonicalFilePath();
        const QString absolute = next == 0 && !canonical.isEmpty() ? canonical : info.absoluteFilePath();
        if (!seen.contains(absolute)) { seen.insert(absolute); paths->append(absolute); }
    }
    if (category != 0 && input.size() != 1) {
        *error = "数据集或视频每次只能拖入一个；多张图片可以一起拖入。"; return false;
    }
    *kind = category == 1 ? PreviewDropKind::Dataset : category == 2 ? PreviewDropKind::Video : PreviewDropKind::Images;
    return true;
}

class ModelLibraryList final : public QListWidget
{
  public:
    using QListWidget::QListWidget;
    void setLibraryBusy(bool busy)
    {
        busy_ = busy;
        setDragEnabled(!busy);
        setAcceptDrops(!busy);
    }

  protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        // A context action targets the item under the pointer without selecting
        // a different global inference model, including while inference runs.
        if (busy_ || event->button() == Qt::RightButton)
            event->accept();
        else
            QListWidget::mousePressEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (busy_ || event->button() == Qt::RightButton)
            event->accept();
        else
            QListWidget::mouseReleaseEvent(event);
    }
    void mouseDoubleClickEvent(QMouseEvent *event) override
    {
        if (busy_ || event->button() == Qt::RightButton)
            event->accept();
        else
            QListWidget::mouseDoubleClickEvent(event);
    }
    void keyPressEvent(QKeyEvent *event) override
    {
        if (busy_ && event->key() != Qt::Key_Menu &&
            !(event->key() == Qt::Key_F10 && event->modifiers() == Qt::ShiftModifier))
            event->accept();
        else
            QListWidget::keyPressEvent(event);
    }

  private:
    bool busy_ = false;
};

template <typename Base> class WheelSafeSpinBox final : public Base
{
  public:
    using Base::Base;
  protected:
    void wheelEvent(QWheelEvent *event) override
    {
        // Ignoring lets the surrounding configuration panel receive scrolling,
        // even when this input currently has keyboard focus.
        event->ignore();
    }
};
vision::ComputeDevice computeMode(const QString &value)
{
    return value == "cuda" ? vision::ComputeDevice::CUDA
                           : (value == "cpu" ? vision::ComputeDevice::CPU : vision::ComputeDevice::Auto);
}
QString computeName(vision::ComputeDevice value)
{
    return value == vision::ComputeDevice::CUDA
               ? QStringLiteral("NVIDIA GPU")
               : (value == vision::ComputeDevice::CPU ? QStringLiteral("CPU") : QStringLiteral("自动"));
}
QString stereoCode(vision::StereoView view)
{
    return view == vision::StereoView::Left ? "left" : (view == vision::StereoView::Right ? "right" : "full");
}
QString stereoName(vision::StereoView view)
{
    return view == vision::StereoView::Left ? "双目左目"
                                            : (view == vision::StereoView::Right ? "双目右目" : "完整画面");
}
vision::StereoView stereoMode(const QString &value)
{
    return value == "left" ? vision::StereoView::Left
                           : (value == "right" ? vision::StereoView::Right : vision::StereoView::Full);
}
QLabel *text(const QString &value, const char *name = "body")
{
    auto *w = new QLabel(value);
    w->setObjectName(name);
    w->setTextFormat(Qt::PlainText);
    return w;
}
QPushButton *button(const QString &title, const QString &iconName = {}, const char *role = "secondary")
{
    auto *b = new QPushButton(title);
    b->setObjectName(role);
    b->setProperty("fluentRole", QString::fromLatin1(role));
    b->setCursor(Qt::PointingHandCursor);
    b->setMinimumHeight(36);
    if (!iconName.isEmpty())
        b->setIcon(ui::icon(iconName, role == QString("primary") ? QColor("#00374D") : QColor("#D2D2D2")));
    b->setIconSize(QSize(18, 18));
    return b;
}
QFrame *card(const char *name = "card")
{
    auto *w = new QFrame;
    w->setObjectName(name);
    return w;
}

QLabel *section(const QString &title, const QString &number)
{
    auto *l = text(number + "   " + title, "sectionTitle");
    l->setMinimumHeight(20);
    return l;
}
void tableStyle(QTableWidget *t)
{
    t->setAlternatingRowColors(false);
    t->setShowGrid(false);
    t->setSelectionBehavior(QAbstractItemView::SelectRows);
    t->setSelectionMode(QAbstractItemView::SingleSelection);
    t->setEditTriggers(QAbstractItemView::NoEditTriggers);
    t->verticalHeader()->hide();
    t->verticalHeader()->setDefaultSectionSize(42);
    t->horizontalHeader()->setHighlightSections(false);
    t->setFrameShape(QFrame::NoFrame);
    t->setFocusPolicy(Qt::StrongFocus);
}
QString taskName(vision::ModelTask t)
{
    return t == vision::ModelTask::YoloV5   ? "YOLOv5"
           : t == vision::ModelTask::YoloV8 ? "YOLOv8 / 11"
                                            : "图像分类";
}
QString cleanName(const QString &s)
{
    QString v = QFileInfo(s).completeBaseName();
    if (v.isEmpty())
        v = "frame";
    v.replace(QRegularExpression("[^\\p{L}\\p{N}_-]"), "_");
    return v.left(70);
}
bool isPtModel(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == "pt" || suffix == "torchscript";
}
QString modelFormat(const QString &path)
{
    if (QFileInfo(path).suffix().compare("torchscript", Qt::CaseInsensitive) == 0)
        return "TorchScript";
    return isPtModel(path) ? QString("PT") : QString("ONNX");
}
QStringList convertedModelLabels(const QJsonObject &metadata)
{
    QStringList result;
    const auto names = metadata.value("names").toObject();
    if (names.isEmpty() || names.size() > 100000)
        return {};
    for (int index = 0; index < names.size(); ++index)
    {
        const QJsonValue value = names.value(QString::number(index));
        if (!value.isString() || value.toString().trimmed().isEmpty() || value.toString().size() > 1000)
            return {};
        result.append(value.toString());
    }
    return result;
}
bool validConvertedModelMetadata(const QString &path, const QJsonObject &metadata)
{
    const QString format = metadata.value("format").toString();
    const QString suffix = QFileInfo(path).suffix().toLower();
    if ((format != "onnx" || suffix != "onnx") &&
        (format != "torchscript" || (suffix != "torchscript" && suffix != "pt")))
        return false;
    if (QFileInfo(metadata.value("output").toString()).absoluteFilePath() != QFileInfo(path).absoluteFilePath())
        return false;
    const QString task = metadata.value("task").toString(), layout = metadata.value("layout").toString();
    if ((task != "detect" || (layout != "v5" && layout != "v8")) &&
        (task != "classify" || layout != "classify"))
        return false;
    const double channels = metadata.value("input_channels").toDouble(-1);
    if ((channels != 1 && channels != 3) || metadata.value("ch").toDouble(-1) != channels ||
        convertedModelLabels(metadata).isEmpty() || metadata.value("precision").toString() != "fp32" ||
        !metadata.value("nms").isBool() || metadata.value("nms").toBool() ||
        metadata.value("dynamic").toBool())
        return false;
    const auto shape = metadata.value("shape").toArray();
    if (shape.size() != 4 || shape[0].toDouble(-1) != 1 || shape[1].toDouble(-1) != channels)
        return false;
    const double size = shape[2].toDouble(-1);
    return size >= 32 && size <= 4096 && size == int(size) && int(size) % 32 == 0 &&
           shape[3].toDouble(-1) == size;
}
QString fileIdentity(const QString &path)
{
    const QFileInfo file(path);
    return file.isFile() ? file.absoluteFilePath() + "|" + QString::number(file.size()) + "|" +
                              QString::number(file.lastModified().toMSecsSinceEpoch()) : QString();
}
bool atomicWrite(const QString &path, const QByteArray &data, QString *error = nullptr)
{
    QSaveFile f(path);
    if (!f.open(QIODevice::WriteOnly) || f.write(data) != data.size() || !f.commit())
    {
        if (error)
            *error = f.errorString();
        return false;
    }
    return true;
}
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    qRegisterMetaType<vision::InferenceResult>();
    qRegisterMetaType<vision::JobRequest>();
    qRegisterMetaType<vision::ModelConfig>();
    const QString besideExecutable = QDir(QCoreApplication::applicationDirPath() + "/../..").absolutePath();
    const QString runtimeRoot =
        QFileInfo::exists(besideExecutable + "/models") && QFileInfo::exists(besideExecutable + "/assets")
            ? besideExecutable
            : QString(VISION_PROJECT_DIR);
    projectRoot_ = qEnvironmentVariable("VISION_STUDIO_HOME", runtimeRoot);
    if (!qEnvironmentVariableIsSet("VISION_STUDIO_HOME"))
        qputenv("VISION_STUDIO_HOME", projectRoot_.toUtf8());
    const bool systemInstall = qEnvironmentVariable("VISION_STUDIO_SYSTEM_INSTALL") == "1" ||
                               projectRoot_.startsWith("/opt/") || projectRoot_.startsWith("/usr/");
    dataRoot_ = qEnvironmentVariable(
        "VISION_STUDIO_DATA_DIR",
        systemInstall
            ? QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/vision-studio"
            : projectRoot_ + "/output");
    QDir().mkpath(dataRoot_);
    settings_ = new vision::app::PreferencesRepository(dataRoot_ + "/preferences.ini", this);
    exportDir_ = settings_->value("exportDirectory", dataRoot_ + "/results").toString();
    models_ = settings_->value("models").toStringList();
    const bool hasSavedModelLibrary = settings_->contains("models");
    modelDisplayNames_ = settings_->value("modelDisplayNames").toMap();
    modelNotes_ = settings_->value("modelNotes").toMap();
    convertedModelProfiles_ = settings_->value("convertedModelProfiles").toMap();
    modelEvaluationProfiles_ = settings_->value("evaluationProfiles").toMap();
    history_ = vision::app::HistoryRepository(dataRoot_ + "/history.json").load();
    setWindowTitle("Vision Studio — 本地视觉推理工作台");
    resize(1580, 960);
    setMinimumSize(1260, 820);
    setupStyle();
    auto *base = new QWidget;
    auto *body = new QHBoxLayout(base);
    body->setContentsMargins(0, 0, 0, 0);
    body->setSpacing(0);
    body->addWidget(buildSidebar());
    auto *workspace = new QWidget;
    workspace->setObjectName("workspace");
    auto *w = new QVBoxLayout(workspace);
    w->setContentsMargins(24, 12, 24, 0);
    w->setSpacing(12);
    workspaceHeader_ = new QWidget;
    workspaceHeader_->setObjectName("workspaceHeader");
    auto *header = new QHBoxLayout(workspaceHeader_);
    header->setContentsMargins(0, 0, 0, 0);
    auto *titles = new QVBoxLayout;
    titles->setSpacing(4);
    auto *crumb = text("VISION STUDIO  /  工作空间", "eyebrow");
    titles->addWidget(crumb);
    pageTitle_ = text("检测工作台", "pageTitle");
    titles->addWidget(pageTitle_);
    pageSubtitle_ = text("从输入到洞察，让每一次视觉推理清晰可见。", "muted");
    titles->addWidget(pageSubtitle_);
    header->addLayout(titles);
    header->addStretch();
    auto *local = text("●  本地运行", "localBadge");
    header->addWidget(local, 0, Qt::AlignVCenter);
    header->addSpacing(12);
    exportButton_ = button("导出结果", "export");
    exportButton_->setEnabled(false);
    exportButton_->setObjectName("exportResultButton");
    connect(exportButton_, &QPushButton::clicked, this, &MainWindow::exportResult);
    w->addWidget(workspaceHeader_);
    pages_ = new QStackedWidget;
    pages_->setObjectName("workspacePages");
    pages_->addWidget(buildWorkbench());
    pages_->addWidget(buildModels());
    pages_->addWidget(buildModelDisplay());
    pages_->addWidget(buildHistory());
    pages_->addWidget(buildGuide());
    pages_->addWidget(buildRecordings());
    modelConversion_ = new ModelConversionPage(dataRoot_);
    pages_->addWidget(modelConversion_);
    connect(modelConversion_, &ModelConversionPage::chooseLibraryRequested, this, [this] { selectRoute(ui::Route::Models); });
    connect(modelConversion_, &ModelConversionPage::convertedModelReady, this, [this](const QString &path) {
        if (busy_) {
            showNotice("请先结束检测，再将转换结果加入模型库。", true);
            return;
        }
        if (!rememberConvertedModel(path, modelConversion_->completedMetadata()))
            return;
        setModel(path, true);
        selectRoute(ui::Route::Models);
    });
    connect(modelConversion_, &ModelConversionPage::busyChanged, this, [this](bool active) {
        updateStartEligibility();
        if (!active && closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    pages_->addWidget(buildMore());
    w->addWidget(pages_, 1);
    auto *footer = new QFrame;
    footer->setObjectName("footer");
    auto *foot = new QHBoxLayout(footer);
    foot->setContentsMargins(0, 6, 0, 6);
    statusLabel_ = text("●  准备就绪 · 选择模型与输入后开始检测", "statusText");
    statusLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    statusLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    statusLabel_->setMaximumHeight(20);
    foot->addWidget(statusLabel_, 1);
    progress_ = new QProgressBar;
    progress_->setObjectName("detectionProgress");
    progress_->setFixedSize(180, 5);
    progress_->setTextVisible(false);
    progress_->hide();
    foot->addWidget(progress_);
    foot->addSpacing(16);
    backendFooter_ = text("Qt 6.8.3  ·  推理尚未开始", "tiny");
    backendFooter_->setObjectName("actualDeviceFooter");
    backendFooter_->setFixedSize(310, 20);
    backendFooter_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    backendFooter_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    foot->addWidget(backendFooter_);
    w->addWidget(footer);
    body->addWidget(workspace, 1);
    setCentralWidget(base);
    qApp->installEventFilter(this);
    resultWriter_ = new vision::app::ResultWriter(this);
    connect(resultWriter_, &vision::app::ResultWriter::pendingChanged, this, [this](int count) {
        resultIoPending_ = count;
        if (!busy_ && !count) {
            updateStartEligibility();
            if (closing_) QTimer::singleShot(0, this, &QWidget::close);
        }
    });
    connect(resultWriter_, &vision::app::ResultWriter::completed, this,
        [this](const vision::app::ResultWriteOutcome &outcome) {
        if (outcome.generation == imageResultGeneration_) {
            auto cached = imageResults_.find(outcome.source);
            if (cached != imageResults_.end()) {
                if (outcome.snapshotSaved) {
                    cached->snapshotPath = outcome.snapshotPath;
                    cached->result.image = {}; cached->result.originalImage = {};
                } else if (!imageResultCacheWarning_) {
                    imageResultCacheWarning_ = true;
                    showNotice("无法写入临时结果缓存，本轮结果保存在内存中。请检查临时目录空间或权限。", true);
                }
            }
            if (!outcome.historySaved) showNotice("运行记录保存失败：" + outcome.historyError, true);
            if (outcome.automaticExport && !outcome.exportSaved) {
                failed_ = true; lastError_ = "自动导出失败：" + outcome.exportError;
                detection_->requestStop(); showNotice(lastError_, true);
            }
        }
        detection_->acknowledgeImageResult(outcome.ticket);
    });
    gpuService_ = new vision::app::GpuEnvironmentService(projectRoot_, this);
    connect(gpuService_, &vision::app::GpuEnvironmentService::environmentChanged, this,
        [this](const QJsonObject &environment) {
        const int selected = gpuDeviceIndex_->currentData().toInt();
        const QSignalBlocker blocker(gpuDeviceIndex_);
        gpuDeviceIndex_->clear();
        for (const auto &value : environment.value("devices").toArray()) {
            const auto device = value.toObject();
            const int index = device.value("index").toInt();
            gpuDeviceIndex_->addItem(QString("GPU %1 · %2").arg(index).arg(device.value("name").toString()), index);
        }
        if (gpuDeviceIndex_->findData(selected) < 0)
            gpuDeviceIndex_->addItem(QString("GPU %1 · 当前未检测到").arg(selected), selected);
        gpuDeviceIndex_->setCurrentIndex(gpuDeviceIndex_->findData(selected));
        updateDeviceUi();
    });
    connect(gpuService_, &vision::app::GpuEnvironmentService::statusChanged,
        gpuEnvironmentStatus_, &QLabel::setText);
    connect(gpuService_, &vision::app::GpuEnvironmentService::logReset, gpuLog_, &QPlainTextEdit::clear);
    connect(gpuService_, &vision::app::GpuEnvironmentService::logAppended, this, &MainWindow::appendGpuLog);
    connect(gpuService_, &vision::app::GpuEnvironmentService::progressChanged, this,
        [this](int minimum, int maximum, int value, bool visible) {
        gpuSetupProgress_->setRange(minimum, maximum); gpuSetupProgress_->setValue(value);
        gpuSetupProgress_->setVisible(visible);
    });
    connect(gpuService_, &vision::app::GpuEnvironmentService::stateChanged, this, &MainWindow::updateDeviceUi);
    connect(gpuService_, &vision::app::GpuEnvironmentService::preparationFinished, this, [this] {
        if (closing_) QTimer::singleShot(0, this, &QWidget::close);
    });
    connectWorker();
    connectEvaluation();
    const int savedTask = settings_->value("task", 0).toInt(),
              savedInput = settings_->value("inputSize", 640).toInt();
    const double savedConfidence = settings_->value("confidence", 0.25).toDouble(),
                 savedIou = settings_->value("iou", 0.45).toDouble();
    const double savedScale = settings_->value("scale", 1.0 / 255.0).toDouble();
    const double savedR = settings_->value("meanR", 0).toDouble(),
                 savedG = settings_->value("meanG", 0).toDouble(),
                 savedB = settings_->value("meanB", 0).toDouble();
    const bool savedSwap = settings_->value("swapRB", true).toBool(),
               savedAuto = settings_->value("autoExport", false).toBool();
    const QString savedColorMode = settings_->value("colorMode", savedSwap ? "rgb" : "bgr").toString();
    const QString savedStereo = settings_->value("stereoView", "full").toString();
    const int savedCameraIndex = settings_->value("cameraIndex", 0).toInt();
    const QString savedDevice = settings_->value("computeDevice", "auto").toString();
    const int savedDeviceIndex = qBound(0, settings_->value("deviceIndex", 0).toInt(), 63);
    const QStringList savedLabels = settings_->value("labels").toStringList();
    const QString savedLabelsPath = settings_->value("labelsPath").toString();
    const QString bundled = projectRoot_ + "/models/yolov5n.onnx";
    const QString saved = settings_->value("activeModel").toString();
    if (!saved.isEmpty() && QFileInfo(saved).isFile() &&
        (!hasSavedModelLibrary || models_.contains(saved)))
        setModel(saved);
    else
    {
        QString next;
        for (const QString &candidate : models_)
            if (QFileInfo(candidate).isFile())
            {
                next = candidate;
                break;
            }
        if (!next.isEmpty())
            setModel(next);
        else if (!hasSavedModelLibrary && QFileInfo(bundled).isFile())
            setModel(bundled);
    }
    if (!hasSavedModelLibrary)
    {
        for (const QString &name : {QStringLiteral("yolov8n.pt"), QStringLiteral("yolov5n.pt")})
        {
            const QString example = projectRoot_ + "/models/" + name;
            if (QFileInfo(example).isFile() && !models_.contains(example))
                models_.append(example);
        }
    }
    taskBox_->setCurrentIndex(qBound(0, savedTask, 2));
    inputSize_->setValue(savedInput);
    confidence_->setValue(savedConfidence);
    iou_->setValue(savedIou);
    scale_->setValue(savedScale);
    meanR_->setValue(savedR);
    meanG_->setValue(savedG);
    meanB_->setValue(savedB);
    const int savedColorIndex = inputColorMode_->findData(savedColorMode);
    inputColorMode_->setCurrentIndex(savedColorIndex < 0 ? 0 : savedColorIndex);
    const int savedStereoIndex = stereoView_->findData(savedStereo);
    stereoView_->setCurrentIndex(savedStereoIndex < 0 ? 0 : savedStereoIndex);
    cameraIndex_->setValue(savedCameraIndex);
    {
        const QSignalBlocker deviceBlocker(computeDevice_), indexBlocker(gpuDeviceIndex_);
        const int deviceRow = computeDevice_->findData(savedDevice);
        computeDevice_->setCurrentIndex(deviceRow < 0 ? 0 : deviceRow);
        if (gpuDeviceIndex_->findData(savedDeviceIndex) < 0)
            gpuDeviceIndex_->addItem(QString("GPU %1 · 待检查").arg(savedDeviceIndex), savedDeviceIndex);
        gpuDeviceIndex_->setCurrentIndex(gpuDeviceIndex_->findData(savedDeviceIndex));
    }
    autoExport_->setChecked(savedAuto);
    classFilter_->setText(settings_->value("classFilter").toString());
    labels_ = savedLabels;
    labelsPath_ = savedLabelsPath;
    if (!labels_.isEmpty())
    {
        labelButton_->setText(QString("类别标签 · %1 个").arg(labels_.size()));
        labelButton_->setToolTip(labelsPath_);
    }
    applyConvertedModelProfile(modelPath_);
    canvas_->clear();
    updateSourceUi();
    refreshModelLibrary();
    refreshHistory();
    refreshRecordings();
    selectRoute(ui::Route::Workbench);
    updateTaskUi();
    updateDeviceUi();
    initializing_ = false;
    updateRangeUi();
    updateClassFilterUi();
    inspectSelectedModel();
    persist();
    QTimer::singleShot(0, this, &MainWindow::checkGpuEnvironment);
    auto shortcut = [this](const QKeySequence &keys, auto action)
    {
        auto *s = new QShortcut(keys, this);
        s->setContext(Qt::WindowShortcut);
        connect(s, &QShortcut::activated, this, action);
    };
    shortcut(QKeySequence("Ctrl+O"),
             [this]
             {
                 if (!busy_)
                     evaluation_->chooseDataset();
             });
    shortcut(QKeySequence("Ctrl+M"),
             [this]
             {
                 if (!busy_)
                     importModel();
             });
    shortcut(QKeySequence("Ctrl+R"), [this] { startWorkbenchTask(); });
    shortcut(QKeySequence("Ctrl+E"),
             [this]
             {
                 if (!busy_)
                     exportResult();
             });
    shortcut(QKeySequence(Qt::Key_Escape),
             [this]
             {
                 if (modelConversion_ && modelConversion_->isBusy() && pages_->currentIndex() == ui::routeIndex(ui::Route::ModelConversion))
                     modelConversion_->cancel();
                 else if (busy_)
                     stopInference();
                 else
                 {
                     predictionTable_->clearSelection();
                     canvas_->setSelectedPrediction(-1);
                 }
             });
}
MainWindow::~MainWindow()
{
    qApp->removeEventFilter(this);
    if (droppedImageImportCancellation_) droppedImageImportCancellation_->store(true);
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    if (modelInspectionCancellation_)
        modelInspectionCancellation_->store(true);
    gpuService_->requestShutdown();
    if (modelViewer_)
        modelViewer_->stop();
    stopRecordingPlayback();
    if (evaluationWorker_) {
        evaluationWorker_->requestStop();
        evaluationWorker_->wait();
    }
    detection_->requestShutdown();
    resultWriter_->requestShutdown();
}

void MainWindow::setupStyle()
{
    // Apply the visual layer to the existing widget tree; native actions stay intact.
    QPalette colors = palette();
    colors.setColor(QPalette::Window, QColor("#202020"));
    colors.setColor(QPalette::WindowText, QColor("#F5F5F5"));
    colors.setColor(QPalette::Base, QColor("#303030"));
    colors.setColor(QPalette::AlternateBase, QColor("#272727"));
    colors.setColor(QPalette::Text, QColor("#F5F5F5"));
    colors.setColor(QPalette::Button, QColor("#333333"));
    colors.setColor(QPalette::ButtonText, QColor("#F5F5F5"));
    colors.setColor(QPalette::Highlight, QColor("#354C59"));
    colors.setColor(QPalette::HighlightedText, Qt::white);
    colors.setColor(QPalette::PlaceholderText, QColor("#A5A5A5"));
    colors.setColor(QPalette::Disabled, QPalette::Text, QColor("#858585"));
    colors.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#858585"));
    // Style-sheet popup frames and their menu margins resolve the application
    // palette, even when the item view inherits this window's dark stylesheet.
    QApplication::setPalette(colors);
    setPalette(colors);
    QFile theme(QStringLiteral(":/fluent-dark.qss"));
    if (theme.open(QIODevice::ReadOnly))
        setStyleSheet(QString::fromUtf8(theme.readAll()) + QStringLiteral(
            "QTabWidget::pane { background:#202020; border:1px solid #353535; border-radius:6px; top:-1px; }"
            "QTabBar::tab { background:#272727; color:#B8B8B8; border:1px solid #353535; border-bottom:2px solid transparent; border-top-left-radius:4px; border-top-right-radius:4px; padding:8px 14px; margin-right:4px; }"
            "QTabBar::tab:selected { background:#303030; color:#F5F5F5; border-bottom:2px solid #60CDFF; }"
            "QTabBar::tab:hover { background:#353535; color:#F5F5F5; }"
            "QTabBar::tab:disabled { color:#828282; background:#252525; }"));
}

QWidget *MainWindow::buildSidebar()
{
    auto *side = new QWidget;
    side->setObjectName("sidebar");
    side->setFixedWidth(212);
    auto *l = new QVBoxLayout(side);
    l->setContentsMargins(12, 24, 12, 20);
    l->setSpacing(8);
    auto *brand = new QHBoxLayout;
    auto *logo = new QLabel;
    logo->setPixmap(QIcon(":/app-icon.svg").pixmap(30, 30));
    brand->addWidget(logo);
    brand->addWidget(text("Vision Studio", "brand"));
    brand->addStretch();
    l->addLayout(brand);
    auto *cap = text(QCoreApplication::applicationVersion() + "  ·  本地视觉工作空间", "brandCaption");
    cap->setContentsMargins(4, 4, 0, 0);
    l->addWidget(cap);
    l->addSpacing(24);
    l->addWidget(text("工作空间", "eyebrow"));
    l->addSpacing(5);
    const QStringList titles = {"检测工作台", "模型库",   "模型显示", "运行记录",
                                "使用指南",   "录制视频", "模型转换", "更多"};
    const QStringList icons = {"work", "model", "graph", "history", "help", "video", "convert", "more"};
    for (int i = 0; i < titles.size(); ++i)
    {
        auto *b = button(titles[i], icons[i], "nav");
        b->setCheckable(true);
        navButtons_.append(b);
        connect(b, &QPushButton::clicked, this, [this, i] { selectRoute(i); });
    }
    // Keep route IDs stable while placing help beside the application settings.
    QPushButton *previousEntry = nullptr;
    for (int route : {0, 1, 2, 3, 5, 6, 4, 7}) {
        l->addWidget(navButtons_[route]);
        if (previousEntry) QWidget::setTabOrder(previousEntry, navButtons_[route]);
        previousEntry = navButtons_[route];
    }
    l->addStretch();
    auto *info = card();
    auto *il = new QVBoxLayout(info);
    il->setContentsMargins(12, 14, 12, 14);
    il->setSpacing(7);
    il->addWidget(text("●  本机工作空间", "tiny"));
    auto *t = text("模型与图像\n始终留在本机", "body");
    t->setStyleSheet("font-size: 12px; color: #D2D2D2;");
    il->addWidget(t);
    il->addWidget(text("无需账户 · 离线推理", "tiny"));
    l->addWidget(info);
    l->addSpacing(12);
    l->addWidget(text("VISION STUDIO\nLocal inference, clear insight.", "tiny"));
    return side;
}

QWidget *MainWindow::buildWorkbench()
{
    auto *page = new QWidget;
    page->setObjectName("workbenchPage");
    auto *all = new QVBoxLayout(page);
    all->setContentsMargins(0, 0, 0, 0);
    all->setSpacing(12);
    auto *metrics = new QHBoxLayout;
    metrics->setSpacing(12);
    metrics->setContentsMargins(0, 0, 0, 0);
    auto metric = [&](const QString &label, const QString &symbol, QLabel *&value, const QString &unit)
    {
        auto *f = card("metricCard");
        auto *x = new QVBoxLayout(f);
        x->setContentsMargins(18, 8, 18, 8);
        x->setSpacing(5);
        auto *top = new QHBoxLayout;
        top->addWidget(text(label, "muted"));
        top->addStretch();
        auto *ic = new QLabel;
        ic->setPixmap(ui::icon(symbol, QColor("#60CDFF"), 17).pixmap(17, 17));
        top->addWidget(ic);
        x->addLayout(top);
        auto *row = new QHBoxLayout;
        value = text("—", "metricValue");
        row->addWidget(value);
        row->addWidget(text(unit, "metricUnit"), 0, Qt::AlignBottom);
        row->addStretch();
        x->addLayout(row);
        metrics->addWidget(f, 1);
    };
    metric("检测目标", "eye", countMetric_, "objects");
    metric("模型推理", "cpu", latencyMetric_, "ms");
    metric("识别类别", "model", classMetric_, "classes");
    metric("输入分辨率", "image", sizeMetric_, "px");
    auto *previewMetrics = new QWidget;
    previewMetrics->setObjectName("previewMetrics");
    previewMetrics->setLayout(metrics);
    all->addWidget(previewMetrics);
    auto *inputCommands = card("commandBar");
    auto *commandLayout = new QVBoxLayout(inputCommands);
    commandLayout->setContentsMargins(12, 7, 12, 7);
    commandLayout->setSpacing(4);
    auto *commands = new QHBoxLayout;
    commands->setSpacing(6);
    evaluation_ = new EvaluationPanel;
    auto *datasetButton = button("YOLO 数据集", "folder");
    datasetButton->setObjectName("openYoloDatasetButton");
    datasetButton->setToolTip("选择标准 YOLO 数据集文件夹或 YAML 配置 · Ctrl+O");
    evaluation_->setModeControlVisible(false);
    commands->addWidget(datasetButton);
    lockedControls_.append(datasetButton);
    connect(datasetButton, &QPushButton::clicked, evaluation_, &EvaluationPanel::chooseDataset);
    auto *video = button("视频", "video");
    video->setObjectName("openVideoButton");
    auto *camera = button("摄像头", "camera");
    camera->setObjectName("openCameraButton");
    commands->addWidget(video); commands->addWidget(camera);
    lockedControls_.append(video); lockedControls_.append(camera);
    connect(video, &QPushButton::clicked, this, &MainWindow::chooseVideo);
    connect(camera, &QPushButton::clicked, this, &MainWindow::chooseCamera);
    sourceLabel_ = text("选择数据集或连接实时输入", "tiny");
    sourceLabel_->setObjectName("workbenchSourceSummary");
    sourceLabel_->setWordWrap(false);
    sourceLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    commands->addWidget(sourceLabel_, 1);
    commands->addStretch();
    commands->addSpacing(8);
    commands->addWidget(text("设备", "tiny"));
    computeDevice_ = new QComboBox;
    computeDevice_->setObjectName("computeDevice");
    computeDevice_->setAccessibleName("推理设备");
    computeDevice_->addItem("自动", "auto");
    computeDevice_->addItem("CPU", "cpu");
    computeDevice_->addItem("NVIDIA GPU", "cuda");
    computeDevice_->setFixedWidth(140);
    commands->addWidget(computeDevice_);
    gpuDeviceLabel_ = text("GPU", "tiny");
    commands->addWidget(gpuDeviceLabel_);
    gpuDeviceIndex_ = new QComboBox;
    gpuDeviceIndex_->setObjectName("gpuDeviceIndex");
    gpuDeviceIndex_->setAccessibleName("GPU 设备编号");
    gpuDeviceIndex_->addItem("GPU 0 · 待检查", 0);
    gpuDeviceIndex_->setFixedWidth(190);
    commands->addWidget(gpuDeviceIndex_);
    deviceHint_ = text("●", "tiny");
    deviceHint_->setObjectName("deviceEnvironmentHint");
    deviceHint_->setFixedWidth(14);
    commands->addWidget(deviceHint_);
    commandLayout->addLayout(commands);
    commandLayout->addWidget(evaluation_->controlsWidget());
    all->addWidget(inputCommands);
    auto *columns = new QSplitter;
    columns->setChildrenCollapsible(false);
    columns->setHandleWidth(12);
    auto *config = card();
    config->setMinimumWidth(260);
    config->setMaximumWidth(320);
    auto *cl = new QVBoxLayout(config);
    cl->setContentsMargins(14, 14, 14, 14);
    cl->setSpacing(10);
    cl->addWidget(text("检测配置", "sectionTitle"));
    lockedControls_.append(computeDevice_);
    lockedControls_.append(gpuDeviceIndex_);
    const auto deviceChanged = [this]
    {
        if (busy_)
            return;
        actualDeviceKnown_ = false;
        actualBackend_.clear();
        actualDeviceNotice_.clear();
        updateDeviceUi();
        updateInputPreview();
        persist();
    };
    connect(computeDevice_, &QComboBox::currentIndexChanged, this, deviceChanged);
    connect(gpuDeviceIndex_, &QComboBox::currentIndexChanged, this, deviceChanged);
    auto *scroll = new QScrollArea;
    scroll->setObjectName("configurationScrollArea");
    scroll->setWidgetResizable(true);
    scroll->viewport()->setAutoFillBackground(false);
    auto *inner = new QWidget;
    inner->setObjectName("configInner");
    auto *fields = new QVBoxLayout(inner);
    fields->setContentsMargins(0, 0, 2, 0);
    fields->setSpacing(9);
    fields->addWidget(section("输入选项", "01"));
    cameraIndex_ = new QSpinBox;
    cameraIndex_->setObjectName("cameraIndex");
    cameraIndex_->setRange(0, 10);
    cameraIndex_->setPrefix("摄像头编号  ");
    cameraIndex_->hide();
    fields->addWidget(cameraIndex_);
    lockedControls_.append(cameraIndex_);
    auto *sourceOptions = new QFormLayout;
    sourceOptions->setSpacing(7);
    sourceOptions->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inputColorMode_ = new QComboBox;
    inputColorMode_->setObjectName("inputColorMode");
    inputColorMode_->addItem("RGB · 彩色", "rgb");
    inputColorMode_->addItem("BGR · 彩色", "bgr");
    inputColorMode_->addItem("灰度 · 明度", "grayscale");
    inputColorMode_->setAccessibleName("输入颜色");
    inputColorMode_->setToolTip("选择模型实际接收的颜色。灰度会转换为明度：单通道模型直接使用，三通道模型复制"
                                "为三个相同通道。预览同步显示所选颜色模式。");
    sourceOptions->addRow("输入颜色", inputColorMode_);
    stereoView_ = new QComboBox;
    stereoView_->setObjectName("stereoView");
    stereoView_->addItem("完整画面", "full");
    stereoView_->addItem("双目左目", "left");
    stereoView_->addItem("双目右目", "right");
    stereoView_->setAccessibleName("双目画面选择");
    stereoView_->setToolTip("单设备左右并排（SBS）输入：沿水平中线裁出左目或右目，预览、推理和导出均使用所选"
                            "眼；完整画面不裁剪。");
    stereoLabel_ = text("画面选择");
    sourceOptions->addRow(stereoLabel_, stereoView_);
    stereoLabel_->hide();
    stereoView_->hide();
    stereoView_->setEnabled(false);
    fields->addLayout(sourceOptions);
    preprocessHint_ = text("彩色输入 · RGB / BGR 通道顺序", "tiny");
    preprocessHint_->setWordWrap(true);
    fields->addWidget(preprocessHint_);
    lockedControls_.append(inputColorMode_);
    lockedControls_.append(stereoView_);
    fields->addSpacing(7);
    fields->addWidget(section("检测模型", "02"));
    modelName_ = text("选择视觉模型", "modelName");
    modelName_->setObjectName("activeModelName");
    modelName_->setWordWrap(true);
    fields->addWidget(modelName_);
    modelMeta_ = text("支持 ONNX / PT 检测模型", "tiny");
    modelMeta_->setObjectName("modelMetadata");
    modelMeta_->setWordWrap(true);
    fields->addWidget(modelMeta_);
    modelButton_ = button("导入模型", "plus");
    fields->addWidget(modelButton_);
    connect(modelButton_, &QPushButton::clicked, this, &MainWindow::importModel);
    lockedControls_.append(modelButton_);
    taskBox_ = new QComboBox;
    taskBox_->setObjectName("modelTask");
    taskBox_->addItems({"YOLOv5 · 目标检测", "YOLOv8 / 11 · 目标检测", "图像分类"});
    taskBox_->setToolTip("按模型实际输出选择格式，分割、姿态与含 NMS 的模型暂不支持");
    fields->addWidget(taskBox_);
    lockedControls_.append(taskBox_);
    labelButton_ = button("类别标签 · 默认 COCO 80", "help");
    labelButton_->setToolTip("导入 UTF-8 文本，每行一个类别名。自定义模型应配置自己的标签。");
    fields->addWidget(labelButton_);
    auto *labelMenu = new QMenu(labelButton_);
    labelMenu->addAction("导入标签文件", this, &MainWindow::importLabels);
    labelMenu->addAction("使用模型默认类别", this,
                         [this]
                         {
                             labels_.clear();
                             labelsPath_.clear();
                             invalidateConfiguredResults();
                             updateTaskUi();
                             persist();
                             showNotice("已恢复模型默认类别。");
                         });
    labelButton_->setMenu(labelMenu);
    lockedControls_.append(labelButton_);
    auto *filterForm = new QFormLayout;
    filterForm->setContentsMargins(0, 0, 0, 0);
    classFilter_ = new QLineEdit;
    classFilter_->setObjectName("classFilter");
    classFilter_->setAccessibleName("类别筛选 ID");
    classFilter_->setPlaceholderText("全部类别，或 0,2,32");
    classFilter_->setMaxLength(4096);
    classFilter_->setToolTip("按模型原始 class_id 筛选，ID 从 0 开始；逗号、中文逗号或空格分隔。\n"
                             "类别标签只映射名称，不改变 ID。COCO 的 ID 32 是 sports ball，包含多种运动球，筛选它不等于只识别足球。");
    filterForm->addRow("类别筛选", classFilter_);
    fields->addLayout(filterForm);
    classFilterHint_ = text("留空检测全部类别", "tiny");
    classFilterHint_->setObjectName("classFilterHint");
    classFilterHint_->setWordWrap(true);
    classFilterHint_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fields->addWidget(classFilterHint_);
    lockedControls_.append(classFilter_);
    connect(classFilter_, &QLineEdit::textChanged, this, [this] {
        if (!initializing_ && !busy_ && !modelInspectionPending_ &&
            inspectedModelIdentity_ != modelFileIdentity()) inspectSelectedModel();
        updateClassFilterUi();
        if (!initializing_ && !busy_) invalidateConfiguredResults();
        persist();
    });
    fields->addSpacing(7);
    fields->addWidget(section("推理参数", "03"));
    auto *form = new QFormLayout;
    form->setSpacing(9);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inputSize_ = new WheelSafeSpinBox<QSpinBox>;
    inputSize_->setObjectName("inputSize");
    inputSize_->setRange(32, 4096);
    inputSize_->setSingleStep(32);
    inputSize_->setValue(640);
    inputSize_->setSuffix(" px");
    form->addRow("输入尺寸", inputSize_);
    confidence_ = new WheelSafeSpinBox<QDoubleSpinBox>;
    confidence_->setObjectName("confidence");
    confidence_->setRange(0.001, 1.0);
    confidence_->setDecimals(3);
    confidence_->setSingleStep(0.05);
    confidence_->setValue(0.25);
    form->addRow("置信度", confidence_);
    iou_ = new WheelSafeSpinBox<QDoubleSpinBox>;
    iou_->setObjectName("nmsIou");
    iou_->setRange(0.01, 1.0);
    iou_->setDecimals(2);
    iou_->setSingleStep(0.05);
    iou_->setValue(0.45);
    form->addRow("NMS IoU", iou_);
    fields->addLayout(form);
    for (auto *p : QList<QWidget *>{inputSize_, confidence_, iou_})
        lockedControls_.append(p);
    auto *advanced = button("预处理设置 ▾", {}, "ghost");
    fields->addWidget(advanced);
    auto *adv = new QWidget;
    auto *af = new QFormLayout(adv);
    af->setContentsMargins(0, 5, 0, 0);
    af->setSpacing(6);
    scale_ = new QDoubleSpinBox;
    scale_->setDecimals(8);
    scale_->setRange(0.00000001, 10);
    scale_->setValue(1.0 / 255.0);
    scale_->setSingleStep(0.001);
    af->addRow("缩放系数", scale_);
    auto mean = [&](QDoubleSpinBox *&m, const QString &label)
    {
        m = new QDoubleSpinBox;
        m->setRange(-1024, 1024);
        m->setDecimals(3);
        af->addRow(label, m);
        lockedControls_.append(m);
    };
    mean(meanR_, "均值 R");
    mean(meanG_, "均值 G");
    mean(meanB_, "均值 B");
    meanR_->setObjectName("meanR");
    meanG_->setObjectName("meanG");
    meanB_->setObjectName("meanB");
    meanRLabel_ = qobject_cast<QLabel *>(af->labelForField(meanR_));
    adv->hide();
    fields->addWidget(adv);
    connect(advanced, &QPushButton::clicked, this,
            [adv, advanced]
            {
                adv->setVisible(!adv->isVisible());
                advanced->setText(adv->isVisible() ? "预处理设置 ▴" : "预处理设置 ▾");
            });
    lockedControls_.append(scale_);
    autoExport_ = new QCheckBox("自动保存每张图片的结果");
    autoExport_->setToolTip("保存标注 PNG、JSON 和 CSV；视频与摄像头在停止后保存最后一帧");
    fields->addWidget(autoExport_);
    auto *openExportDirectory = button("打开保存文件夹", "folder", "ghost");
    openExportDirectory->setObjectName("openExportDirectoryButton");
    openExportDirectory->setToolTip(exportDir_);
    fields->addWidget(openExportDirectory);
    connect(openExportDirectory, &QPushButton::clicked, this,
            [this, openExportDirectory]
            {
                openExportDirectory->setToolTip(exportDir_);
                if (!QDir().mkpath(exportDir_))
                    showNotice("无法创建保存文件夹：" + exportDir_, true);
                else if (!QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(exportDir_).absoluteFilePath())))
                    showNotice("无法打开保存文件夹：" + exportDir_, true);
            });
    lockedControls_.append(autoExport_);
    backendBadge_ = text("设备待验证", "chip");
    backendBadge_->setObjectName("actualDeviceBadge");
    backendBadge_->setWordWrap(true);
    backendBadge_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fields->addWidget(backendBadge_);
    fields->addStretch();
    scroll->setWidget(inner);
    cl->addWidget(scroll, 1);
    detectionRangePanel_ = new QWidget;
    detectionRangePanel_->setObjectName("detectionRangePanel");
    auto *rangeLayout = new QVBoxLayout(detectionRangePanel_);
    rangeLayout->setContentsMargins(0, 0, 0, 0);
    rangeLayout->setSpacing(5);
    auto *rangeForm = new QFormLayout;
    rangeForm->setContentsMargins(0, 0, 0, 0);
    detectionRange_ = new QComboBox;
    detectionRange_->setObjectName("detectionRange");
    detectionRange_->setAccessibleName("检测范围");
    detectionRange_->addItems({"检测当前集合", "从当前图片向后检测", "只检测当前图片"});
    detectionRange_->setToolTip("点击开始时固定本次范围和队列顺序；检测当前集合始终从第一张开始。");
    rangeForm->addRow("检测范围", detectionRange_);
    rangeLayout->addLayout(rangeForm);
    rangeHint_ = text("请先导入 YOLO 数据集", "tiny");
    rangeHint_->setObjectName("detectionRangeHint");
    rangeHint_->setWordWrap(true);
    rangeLayout->addWidget(rangeHint_);
    cl->addWidget(detectionRangePanel_);
    lockedControls_.append(detectionRange_);
    connect(detectionRange_, &QComboBox::currentIndexChanged, this, [this] { updateRangeUi(); });
    runButton_ = button("开始检测", "play", "primary");
    runButton_->setObjectName("startDetectionButton");
    runButton_->setMinimumHeight(44);
    cl->addWidget(runButton_);
    connect(runButton_, &QPushButton::clicked, this, &MainWindow::startWorkbenchTask);
    stopButton_ = button("停止运行", "stop", "danger");
    stopButton_->hide();
    cl->addWidget(stopButton_);
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::stopInference);
    cl->addWidget(text("Ctrl + R 开始  ·  Esc 停止", "tiny"), 0, Qt::AlignCenter);
    columns->addWidget(config);
    auto *center = new QWidget;
    center->setMinimumWidth(390);
    auto *ml = new QVBoxLayout(center);
    ml->setContentsMargins(0, 0, 0, 0);
    ml->setSpacing(4);
    auto *canvasFrame = card();
    auto *cf = new QVBoxLayout(canvasFrame);
    cf->setContentsMargins(0, 0, 0, 0);
    cf->setSpacing(0);
    auto *toolbar = new QWidget;
    toolbar->setObjectName("imagePreviewToolbar");
    toolbar->setStyleSheet("QWidget#imagePreviewToolbar { background:#2C2C2C; border-bottom:1px solid #3B3B3B; border-top-left-radius:8px; border-top-right-radius:8px; }");
    auto *tl = new QHBoxLayout(toolbar);
    tl->setContentsMargins(16, 3, 14, 3);
    tl->setSpacing(8);
    canvasTitle_ = text("视觉预览", "sectionTitle");
    tl->addWidget(canvasTitle_, 1);
    showBoxes_ = new QCheckBox("检测框");
    showBoxes_->setChecked(true);
    tl->addWidget(showBoxes_);
    showLabels_ = new QCheckBox("标签");
    showLabels_->setChecked(true);
    tl->addWidget(showLabels_);
    showGroundTruth_ = new QCheckBox("GT 标注");
    showGroundTruth_->setObjectName("showGroundTruth");
    showGroundTruth_->setChecked(true);
    showGroundTruth_->hide();
    tl->addWidget(showGroundTruth_);
    auto *fit = button({}, "fit", "ghost");
    fit->setFixedSize(32, 32);
    fit->setMinimumHeight(32);
    fit->setToolTip("适应画布 · 双击图像");
    tl->addWidget(fit);
    auto *actual = button("1:1", {}, "ghost");
    actual->setFixedSize(38, 32);
    actual->setMinimumHeight(32);
    actual->setToolTip("原始像素大小");
    tl->addWidget(actual);
    cf->addWidget(toolbar);
    auto *recordingRow = new QHBoxLayout;
    recordingRow->setContentsMargins(14, 2, 12, 2);
    recordingStatus_ = text("视频 / 摄像头检测时可录制", "tiny");
    recordingStatus_->setObjectName("recordingStatus");
    recordingRow->addWidget(recordingStatus_, 1);
    recordButton_ = button("开始录制", "record");
    recordButton_->setObjectName("recordButton");
    recordButton_->setEnabled(false);
    recordButton_->setToolTip("先开始视频或摄像头检测，再点击录制。再次点击结束并保存到录制视频页。");
    recordingRow->addWidget(recordButton_);
    cf->addLayout(recordingRow);
    connect(recordButton_, &QPushButton::clicked, this, &MainWindow::toggleRecording);
    recordingClock_ = new QTimer(this);
    recordingClock_->setInterval(250);
    connect(recordingClock_, &QTimer::timeout, this, &MainWindow::updateRecordingUi);
    canvas_ = new ImageCanvas;
    canvas_->setObjectName("imageCanvas");
    canvas_->setMinimumHeight(280);
    cf->addWidget(canvas_, 1);
    auto *cb = new QHBoxLayout;
    cb->setContentsMargins(14, 3, 14, 3);
    cb->addWidget(text("滚轮缩放 · 拖动平移 · 点击目标查看", "tiny"), 1);
    zoomLabel_ = text("100%", "tiny");
    cb->addWidget(zoomLabel_);
    cf->addLayout(cb);
    ml->addWidget(canvasFrame, 1);
    connect(showGroundTruth_, &QCheckBox::toggled, canvas_, &ImageCanvas::setGroundTruthVisible);
    connect(fit, &QPushButton::clicked, canvas_, &ImageCanvas::fitToView);
    connect(actual, &QPushButton::clicked, canvas_, &ImageCanvas::actualSize);
    connect(showBoxes_, &QCheckBox::toggled, canvas_, &ImageCanvas::setBoxesVisible);
    connect(showLabels_, &QCheckBox::toggled, canvas_, &ImageCanvas::setLabelsVisible);
    connect(canvas_, &ImageCanvas::zoomChanged, this,
            [this](int z) { zoomLabel_->setText(QString::number(z) + "%"); });
    connect(canvas_, &ImageCanvas::filesDropped, this, [this](const QStringList &paths) { handlePreviewDrop(paths); });
    auto *queueFrame = card();
    auto *ql = new QHBoxLayout(queueFrame);
    ql->setContentsMargins(12, 6, 12, 6);
    ql->setSpacing(5);
    auto *qh = new QVBoxLayout;
    qh->addWidget(text("输入队列", "sectionTitle"), 1);
    auto *clear = button("清空", "cross", "ghost");
    clear->setMinimumHeight(25);
    clear->setMaximumHeight(27);
    qh->addWidget(clear);
    lockedControls_.append(clear);
    connect(clear, &QPushButton::clicked, this,
            [this]
            {
                cancelPendingDatasetSessionRestore();
                clearDatasetContext();
                clearImageResults();
                files_.clear();
                queue_->clear();
                sourceKind_ = vision::SourceKind::Images;
                lastResult_ = {};
                canvas_->clear();
                predictionTable_->setRowCount(0);
                emptyResults_->show();
                exportButton_->setEnabled(false);
                updateSourceUi();
            });
    ql->addLayout(qh);
    queue_ = new LazyImageQueue;
    queue_->setObjectName("inputQueue");
    queue_->setFlow(QListView::LeftToRight);
    queue_->setViewMode(QListView::IconMode);
    queue_->setIconSize(QSize(64, 36));
    queue_->setGridSize(QSize(100, 64));
    queue_->setWrapping(false);
    queue_->setFixedHeight(68);
    queue_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ql->addWidget(queue_, 1);
    ml->addWidget(queueFrame);
    connect(queue_, &QListWidget::currentRowChanged, this, &MainWindow::selectQueueImage);
    connect(queue_, &QListWidget::itemSelectionChanged, this, [this] { updateRangeUi(); });
    columns->addWidget(center);
    auto *inspector = card();
    inspector->setMinimumWidth(230);
    inspector->setMaximumWidth(300);
    auto *rl = new QVBoxLayout(inspector);
    rl->setContentsMargins(12, 14, 12, 14);
    rl->setSpacing(12);
    auto *ih = new QHBoxLayout;
    ih->addWidget(text("检测结果", "sectionTitle"), 1);
    ih->addWidget(text("LIVE", "chip"));
    rl->addLayout(ih);
    resultInfo_ = text("运行模型后，目标和置信度将显示在这里。", "muted");
    resultInfo_->setWordWrap(true);
    resultInfo_->setMinimumHeight(44);
    resultInfo_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    rl->addWidget(resultInfo_);
    predictionTable_ = new QTableWidget(0, 3);
    predictionTable_->setHorizontalHeaderLabels({"类别", "置信度", "类别 ID"});
    tableStyle(predictionTable_);
    predictionTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    predictionTable_->setColumnWidth(1, 78);
    predictionTable_->setColumnWidth(2, 58);
    rl->addWidget(predictionTable_, 1);
    emptyResults_ = text("等待检测\n\n目标详情将在运行后呈现。", "muted");
    emptyResults_->setAlignment(Qt::AlignCenter);
    emptyResults_->setWordWrap(true);
    rl->addWidget(emptyResults_);
    rl->addStretch(0);
    auto *hints = text("点击列表中的目标，可在画布中定位。\n检测框坐标以原始图像像素为准。", "tiny");
    hints->setWordWrap(true);
    rl->addWidget(hints);
    columns->addWidget(inspector);
    connect(predictionTable_, &QTableWidget::itemSelectionChanged, this,
            [this]
            {
                canvas_->setSelectedPrediction(
                    predictionTable_->selectedItems().isEmpty() ? -1 : predictionTable_->currentRow());
            });
    connect(canvas_, &ImageCanvas::predictionSelected, this,
            [this](int row)
            {
                if (row >= 0 && row < predictionTable_->rowCount())
                    predictionTable_->selectRow(row);
                else
                    predictionTable_->clearSelection();
            });
    connect(taskBox_, &QComboBox::currentIndexChanged, this, [this] {
        updateTaskUi();
        if (!initializing_ && !busy_) {
            invalidateConfiguredResults();
            inspectSelectedModel();
        }
    });
    const auto parametersChanged = [this] {
        if (!initializing_ && !busy_) {
            invalidateConfiguredResults();
            persist();
        }
    };
    connect(inputSize_, &QSpinBox::valueChanged, this, parametersChanged);
    for (auto *control : {confidence_, iou_, scale_, meanR_, meanG_, meanB_})
        connect(control, &QDoubleSpinBox::valueChanged, this, parametersChanged);
    connect(inputColorMode_, &QComboBox::currentIndexChanged, this,
            [this]
            {
                updateTaskUi();
                updateInputPreview();
                persist();
            });
    connect(stereoView_, &QComboBox::currentIndexChanged, this,
            [this]
            {
                if (!busy_ && sourceKind_ != vision::SourceKind::Images)
                {
                    lastResult_ = {};
                    canvas_->clear();
                    predictionTable_->setRowCount(0);
                    emptyResults_->show();
                    exportButton_->setEnabled(false);
                }
                updateSourceUi();
                persist();
            });
    connect(cameraIndex_, &QSpinBox::valueChanged, this,
            [this]
            {
                updateSourceUi();
                persist();
            });
    columns->setStretchFactor(0, 0);
    columns->setStretchFactor(1, 1);
    columns->setStretchFactor(2, 0);
    columns->setSizes({280, 660, 270});
    auto *previewPage = new QWidget;
    previewPage->setObjectName("imagePreviewPage");
    auto *previewLayout = new QVBoxLayout(previewPage);
    previewLayout->setContentsMargins(0, 0, 0, 0);
    previewLayout->setSpacing(6);
    evaluationReviewBar_ = card("evaluationReviewBar");
    auto *reviewLayout = new QHBoxLayout(evaluationReviewBar_);
    reviewLayout->setContentsMargins(10, 5, 10, 5);
    reviewLayout->setSpacing(6);
    evaluationReviewSummaryLabel_ = text({}, "muted");
    evaluationReviewSummaryLabel_->setObjectName("evaluationReviewSummaryLabel");
    evaluationReviewSummaryLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    reviewLayout->addWidget(evaluationReviewSummaryLabel_, 1);
    evaluationPreviewFilter_ = new QComboBox;
    evaluationPreviewFilter_->setObjectName("evaluationPreviewFilterCombo");
    for (const auto &entry : QList<QPair<QString, QString>>{{"全部图片", "all"}, {"包含误检", "fp"},
             {"包含漏检", "fn"}, {"无目标误报", "negative_fp"}, {"全部错例", "errors"}})
        evaluationPreviewFilter_->addItem(entry.first, entry.second);
    evaluationPreviewFilter_->setMaximumWidth(120);
    reviewLayout->addWidget(evaluationPreviewFilter_);
    connect(evaluationPreviewFilter_, &QComboBox::currentIndexChanged, this, [this] {
        applyEvaluationFilter(evaluationPreviewFilter_->currentData().toString());
    });
    evaluationReviewPreviousButton_ = button("上一张");
    evaluationReviewPreviousButton_->setObjectName("evaluationReviewPreviousButton");
    evaluationReviewNextButton_ = button("下一张");
    evaluationReviewNextButton_->setObjectName("evaluationReviewNextButton");
    reviewLayout->addWidget(evaluationReviewPreviousButton_);
    reviewLayout->addWidget(evaluationReviewNextButton_);
    connect(evaluationReviewPreviousButton_, &QPushButton::clicked, this, [this] { navigateEvaluationReview(-1); });
    connect(evaluationReviewNextButton_, &QPushButton::clicked, this, [this] { navigateEvaluationReview(1); });
    auto *allReview = button("全部图片", "image");
    allReview->setObjectName("evaluationReviewAllButton");
    reviewLayout->addWidget(allReview);
    connect(allReview, &QPushButton::clicked, this, [this] { applyEvaluationFilter("all"); });
    auto *analysisReview = button("返回评估", "graph");
    analysisReview->setObjectName("evaluationReviewAnalysisButton");
    reviewLayout->addWidget(analysisReview);
    connect(analysisReview, &QPushButton::clicked, this, [this] { workbenchTabs_->setCurrentIndex(1); });
    for (auto *control : {evaluationReviewPreviousButton_, evaluationReviewNextButton_, allReview, analysisReview}) {
        control->setMinimumHeight(30);
        control->setIconSize(QSize(14, 14));
    }
    evaluationReviewSummaryLabel_->setStyleSheet("font-size:11px; color:#D2D2D2;");
    evaluationPreviewFilter_->setStyleSheet("QComboBox { min-height:20px; padding:4px 6px; }");
    evaluationReviewBar_->hide();
    previewLayout->addWidget(evaluationReviewBar_);
    previewLayout->addWidget(columns);
    previewPage->setAcceptDrops(true);
    for (auto *child : previewPage->findChildren<QWidget *>()) child->setAcceptDrops(true);
    auto *evaluationPage = new QWidget;
    evaluationPage->setObjectName("evaluationWorkspacePage");
    auto *evaluationLayout = new QVBoxLayout(evaluationPage);
    evaluationLayout->setContentsMargins(0, 0, 0, 0);
    evaluationLayout->setSpacing(8);
    auto *taskCard = card("evaluationTaskBar");
    auto *taskLayout = new QVBoxLayout(taskCard);
    taskLayout->setContentsMargins(10, 6, 10, 6);
    taskLayout->setSpacing(3);
    auto *taskActions = new QHBoxLayout;
    evaluationTaskSummary_ = text("导入带标注的 YOLO 数据集，开始模型评估", "muted");
    evaluationTaskSummary_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    taskActions->addWidget(evaluationTaskSummary_, 1);
    evaluationRange_ = new QComboBox;
    evaluationRange_->setObjectName("evaluationRangeCombo");
    for (int i = 0; i < detectionRange_->count(); ++i) evaluationRange_->addItem(detectionRange_->itemText(i));
    taskActions->addWidget(evaluationRange_);
    lockedControls_.append(evaluationRange_);
    connect(evaluationRange_, &QComboBox::currentIndexChanged, detectionRange_, &QComboBox::setCurrentIndex);
    connect(detectionRange_, &QComboBox::currentIndexChanged, evaluationRange_, &QComboBox::setCurrentIndex);
    auto *configure = button("推理配置", "settings");
    configure->setObjectName("evaluationConfigureButton");
    taskActions->addWidget(configure);
    connect(configure, &QPushButton::clicked, this, [this] { workbenchTabs_->setCurrentIndex(0); });
    evaluationReviewButton_ = button("查看错例", "eye");
    evaluationReviewButton_->setObjectName("evaluationReviewButton");
    evaluationReviewButton_->setEnabled(false);
    taskActions->addWidget(evaluationReviewButton_);
    connect(evaluationReviewButton_, &QPushButton::clicked, this, &MainWindow::showEvaluationErrors);
    evaluationStartButton_ = button("开始评估", "play", "primary");
    evaluationStartButton_->setObjectName("evaluationStartButton");
    taskActions->addWidget(evaluationStartButton_);
    connect(evaluationStartButton_, &QPushButton::clicked, this, &MainWindow::startWorkbenchTask);
    evaluationStopButton_ = button("停止评估", "stop", "danger");
    evaluationStopButton_->setObjectName("evaluationStopButton");
    evaluationStopButton_->hide();
    taskActions->addWidget(evaluationStopButton_);
    connect(evaluationStopButton_, &QPushButton::clicked, this, &MainWindow::stopInference);
    taskLayout->addLayout(taskActions);
    evaluationRangeHint_ = text("请先导入 YOLO 数据集；评估需要有效标注与类别映射。", "tiny");
    evaluationRangeHint_->setObjectName("evaluationRangeHint");
    evaluationRangeHint_->setWordWrap(true);
    taskLayout->addWidget(evaluationRangeHint_);
    evaluationLayout->addWidget(taskCard);
    evaluationLayout->addWidget(evaluation_, 1);
    workbenchTabs_ = new QTabWidget;
    workbenchTabs_->setObjectName("workbenchAnalysisTabs");
    workbenchTabs_->addTab(previewPage, "图像预览");
    workbenchTabs_->addTab(evaluationPage, "评估与阈值");
    all->addWidget(workbenchTabs_, 1);
    connect(workbenchTabs_, &QTabWidget::currentChanged, this, [this](int) { updateSourceUi(); });
    return page;
}

QWidget *MainWindow::buildModels()
{
    auto *page = new QWidget;
    page->setObjectName("modelLibraryPage");
    auto *l = new QVBoxLayout(page);
    l->setContentsMargins(0, 0, 0, 0);
    l->setSpacing(16);
    auto *row = new QHBoxLayout;
    modelCount_ = text("本地模型库", "sectionTitle");
    row->addWidget(modelCount_, 1);
    auto *add = button("导入 ONNX / PT 模型", "plus", "primary");
    row->addWidget(add);
    connect(add, &QPushButton::clicked, this, &MainWindow::importModel);
    lockedControls_.append(add);
    l->addLayout(row);
    auto *frame = card();
    auto *fl = new QVBoxLayout(frame);
    fl->setContentsMargins(20, 18, 20, 18);
    fl->setSpacing(12);
    fl->addWidget(text("集中管理你的模型", "modelName"));
    fl->addWidget(text("点击模型即全局选用：检测工作台与模型显示同步，模型结构会在后台预加载。", "muted"));
    modelLibraryDropHint_ = text(modelLibraryDropPrompt, "muted");
    modelLibraryDropHint_->setObjectName("modelLibraryDropHint");
    modelLibraryDropHint_->setWordWrap(true);
    fl->addWidget(modelLibraryDropHint_);
    modelList_ = new ModelLibraryList;
    modelList_->setObjectName("globalModelList");
    modelList_->setIconSize(QSize(34, 34));
    modelList_->setDragEnabled(true);
    modelList_->setAcceptDrops(true);
    modelList_->setDropIndicatorShown(true);
    modelList_->setDragDropMode(QAbstractItemView::InternalMove);
    modelList_->setDefaultDropAction(Qt::MoveAction);
    modelList_->setDragDropOverwriteMode(false);
    modelList_->setSelectionMode(QAbstractItemView::SingleSelection);
    modelList_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    modelList_->setToolTip("拖入本地模型文件以添加；库内拖动调整顺序；点击全局选用，右键管理模型。");
    modelList_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(modelList_, &QListWidget::customContextMenuRequested, this,
            &MainWindow::showModelContextMenu);
    connect(modelList_->model(), &QAbstractItemModel::rowsAboutToBeMoved, this,
            [this] { modelListReordering_ = true; });
    connect(modelList_->model(), &QAbstractItemModel::rowsMoved, this,
            [this]
            {
                models_.clear();
                const QSignalBlocker blocker(modelList_);
                for (int row = 0; row < modelList_->count(); ++row)
                {
                    auto *item = modelList_->item(row);
                    const QString path = item->data(Qt::UserRole).toString();
                    models_.append(path);
                    if (path == modelPath_)
                        modelList_->setCurrentItem(item);
                }
                modelListReordering_ = false;
                persist();
            });
    fl->addWidget(modelList_, 1);
    auto *actions = new QHBoxLayout;
    auto *activate = button("前往检测工作台", "arrow", "primary");
    auto *structure = button("查看模型结构", "graph");
    structure->setObjectName("showModelStructureButton");
    auto *remove = button("从列表移除", "cross");
    lockedControls_.append(activate);
    lockedControls_.append(remove);
    actions->addWidget(activate);
    actions->addWidget(structure);
    actions->addWidget(remove);
    auto *evaluationReport = button("查看评估 / 检测配置", "history");
    evaluationReport->setObjectName("modelEvaluationReportButton");
    actions->addWidget(evaluationReport);
    connect(evaluationReport, &QPushButton::clicked, this, [this] {
        const auto profile = modelEvaluationProfiles_.value(modelPath_).toMap();
        if (profile.value("report_directory").toString().isEmpty()) { selectRoute(ui::Route::Workbench); evaluation_->chooseDataset(); }
        else loadEvaluationReport(profile.value("report_directory").toString());
    });
    actions->addStretch();
    fl->addLayout(actions);
    connect(structure, &QPushButton::clicked, this, [this] { selectRoute(ui::Route::ModelDisplay); });
    connect(activate, &QPushButton::clicked, this, [this] { selectRoute(ui::Route::Workbench); });
    connect(modelList_, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem *item)
            {
                if (item && !busy_ && !modelListReordering_)
                    setModel(item->data(Qt::UserRole).toString());
            });
    connect(modelList_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *) { selectRoute(ui::Route::Workbench); });
    connect(remove, &QPushButton::clicked, this,
            [this]
            {
                auto *i = modelList_->currentItem();
                if (i)
                    removeLibraryModel(i->data(Qt::UserRole).toString());
            });
    l->addWidget(frame, 1);
    auto *notes = card();
    auto *nl = new QVBoxLayout(notes);
    nl->setContentsMargins(18, 14, 18, 14);
    nl->addWidget(text("推荐模型导出配置", "sectionTitle"));
    auto *info = text(
        "PT · 本机 PyTorch 直接推理，自动读取模型内置类别。\nONNX · batch=1、固定正方形输入、FP32、不包含 "
        "NMS。YOLOv5 使用 [1, N, 5+C]，YOLOv8 / 11 使用 [1, 4+C, N]。",
        "muted");
    info->setWordWrap(true);
    nl->addWidget(info);
    l->addWidget(notes);
    page->setAcceptDrops(true);
    for (auto *child : page->findChildren<QWidget *>()) child->setAcceptDrops(true);
    return page;
}
QWidget *MainWindow::buildModelDisplay()
{
    auto *page = new QWidget;
    page->setObjectName("modelDisplayPage");
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(14);
    auto *controls = card();
    auto *controlLayout = new QVBoxLayout(controls);
    controlLayout->setContentsMargins(18, 16, 18, 16);
    controlLayout->setSpacing(10);
    auto *heading = new QHBoxLayout;
    structureModelName_ = text("在模型库中选择模型", "structureModelName");
    structureModelName_->setStyleSheet("font-size:16px; font-weight:600; color:#F5F5F5;");
    heading->addWidget(structureModelName_, 1);
    structureStatus_ = text("等待模型", "chip");
    structureStatus_->setObjectName("structureLoadStatus");
    heading->addWidget(structureStatus_);
    heading->addWidget(text("Netron 9.3.1 · 本地查看", "chip"));
    controlLayout->addLayout(heading);
    structureModelMeta_ = text("支持 ONNX、PyTorch、TorchScript 等模型格式。", "structureModelMeta");
    structureModelMeta_->setStyleSheet("font-size:12px; color:#B8B8B8;");
    controlLayout->addWidget(structureModelMeta_);
    structureHint_ = text("想看完整结构，建使用导出的 ONNX。", "structureOnnxHint");
    structureHint_->setStyleSheet("font-size:11px; color:#A5A5A5;");
    structureHint_->hide();
    controlLayout->addWidget(structureHint_);
    layout->addWidget(controls);
    modelViewer_ = new ModelViewer;
    modelViewer_->setObjectName("modelStructureViewer");
    layout->addWidget(modelViewer_, 1);
    connect(modelViewer_, &ModelViewer::stateChanged, this,
            [this](ModelViewer::State state)
            {
                structureStatus_->setText(state == ModelViewer::State::Loading ? "后台加载中…"
                                          : state == ModelViewer::State::Ready ? QString("可用视图已缓存")
                                          : state == ModelViewer::State::Error ? "无法显示"
                                                                               : "等待模型");
            });
    connect(modelViewer_, &ModelViewer::modelLoaded, this,
            [this](const QString &path)
            {
                if (!busy_ && (pages_->currentIndex() == ui::routeIndex(ui::Route::Models) || pages_->currentIndex() == ui::routeIndex(ui::Route::ModelDisplay)))
                    showNotice(
                        QString("模型结构、层级树与参数表已缓存 · %1").arg(QFileInfo(path).fileName()));
            });
    connect(modelViewer_, &ModelViewer::loadFailed, this,
            [this](const QString &error)
            {
                if (pages_->currentIndex() == ui::routeIndex(ui::Route::ModelDisplay))
                    showNotice(error, true);
            });
    auto *note = text("使用模型库中全局选中的模型，各视图共用后台解析缓存。结构图可缩放与平移；层级树和参数"
                      "表支持搜索与详情查看。包含受支持架构信息的 PT 模型还可查看 YOLO YAML。",
                      "tiny");
    note->setWordWrap(true);
    layout->addWidget(note);
    return page;
}

void MainWindow::displayModelStructure()
{
    const QString &path = modelPath_;
    structureModelName_->setText(path.isEmpty() ? "在模型库中选择模型" : modelDisplayName(path));
    structureModelName_->setToolTip(path);
    structureModelMeta_->setText(path.isEmpty() ? "检测工作台与模型显示使用同一个全局模型。"
                                                : QFileInfo(path).suffix().toUpper() + " · " + path);
    structureModelMeta_->setToolTip(path);
    structureModelMeta_->setWordWrap(true);
    structureHint_->setVisible(!path.isEmpty() &&
                               QFileInfo(path).suffix().compare("onnx", Qt::CaseInsensitive) != 0);
    if (!path.isEmpty() && vision::cpuRuntimeDownloadNeeded()) {
        modelViewer_->stop();
        structureModelMeta_->setText("首次查看模型结构需要准备运行环境，可打开模型显示页或在更多页点击“准备 CPU 运行环境”。");
        if (!initializing_ && pages_->currentIndex() == ui::routeIndex(ui::Route::ModelDisplay))
            ui::ensureCpuRuntime(this, [this] { displayModelStructure(); });
        return;
    }
    if (modelViewer_->modelPath() != path || !modelViewer_->matchesModelFile(path) ||
        modelViewer_->state() == ModelViewer::State::Empty)
        modelViewer_->openModel(path);
}

QWidget *MainWindow::buildHistory()
{
    auto *page = new QWidget;
    auto *l = new QVBoxLayout(page);
    l->setContentsMargins(0, 0, 0, 0);
    l->setSpacing(16);
    auto *row = new QHBoxLayout;
    row->addWidget(text("最近 200 次检测与评估记录 · 双击评估记录回看报告", "sectionTitle"), 1);
    auto *exportHistory = button("导出运行记录", "export");
    row->addWidget(exportHistory);
    connect(exportHistory, &QPushButton::clicked, this,
            [this]
            {
                QString p = QFileDialog::getSaveFileName(this, "导出运行记录",
                                                         dataRoot_ + "/history-export.json", "JSON (*.json)");
                if (p.isEmpty())
                    return;
                QString err;
                if (atomicWrite(p, QJsonDocument(history_).toJson(), &err))
                    showNotice("运行记录已导出：" + p);
                else
                    showNotice("导出失败：" + err, true);
            });
    l->addLayout(row);
    auto *f = card();
    auto *fl = new QVBoxLayout(f);
    fl->setContentsMargins(12, 12, 12, 12);
    historyTable_ = new QTableWidget(0, 6);
    historyTable_->setHorizontalHeaderLabels({"时间", "输入", "模型", "目标数", "推理耗时", "任务"});
    tableStyle(historyTable_);
    historyTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    historyTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    historyTable_->setColumnWidth(0, 165);
    historyTable_->setColumnWidth(3, 85);
    historyTable_->setColumnWidth(4, 100);
    historyTable_->setColumnWidth(5, 120);
    fl->addWidget(historyTable_);
    l->addWidget(f, 1);
    auto *historyHint = text("运行记录已自动保存。导出图像与原始预测数据可在工作台完成。", "muted");
    historyHint->setToolTip(dataRoot_ + "/history.json");
    l->addWidget(historyHint);
    return page;
}
QWidget *MainWindow::buildMore()
{
    auto *page = new QWidget;
    page->setObjectName("morePage");
    auto *outer = new QVBoxLayout(page);
    outer->setContentsMargins(0, 0, 0, 0);
    auto *scroll = new QScrollArea;
    scroll->setObjectName("gpuMoreScroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->viewport()->setObjectName("moreViewport");
    scroll->viewport()->setAutoFillBackground(false);
    auto *content = new QWidget;
    content->setObjectName("moreContent");
    auto *layout = new QVBoxLayout(content);
    scroll->setWidget(content);
    outer->addWidget(scroll);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(16);
    auto *startup = card();
    auto *startupLayout = new QVBoxLayout(startup);
    startupLayout->setContentsMargins(24, 18, 24, 18);
    startupLayout->setSpacing(10);
    startupLayout->addWidget(text("启动与数据集", "modelName"));
    restoreDatasetOnStartup_ = new QCheckBox("启动时恢复上次数据集");
    restoreDatasetOnStartup_->setObjectName("restoreDatasetOnStartup");
    restoreDatasetOnStartup_->setChecked(settings_->value("restoreDatasetOnStartup", true).toBool());
    startupLayout->addWidget(restoreDatasetOnStartup_);
    auto *startupHint = text("默认开启。下次打开 Vision Studio 时，在后台载入上次成功导入的数据集与集合，不自动开始检测或评估。"
                             "关闭后仍保留路径记录，当前已载入的数据集不受影响；重新开启在下次启动生效。", "muted");
    startupHint->setObjectName("restoreDatasetOnStartupHint");
    startupHint->setWordWrap(true);
    startupHint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    startupLayout->addWidget(startupHint);
    connect(restoreDatasetOnStartup_, &QCheckBox::toggled, this, [this](bool enabled) {
        settings_->setValue("restoreDatasetOnStartup", enabled);
        settings_->sync();
        // Suppress an entry already queued by ordinary startup. A preference
        // change must never cancel an accepted manual dataset import.
        if (!enabled) ++datasetSessionRestoreGeneration_;
    });
    layout->addWidget(startup);
    auto *environment = card();
    auto *gpuLayout = new QVBoxLayout(environment);
    gpuLayout->setContentsMargins(24, 18, 24, 18);
    gpuLayout->setSpacing(10);
    gpuLayout->addWidget(text("NVIDIA GPU 推理环境", "modelName"));
    auto *gpuHint = text("CPU 可直接使用。首次准备 GPU 支持需要联网下载约 4 GB，建议至少预留 20 GB 空间；"
                         "依赖安装在个人目录，保留 CPU 环境，不修改显卡驱动。",
                         "muted");
    gpuHint->setWordWrap(true);
    gpuHint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    gpuLayout->addWidget(gpuHint);
    gpuEnvironmentStatus_ = text("正在检查 GPU 环境…", "body");
    gpuEnvironmentStatus_->setObjectName("gpuEnvironmentStatus");
    gpuEnvironmentStatus_->setWordWrap(true);
    gpuEnvironmentStatus_->setTextFormat(Qt::PlainText);
    gpuEnvironmentStatus_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    gpuEnvironmentStatus_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    gpuLayout->addWidget(gpuEnvironmentStatus_);
    auto *gpuActions = new QHBoxLayout;
    gpuCheckButton_ = button("检查 GPU 环境", "history");
    gpuCheckButton_->setObjectName("gpuCheckButton");
    gpuPrepareButton_ = button("准备 GPU 支持", "plus", "primary");
    gpuPrepareButton_->setObjectName("gpuPrepareButton");
    gpuCancelButton_ = button("取消准备", "stop", "danger");
    gpuCancelButton_->setObjectName("gpuCancelButton");
    gpuCancelButton_->hide();
    gpuActions->addWidget(gpuCheckButton_);
    gpuActions->addWidget(gpuPrepareButton_);
    gpuActions->addWidget(gpuCancelButton_);
    gpuActions->addStretch();
    gpuLayout->addLayout(gpuActions);
    gpuSetupProgress_ = new QProgressBar;
    gpuSetupProgress_->setObjectName("gpuSetupProgress");
    gpuSetupProgress_->setRange(0, 0);
    gpuSetupProgress_->setFixedHeight(7);
    gpuSetupProgress_->setTextVisible(false);
    gpuSetupProgress_->hide();
    gpuLayout->addWidget(gpuSetupProgress_);
    gpuLog_ = new QPlainTextEdit;
    gpuLog_->setObjectName("gpuSetupLog");
    gpuLog_->setReadOnly(true);
    gpuLog_->setMaximumBlockCount(600);
    gpuLog_->setMaximumHeight(135);
    gpuLog_->setMinimumHeight(90);
    gpuLog_->setStyleSheet("QPlainTextEdit { background:#202020; border:1px solid #454545; "
                           "border-radius:4px; color:#D2D2D2; padding:8px; font-size:12px; }");
    gpuLog_->hide();
    gpuLayout->addWidget(gpuLog_);
    connect(gpuCheckButton_, &QPushButton::clicked, this, [this] {
        if (ui::ensureCpuRuntime(this, [this] { checkGpuEnvironment(); })) checkGpuEnvironment();
    });
#ifdef Q_OS_WIN
    auto *cpuPrepare = button("准备 CPU 运行环境", "plus");
    cpuPrepare->setObjectName("cpuPrepareButton");
    cpuPrepare->setVisible(vision::cpuRuntimeDownloadNeeded());
    gpuLayout->insertWidget(0, cpuPrepare);
    connect(cpuPrepare, &QPushButton::clicked, this, [this] {
        const auto ready = [this] { inspectSelectedModel(); displayModelStructure(); checkGpuEnvironment(); restoreDatasetSession();
            showNotice("CPU 运行环境已就绪，可离线使用 PT、转换、模型视图及评估。"); };
        if (ui::ensureCpuRuntime(this, ready)) ready();
    });
#endif
    connect(gpuPrepareButton_, &QPushButton::clicked, this, &MainWindow::prepareGpuEnvironment);
    connect(gpuCancelButton_, &QPushButton::clicked, this, &MainWindow::cancelGpuPreparation);
    layout->addWidget(environment);
    auto *exports = card();
    auto *exportLayout = new QVBoxLayout(exports);
    exportLayout->setContentsMargins(24, 22, 24, 22);
    exportLayout->setSpacing(12);
    exportLayout->addWidget(text("导出检测结果", "modelName"));
    auto *exportHint =
        text("将当前检测结果保存为标注 PNG、JSON 和 CSV。请先在工作台完成检测；也可使用 Ctrl + E。", "muted");
    exportHint->setWordWrap(true);
    exportLayout->addWidget(exportHint);
    exportLayout->addWidget(exportButton_, 0, Qt::AlignLeft);
    layout->addWidget(exports);
    layout->addStretch();
    return page;
}

QWidget *MainWindow::buildRecordings()
{
    auto *page = new QWidget;
    page->setObjectName("recordingsPage");
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(14);
    auto *actions = new QHBoxLayout;
    recordingsCount_ = text("本地录像 · 0 个文件", "sectionTitle");
    actions->addWidget(recordingsCount_, 1);
    auto *refresh = button("刷新", "history");
    refresh->setObjectName("refreshRecordingsButton");
    connect(refresh, &QPushButton::clicked, this, &MainWindow::refreshRecordings);
    actions->addWidget(refresh);
    auto *folder = button("打开文件夹", "folder");
    folder->setObjectName("openRecordingsFolderButton");
    connect(folder, &QPushButton::clicked, this,
            [this]
            {
                const QString directory = dataRoot_ + "/recordings";
                if (!QDir().mkpath(directory) || !QDesktopServices::openUrl(QUrl::fromLocalFile(directory)))
                    showNotice("无法打开录像文件夹：" + directory, true);
            });
    actions->addWidget(folder);
    layout->addLayout(actions);
    auto *split = new QSplitter(Qt::Horizontal);
    auto *list = card();
    auto *listLayout = new QVBoxLayout(list);
    listLayout->setContentsMargins(12, 12, 12, 12);
    recordingsTable_ = new QTableWidget(0, 4);
    recordingsTable_->setObjectName("recordingsTable");
    recordingsTable_->setHorizontalHeaderLabels({"录像 / 时间", "时长", "分辨率", "大小"});
    tableStyle(recordingsTable_);
    recordingsTable_->setWordWrap(false);
    recordingsTable_->verticalHeader()->setDefaultSectionSize(58);
    recordingsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    recordingsTable_->setColumnWidth(1, 72);
    recordingsTable_->setColumnWidth(2, 96);
    recordingsTable_->setColumnWidth(3, 72);
    listLayout->addWidget(recordingsTable_);
    connect(recordingsTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::selectRecording);
    split->addWidget(list);
    auto *viewer = card();
    auto *viewerLayout = new QVBoxLayout(viewer);
    viewerLayout->setContentsMargins(12, 12, 12, 12);
    viewerLayout->setSpacing(12);
    viewerLayout->addWidget(text("录像预览", "sectionTitle"));
    recordingCanvas_ = new ImageCanvas;
    recordingCanvas_->setObjectName("recordingCanvas");
    recordingCanvas_->setAcceptDrops(false);
    recordingCanvas_->setBoxesVisible(false);
    recordingCanvas_->setLabelsVisible(false);
    viewerLayout->addWidget(recordingCanvas_, 1);
    recordingDetails_ = text("在工作台开始视频或摄像头检测，再点击「开始录制」。", "muted");
    recordingDetails_->setWordWrap(true);
    viewerLayout->addWidget(recordingDetails_);
    auto *playActions = new QHBoxLayout;
    recordingPlayButton_ = button("播放录像", "play");
    recordingPlayButton_->setObjectName("recordingPlayButton");
    recordingPlayButton_->setEnabled(false);
    connect(recordingPlayButton_, &QPushButton::clicked, this, &MainWindow::toggleRecordingPlayback);
    playActions->addWidget(recordingPlayButton_);
    exportRecordingButton_ = button("导出录像", "export");
    exportRecordingButton_->setObjectName("exportRecordingButton");
    exportRecordingButton_->setEnabled(false);
    connect(exportRecordingButton_, &QPushButton::clicked, this, &MainWindow::exportRecording);
    playActions->addWidget(exportRecordingButton_);
    playActions->addStretch();
    viewerLayout->addLayout(playActions);
    split->addWidget(viewer);
    split->setStretchFactor(0, 1);
    split->setStretchFactor(1, 1);
    layout->addWidget(split, 1);
    auto *hint = text("录像保存为 MJPEG AVI，包含所选画面和检测标注。停止录制或结束检测后自动保存。", "tiny");
    hint->setWordWrap(true);
    hint->setToolTip(dataRoot_ + "/recordings");
    layout->addWidget(hint);
    playbackTimer_ = new QTimer(this);
    connect(playbackTimer_, &QTimer::timeout, this, &MainWindow::readRecordingFrame);
    return page;
}

void MainWindow::clearImageResults()
{
    ++imagePreviewGeneration_;
    if (imagePreviewLoader_) imagePreviewLoader_->cancel();
    ++imageResultGeneration_;
    imageResults_.clear();
    imageResultDirectory_.reset();
    imageResultCacheWarning_ = false;
    imageConfigFingerprint_.clear();
    if (queue_)
        for (int row = 0; row < queue_->count(); ++row)
            queue_->item(row)->setForeground(QBrush());
}

void MainWindow::cacheImageResult(const vision::InferenceResult &result)
{
    if (result.image.isNull() || result.demonstration || !files_.contains(result.source))
        return;
    // The UI immediately keeps the accepted result. The bounded background writer
    // replaces its pixels with a lossless snapshot only after a successful commit.
    // The file may be replaced externally while inference is running. Never
    // relabel an already loaded model or image with the replacement's identity.
    imageConfigFingerprint_ = activeImageFingerprint_;
    imageResults_.insert(result.source, CachedImageResult{
        result, lastConfig_, {}, taskSourceIdentities_.value(result.source)});
}

void MainWindow::saveImageResultInBackground(const vision::InferenceResult &result, quint64 ticket)
{
    if (!imageResultDirectory_)
        imageResultDirectory_ = std::make_shared<QTemporaryDir>(QDir::tempPath() + "/vision-studio-image-results-XXXXXX");
    vision::app::ResultWriteRequest request;
    request.result = result; request.ticket = ticket; request.generation = imageResultGeneration_;
    request.directory = imageResultDirectory_;
    if (request.directory->isValid())
        request.snapshotPath = request.directory->filePath(QUuid::createUuid().toString(QUuid::Id128) + ".png");
    request.automaticExport = autoExport_->isChecked();
    if (request.automaticExport) request.exported = prepareResultExport(result, exportDir_, lastConfig_);
    request.historyPath = dataRoot_ + "/history.json";
    request.history = vision::app::HistoryRepository::serialize(history_);
    if (!resultWriter_->enqueue(std::move(request))) detection_->acknowledgeImageResult(ticket);
}

void MainWindow::selectQueueImage(int row)
{
    if (busy_ || queuePopulating_ || sourceKind_ != vision::SourceKind::Images || row < 0 || row >= files_.size())
        return;
    updateRangeUi();
    if (evaluationReviewActive_ && datasetSamples_.contains(files_.at(row))) {
        ++imagePreviewGeneration_;
        if (imagePreviewLoader_) imagePreviewLoader_->cancel();
        reviewEvaluationImage(row); return;
    }
    if (!imageResults_.isEmpty() && imageConfigFingerprint_ != inferenceFingerprint()) clearImageResults();
    const QString source = files_.at(row);
    auto cached = imageResults_.constFind(source);
    const QString currentSourceIdentity = fileIdentity(source);
    if (cached != imageResults_.cend() && !currentSourceIdentity.isEmpty() && currentSourceIdentity != cached->sourceIdentity) {
        imageResults_.remove(source); queue_->item(row)->setForeground(QBrush()); cached = imageResults_.cend();
    }
    if (cached != imageResults_.cend() && !cached->result.image.isNull()) {
        ++imagePreviewGeneration_; if (imagePreviewLoader_) imagePreviewLoader_->cancel();
        lastConfig_ = cached->config; displayResult(cached->result); return;
    }
    if (cached != imageResults_.cend() && lastResult_.source == source && !lastResult_.image.isNull() &&
        canvas_->result().source == source && !canvas_->result().image.isNull()) return;
    if (!imagePreviewLoader_) {
        imagePreviewLoader_ = new AsyncImagePreview(this);
        connect(imagePreviewLoader_, &AsyncImagePreview::ready, this,
            [this](quint64 generation, const QString &, const QImage &original, const QImage &image, bool primaryUsed, const QString &) {
            if (generation != imagePreviewGeneration_ || busy_ || closing_ ||
                (evaluationReviewActive_ && datasetSamples_.contains(files_.value(queue_->currentRow()))) ||
                sourceKind_ != vision::SourceKind::Images || queue_->currentRow() < 0 || queue_->currentRow() >= files_.size()) return;
            const QString source = files_.at(queue_->currentRow());
            auto cached = imageResults_.constFind(source);
            const bool hadCache = cached != imageResults_.cend();
            const QString sourceIdentity = fileIdentity(source);
            if (hadCache && !sourceIdentity.isEmpty() && sourceIdentity != cached->sourceIdentity) {
                imageResults_.remove(source); queue_->item(queue_->currentRow())->setForeground(QBrush());
                selectQueueImage(queue_->currentRow()); return;
            }
            if (hadCache && primaryUsed && !image.isNull() &&
                (cached->result.sourceFrameSize.isEmpty() || original.size() == cached->result.sourceFrameSize)) {
                auto result = cached->result; result.originalImage = original; result.image = image;
                lastConfig_ = cached->config; displayResult(result); return;
            }
            if (hadCache) {
                imageResults_.remove(source); queue_->item(queue_->currentRow())->setForeground(QBrush());
                showNotice("此图片的临时结果缓存无法读取，请重新检测。", true);
                if (primaryUsed) { selectQueueImage(queue_->currentRow()); return; }
            }
            vision::InferenceResult preview;
            preview.source = source; preview.task = currentConfig().task;
            preview.originalImage = original; preview.image = image;
            attachGroundTruth(preview); canvas_->setResult(preview);
            sizeMetric_->setText(image.isNull() ? "—" : QString("%1 × %2").arg(image.width()).arg(image.height()));
            resultInfo_->setText(image.isNull() ? "无法读取此图片，请检查文件是否存在。" :
                !datasetSamples_.contains(source) ? "临时图片已就绪 · 无 GT，仅参与检测，不计入数据集评估。" : "图片已就绪，点击开始检测。");
        });
    }
    const quint64 generation = ++imagePreviewGeneration_;
    lastResult_ = {}; canvas_->clear();
    canvasTitle_->setText(QFileInfo(source).fileName());
    sizeMetric_->setText("—"); sizeMetric_->setToolTip("预览与导出图像尺寸");
    countMetric_->setText("—"); latencyMetric_->setText("—"); classMetric_->setText("—");
    predictionTable_->setRowCount(0); emptyResults_->show(); emptyResults_->setText("此图片尚无检测结果。");
    resultInfo_->setText("正在载入图片…"); exportButton_->setEnabled(false);
    const bool snapshot = cached != imageResults_.cend() && !cached->snapshotPath.isEmpty();
    imagePreviewLoader_->request(generation, snapshot ? cached->snapshotPath : source,
        snapshot ? source : QString(), cached != imageResults_.cend() ? cached->config.colorMode : currentConfig().colorMode);
}

void MainWindow::updateInputPreview()
{
    if (busy_ || !canvas_)
        return;
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    ++evaluationPreviewGeneration_; ++evaluationFilterGeneration_;
    evaluationReviewActive_ = false;
    resetEvaluationReviewNavigation();
    for (int index = 0; index < queue_->count(); ++index) queue_->item(index)->setHidden(false);
    clearImageResults();
    const auto mode = currentConfig().colorMode;
    static_cast<LazyImageQueue *>(queue_)->setColorMode(mode);
    if (sourceKind_ == vision::SourceKind::Images) { selectQueueImage(queue_->currentRow()); return; }
    const auto previous = canvas_->result();
    if (previous.image.isNull())
        return;
    vision::InferenceResult preview;
    preview.source = previous.source;
    preview.task = currentConfig().task;
    preview.originalImage = previous.originalImage.isNull() ? previous.image : previous.originalImage;
    preview.sourceFrameSize = previous.sourceFrameSize;
    preview.image = vision::inputPreviewImage(preview.originalImage, mode);
    lastResult_ = {};
    attachGroundTruth(preview);
    canvas_->setResult(preview);
    predictionTable_->setRowCount(0);
    emptyResults_->show();
    exportButton_->setEnabled(false);
    countMetric_->setText("—");
    latencyMetric_->setText("—");
    classMetric_->setText("—");
    resultInfo_->setText(mode == vision::InputColorMode::Grayscale ? "灰度预览已就绪，点击开始检测。"
                                                                   : "彩色预览已就绪，点击开始检测。");
}

void MainWindow::toggleRecording()
{
    if (!busy_ || sourceKind_ == vision::SourceKind::Images || recordingStopping_ || inferenceStopping_ ||
        failed_)
        return;
    if (recordingRequested_ || recordingActive_)
    {
        recordingRequested_ = false;
        recordingStopping_ = recordingActive_;
        detection_->requestStopRecording();
    }
    else
    {
        recordingRequested_ = true;
        detection_->requestStartRecording();
    }
    updateRecordingUi();
}

void MainWindow::updateRecordingUi()
{
    if (!recordButton_)
        return;
    const bool recording = recordingRequested_ || recordingActive_;
    recordButton_->setText(recordingStopping_ ? "正在保存…" : (recording ? "结束录制" : "开始录制"));
    recordButton_->setIcon(
        ui::icon(recording ? "stop" : "record", recording ? QColor("#ffb0b0") : QColor("#D2D2D2")));
    recordButton_->setEnabled(busy_ && sourceKind_ != vision::SourceKind::Images && !recordingStopping_ &&
                              !inferenceStopping_ && !failed_);
    recordButton_->setStyleSheet(recording ? "background:#432831; border-color:#8b4855; color:#ffb7b7;" : "");
    recordingStatus_->setStyleSheet(recording ? "color:#ffacb0; font-size:11px;" : "");
    if (recordingStopping_)
        recordingStatus_->setText("正在完成录像文件…");
    else if (recordingActive_)
    {
        const qint64 seconds = recordingElapsed_.elapsed() / 1000;
        recordingStatus_->setText(QString("● REC  %1:%2")
                                      .arg(seconds / 60, 2, 10, QChar('0'))
                                      .arg(seconds % 60, 2, 10, QChar('0')));
    }
    else if (recordingRequested_)
        recordingStatus_->setText("等待下一帧开始录制…");
    else
        recordingStatus_->setText(busy_ && sourceKind_ != vision::SourceKind::Images
                                      ? "可录制当前检测画面"
                                      : "视频 / 摄像头检测时可录制");
}

void MainWindow::refreshRecordings()
{
    if (!recordingsTable_)
        return;
    const QString selected = playbackPath_;
    const QDir directory(dataRoot_ + "/recordings");
    const auto files = directory.entryInfoList({"*.avi"}, QDir::Files, QDir::Time);
    const QSignalBlocker blocker(recordingsTable_);
    recordingsTable_->setRowCount(files.size());
    int selection = files.isEmpty() ? -1 : 0;
    for (int row = 0; row < files.size(); ++row)
    {
        const auto &file = files[row];
        QFile meta(file.absolutePath() + "/" + file.completeBaseName() + ".json");
        QJsonObject metadata;
        if (meta.open(QIODevice::ReadOnly))
            metadata = QJsonDocument::fromJson(meta.readAll()).object();
        const double fps = metadata.value("fps").toDouble();
        const double duration = metadata.value("duration_seconds")
                                    .toDouble(fps > 0 ? metadata.value("frames").toDouble() / fps : 0);
        const int width = metadata.value("content_width").toInt(metadata.value("width").toInt());
        const int height = metadata.value("content_height").toInt(metadata.value("height").toInt());
        const QStringList values = {file.lastModified().toString("yyyy-MM-dd HH:mm:ss") + "\n" +
                                        file.fileName(),
                                    duration > 0 ? QString::number(duration, 'f', 1) + " s" : "—",
                                    width > 0 && height > 0 ? QString("%1×%2").arg(width).arg(height) : "—",
                                    QString::number(file.size() / 1048576.0, 'f', 1) + " MB"};
        for (int column = 0; column < values.size(); ++column)
        {
            auto *item = new QTableWidgetItem(values[column]);
            item->setData(Qt::UserRole, file.absoluteFilePath());
            item->setToolTip(file.absoluteFilePath());
            recordingsTable_->setItem(row, column, item);
        }
        if (file.absoluteFilePath() == selected)
            selection = row;
    }
    recordingsCount_->setText(QString("本地录像 · %1 个文件").arg(files.size()));
    if (selection >= 0)
        recordingsTable_->setCurrentCell(selection, 0);
    // Signals are blocked while rebuilding rows; load the final selection once.
    selectRecording();
}

void MainWindow::selectRecording()
{
    stopRecordingPlayback();
    const int row = recordingsTable_->currentRow();
    if (row < 0 || !recordingsTable_->item(row, 0))
    {
        playbackPath_.clear();
        recordingCanvas_->clear();
        recordingPlayButton_->setEnabled(false);
        exportRecordingButton_->setEnabled(false);
        recordingDetails_->setText("还没有录像。在工作台检测视频或摄像头时，点击「开始录制」。");
        return;
    }
    playbackPath_ = recordingsTable_->item(row, 0)->data(Qt::UserRole).toString();
    QFile file(QFileInfo(playbackPath_).absolutePath() + "/" + QFileInfo(playbackPath_).completeBaseName() +
               ".json");
    QJsonObject metadata;
    if (file.open(QIODevice::ReadOnly))
        metadata = QJsonDocument::fromJson(file.readAll()).object();
    playbackContentSize_ = QSize(metadata.value("content_width").toInt(metadata.value("width").toInt()),
                                 metadata.value("content_height").toInt(metadata.value("height").toInt()));
    playbackCapture_ = std::make_unique<cv::VideoCapture>(QFile::encodeName(playbackPath_).constData());
    const bool opened = playbackCapture_->isOpened();
    recordingPlayButton_->setEnabled(opened);
    exportRecordingButton_->setEnabled(QFileInfo(playbackPath_).isFile());
    if (!opened)
    {
        recordingCanvas_->clear();
        recordingDetails_->setText("无法解码此录像，请检查文件是否完整。");
        return;
    }
    const QString color = metadata.value("color_mode").toString() == "grayscale" ? "灰度" : "彩色";
    recordingDetails_->setText(QString("%1 · %2 · %3 fps\n%4")
                                   .arg(stereoName(stereoMode(metadata.value("stereo_view").toString())))
                                   .arg(color)
                                   .arg(playbackCapture_->get(cv::CAP_PROP_FPS), 0, 'f', 1)
                                   .arg(QFileInfo(playbackPath_).fileName()));
    readRecordingFrame();
}

void MainWindow::toggleRecordingPlayback()
{
    if (playbackTimer_->isActive())
    {
        playbackTimer_->stop();
        recordingPlayButton_->setText("播放录像");
        return;
    }
    if (playbackPath_.isEmpty())
        return;
    if (!playbackCapture_ || !playbackCapture_->isOpened())
        playbackCapture_ = std::make_unique<cv::VideoCapture>(QFile::encodeName(playbackPath_).constData());
    if (!playbackCapture_->isOpened())
        return;
    const double rawFps = playbackCapture_->get(cv::CAP_PROP_FPS);
    const double fps = std::isfinite(rawFps) && rawFps >= .001 ? rawFps : 25;
    playbackTimer_->start(std::max(1, int(std::round(1000 / std::min(fps, 1000.0)))));
    recordingPlayButton_->setText("暂停播放");
}

void MainWindow::readRecordingFrame()
{
    if (!playbackCapture_ || !playbackCapture_->isOpened())
        return;
    cv::Mat frame;
    if (!playbackCapture_->read(frame) || frame.empty())
    {
        playbackCapture_->set(cv::CAP_PROP_POS_FRAMES, 0);
        playbackTimer_->stop();
        recordingPlayButton_->setText("播放录像");
        return;
    }
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage image = QImage(rgb.data, rgb.cols, rgb.rows, qsizetype(rgb.step), QImage::Format_RGB888).copy();
    if (playbackContentSize_.isValid() && playbackContentSize_.width() <= image.width() &&
        playbackContentSize_.height() <= image.height())
        image = image.copy(QRect(QPoint(), playbackContentSize_));
    vision::InferenceResult preview;
    preview.source = playbackPath_;
    preview.image = image;
    recordingCanvas_->setResult(preview);
}

void MainWindow::stopRecordingPlayback()
{
    if (playbackTimer_)
        playbackTimer_->stop();
    if (playbackCapture_)
        playbackCapture_->release();
    if (recordingPlayButton_)
        recordingPlayButton_->setText("播放录像");
}

void MainWindow::exportRecording()
{
    if (!QFileInfo(playbackPath_).isFile())
        return;
    const QString destination = QFileDialog::getSaveFileName(
        this, "导出录像", QDir::homePath() + "/" + QFileInfo(playbackPath_).fileName(), "AVI 视频 (*.avi)");
    if (destination.isEmpty() ||
        QFileInfo(destination).canonicalFilePath() == QFileInfo(playbackPath_).canonicalFilePath())
        return;
    if (QFileInfo(destination).suffix().compare("avi", Qt::CaseInsensitive) != 0)
    {
        showNotice("请将录像保存为 .avi 文件。", true);
        return;
    }
    QFile source(playbackPath_);
    QSaveFile target(destination);
    if (!source.open(QIODevice::ReadOnly) || !target.open(QIODevice::WriteOnly))
    {
        showNotice("无法导出录像，请检查保存位置。", true);
        return;
    }
    while (!source.atEnd())
    {
        const QByteArray block = source.read(1024 * 1024);
        if (block.isEmpty() || target.write(block) != block.size())
        {
            showNotice("录像写入失败，文件未提交。", true);
            return;
        }
    }
    if (!target.commit())
    {
        showNotice("录像保存失败：" + target.errorString(), true);
        return;
    }
    QFile meta(QFileInfo(playbackPath_).absolutePath() + "/" + QFileInfo(playbackPath_).completeBaseName() +
               ".json");
    QString error;
    if (meta.open(QIODevice::ReadOnly) &&
        !atomicWrite(QFileInfo(destination).absolutePath() + "/" + QFileInfo(destination).completeBaseName() +
                         ".json",
                     meta.readAll(), &error))
        showNotice("录像已导出，元数据未保存：" + error, true);
    else
        showNotice("录像已导出：" + destination);
}

QWidget *MainWindow::buildGuide()
{
    auto *f = card();
    f->setObjectName("guidePage");
    auto *l = new QVBoxLayout(f);
    l->setContentsMargins(24, 18, 24, 18);
    auto *html = new QTextBrowser;
    html->setObjectName("guideBrowser");
    html->setOpenExternalLinks(true);
    QFile guide(":/guide.html");
    if (guide.open(QIODevice::ReadOnly))
        html->setHtml(QString::fromUtf8(guide.readAll()));
    else
        html->setPlainText("使用指南资源无法读取，请查阅随版本提供的 docs/使用指南.md。");
    l->addWidget(html);
    return f;
}

void MainWindow::connectWorker()
{
    detection_ = new vision::app::DetectionController(this);
    connect(this, &MainWindow::startRequested, detection_, &vision::app::DetectionController::run, Qt::QueuedConnection);
    connect(detection_, &vision::app::DetectionController::resultReady, this,
            [this](const vision::InferenceResult &result) { onResult(result); });
    connect(detection_, &vision::app::DetectionController::imageResultReady, this,
            [this](const vision::InferenceResult &result, quint64 ticket) { onResult(result, ticket); });
    connect(detection_, &vision::app::DetectionController::modelReady, this,
            [this](const QString &backend, const vision::ModelConfig &config)
            {
                lastConfig_ = config;
                modelClassCount_ = config.modelClassCount;
                inspectedModelIdentity_ = activeModelIdentity_;
                modelInputChannels_ = config.inputChannels;
                updateModelMeta("已加载");
                if (isPtModel(modelPath_))
                {
                    nativeLabels_ = config.labels;
                    taskBox_->setCurrentIndex(static_cast<int>(config.task));
                }
                activeImageFingerprint_ = inferenceFingerprint(activeModelIdentity_);
                actualDeviceKnown_ = true;
                actualDevice_ = config.resolvedDevice;
                actualDeviceIndex_ =
                    config.resolvedDevice == vision::ComputeDevice::CUDA ? config.deviceIndex : -1;
                actualDeviceName_ = config.deviceName;
                actualDeviceNotice_ = config.deviceNotice;
                actualBackend_ = backend;
                updateTaskUi();
                updateDeviceUi();
            });
    connect(detection_, &vision::app::DetectionController::status, this,
            [this](const QString &message)
            {
                if (!failed_ && !inferenceStopping_)
                    showNotice(message);
            });
    connect(detection_, &vision::app::DetectionController::progress, this,
            [this](int done, int total)
            {
                if (sourceKind_ == vision::SourceKind::Images) {
                    updateRangeUi();
                    progress_->setRange(0, taskRangeFiles_.size());
                    progress_->setValue(completed_);
                    return;
                }
                completed_ = done;
                if (total > 0)
                {
                    progress_->setRange(0, total);
                    progress_->setValue(done);
                }
                else
                    progress_->setRange(0, 0);
            });
    connect(detection_, &vision::app::DetectionController::failed, this,
            [this](const QString &s)
            {
                failed_ = true;
                lastError_ = s;
                resultInfo_->setText("推理失败 · 请检查模型与输入配置。");
                showNotice(s, true);
                if (smokeDir_.isEmpty() && !closing_)
                    QMessageBox::warning(this, "推理未完成", s);
            });
    connect(detection_, &vision::app::DetectionController::finished, this, &MainWindow::onFinished);
    connect(detection_, &vision::app::DetectionController::recordingStarted, this,
            [this](const QString &)
            {
                recordingActive_ = true;
                recordingElapsed_.restart();
                recordingClock_->start();
                if (!recordingRequested_)
                {
                    recordingStopping_ = true;
                    detection_->requestStopRecording();
                }
                updateRecordingUi();
            });
    connect(detection_, &vision::app::DetectionController::recordingFinished, this,
            [this](const QString &path, qint64 frames, double)
            {
                recordingRequested_ = recordingActive_ = recordingStopping_ = false;
                recordingClock_->stop();
                refreshRecordings();
                updateRecordingUi();
                showNotice(QString("录制已保存 · %1 帧 · %2").arg(frames).arg(QFileInfo(path).fileName()));
            });
    connect(detection_, &vision::app::DetectionController::recordingFailed, this,
            [this](const QString &error)
            {
                recordingRequested_ = recordingActive_ = recordingStopping_ = false;
                recordingClock_->stop();
                updateRecordingUi();
                showNotice("录制未保存：" + error, true);
            });
}

bool MainWindow::previewDropLocked() const
{
    return busy_ || queuePopulating_ || droppedImageImportActive_ || evaluationWorker_ || gpuService_->isPreparing() ||
        resultIoPending_ || closing_ || (modelConversion_ && modelConversion_->isBusy()) || QApplication::activeModalWidget();
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() != QEvent::DragEnter && event->type() != QEvent::DragMove && event->type() != QEvent::Drop &&
        event->type() != QEvent::DragLeave)
        return QMainWindow::eventFilter(watched, event);
    auto *widget = qobject_cast<QWidget *>(watched);
    if (!widget || widget->window() != this) return QMainWindow::eventFilter(watched, event);
    bool inModelLibrary = false;
    for (auto *parent = widget; parent; parent = parent->parentWidget())
        if (parent->objectName() == "modelLibraryPage") { inModelLibrary = true; break; }
    if (inModelLibrary) {
        if (event->type() == QEvent::DragLeave) {
            if (modelLibraryExternalDragActive_) showModelImportHint(modelLibraryDropPrompt);
            return QMainWindow::eventFilter(watched, event);
        }
        auto *drop = static_cast<QDropEvent *>(event);
        // InternalMove carries Qt's item-model MIME data and must remain owned
        // by QListWidget; external file URLs use a separate import path.
        if (drop->source() == modelList_ || !drop->mimeData()->hasUrls())
            return QMainWindow::eventFilter(watched, event);
        drop->ignore();
        auto reject = [this](const QString &message) { showModelImportHint(message, true); };
        if (previewDropLocked()) { reject("请先结束当前任务，再导入模型；当前模型与任务保持不变。"); return true; }
        if (!(drop->possibleActions() & Qt::CopyAction)) {
            reject("模型库仅引用本地文件，不支持移动或链接操作；请直接拖入文件。"); return true;
        }
        QStringList input;
        for (const auto &url : drop->mimeData()->urls()) {
            if (!url.isLocalFile() || (!url.host().isEmpty() && url.host() != "localhost") ||
                !url.query().isEmpty() || !url.fragment().isEmpty()) {
                reject("只支持本地模型文件，不能混入网络地址；本次未导入任何模型。"); return true;
            }
            input.append(url.toLocalFile());
        }
        QStringList paths; QString error;
        if (!validateModelImport(input, &paths, &error)) { reject(error); return true; }
        if (event->type() == QEvent::Drop) {
            if (modelLibraryExternalDragActive_) showModelImportHint(modelLibraryDropPrompt);
            if (!importModelFiles(paths)) return true;
        } else {
            showModelImportHint(QString("松开以导入 %1 个模型；同一路径自动去重，原文件保留。").arg(paths.size()));
            modelLibraryExternalDragActive_ = true;
        }
        drop->setDropAction(Qt::CopyAction); drop->accept();
        return true;
    }
    bool inPreview = false;
    for (auto *parent = widget; parent; parent = parent->parentWidget())
        if (parent->objectName() == "imagePreviewPage") { inPreview = true; break; }
    if (!inPreview) return QMainWindow::eventFilter(watched, event);
    if (event->type() == QEvent::DragLeave) {
        canvas_->setDropActive(false);
        return QMainWindow::eventFilter(watched, event);
    }
    auto *drop = static_cast<QDropEvent *>(event);
    if (!drop->mimeData()->hasUrls()) return QMainWindow::eventFilter(watched, event);
    drop->ignore();
    canvas_->setDropActive(false);
    if (previewDropLocked()) { showNotice("请先停止当前任务，再拖入新的输入。", true); return true; }
    QStringList input;
    for (const auto &url : drop->mimeData()->urls()) {
        if (!url.isLocalFile()) { showNotice("只支持拖入本地文件，不能混入网络地址。", true); return true; }
        input.append(url.toLocalFile());
    }
    QStringList paths; QString error; PreviewDropKind kind;
    if (!classifyPreviewDrop(input, &paths, &kind, &error)) { showNotice(error, true); return true; }
    if (event->type() == QEvent::Drop && !handlePreviewDrop(paths)) return true;
    if (event->type() != QEvent::Drop) canvas_->setDropActive(true);
    drop->setDropAction(Qt::CopyAction); drop->accept();
    return true;
}

bool MainWindow::handlePreviewDrop(const QStringList &input)
{
    if (previewDropLocked()) { showNotice("请先停止当前任务，再拖入新的输入。", true); return false; }
    cancelPendingDatasetSessionRestore();
    QStringList paths; QString error; PreviewDropKind kind;
    if (!classifyPreviewDrop(input, &paths, &kind, &error)) { showNotice(error, true); return false; }
    if (kind == PreviewDropKind::Dataset) {
        loadDataset(paths.first(), evaluation_->datasetSplit(), evaluation_->metadataPath()); return true;
    }
    if (kind == PreviewDropKind::Video) { loadVideo(paths.first()); return true; }
    bool append = sourceKind_ == vision::SourceKind::Images && datasetManifest_.isEmpty();
    if (!datasetManifest_.isEmpty()) {
        // The modal decision must never apply to a source changed by a nested callback.
        const auto previousFiles = files_;
        const auto previousManifest = datasetManifest_;
        const auto importGeneration = datasetImportGeneration_;
        const auto queueGeneration = queuePopulationGeneration_;
        QDialog dialog(this);
        dialog.setObjectName("imageDropDatasetConflictDialog");
        dialog.setPalette(palette());
        dialog.setStyleSheet("QDialog#imageDropDatasetConflictDialog { background:#202020; border:1px solid #454545; }");
        dialog.setWindowTitle("拖入图片"); dialog.setModal(true); dialog.setMinimumWidth(480);
        auto *layout = new QVBoxLayout(&dialog); layout->setContentsMargins(24, 22, 24, 20); layout->setSpacing(14);
        layout->addWidget(text("如何处理当前数据集？", "sectionTitle"));
        auto *description = text(QString("已拖入 %1 张图片。临时加入会保留原数据集、标注与结果；新图片仅参与检测，不计入评估，也不会在下次启动恢复。").arg(paths.size()), "muted");
        description->setWordWrap(true); description->setMaximumWidth(540); layout->addWidget(description);
        auto *actions = new QHBoxLayout; actions->setSpacing(8);
        auto *replace = button("替换整个数据集"); replace->setObjectName("imageDropReplaceButton");
        auto *temporary = button("临时加入", {}, "primary"); temporary->setObjectName("imageDropAppendButton");
        auto *cancel = button("取消"); cancel->setObjectName("imageDropCancelButton"); cancel->setDefault(true);
        actions->addWidget(replace); actions->addWidget(temporary); actions->addWidget(cancel); layout->addLayout(actions);
        connect(replace, &QPushButton::clicked, &dialog, [&dialog] { dialog.done(1); });
        connect(temporary, &QPushButton::clicked, &dialog, [&dialog] { dialog.done(2); });
        connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
        cancel->setFocus();
        const int choice = dialog.exec();
        if (choice != 1 && choice != 2) return true;
        if (previewDropLocked() || previousFiles != files_ || previousManifest != datasetManifest_ ||
            importGeneration != datasetImportGeneration_ || queueGeneration != queuePopulationGeneration_) {
            showNotice("输入状态已变化，请重新拖入图片。", true); return false;
        }
        append = choice == 2;
    }
    importDroppedImages(paths, append);
    return true;
}

void MainWindow::importDroppedImages(const QStringList &paths, bool append)
{
    const QStringList existing = append ? files_ : QStringList{};
    const auto checkedPaths = std::make_shared<QStringList>();
    droppedImageImportActive_ = true; droppedImageAppend_ = append; droppedImageQueueChanged_ = false;
    droppedImageOriginalCount_ = files_.size(); droppedImageOriginalRow_ = queue_->currentRow();
    const auto cancelled = droppedImageImportCancellation_ = std::make_shared<std::atomic_bool>(false);
    const quint64 generation = ++droppedImageImportGeneration_;
    const auto error = std::make_shared<QString>();
    setBusy(true); progress_->setRange(0, 0); showNotice("正在后台检查拖入的图片…");
    auto *thread = QThread::create([paths, existing, checkedPaths, cancelled, error] {
        QSet<QString> seen;
        for (const auto &path : existing) {
            if (cancelled->load()) return;
            const QFileInfo info(path);
            const auto canonical = info.canonicalFilePath();
            seen.insert(canonical.isEmpty() ? info.absoluteFilePath() : canonical);
        }
        for (const auto &input : paths) {
            if (cancelled->load()) return;
            const QFileInfo info(input);
            if (!info.isFile() || !info.isReadable()) { *error = "无法读取图片，本次拖入未导入任何文件：" + input; return; }
            const auto canonical = info.canonicalFilePath();
            const QString path = canonical.isEmpty() ? info.absoluteFilePath() : canonical;
            if (seen.contains(path)) continue;
            QImageReader reader(path);
            if (!reader.canRead()) { *error = "无法读取图片，本次拖入未导入任何文件：" + path; return; }
            seen.insert(path); checkedPaths->append(path);
        }
    });
    connect(thread, &QThread::finished, this, [this, checkedPaths, cancelled, generation, error] {
        if (cancelled->load() || generation != droppedImageImportGeneration_ || !droppedImageImportActive_ || closing_) return;
        if (!error->isEmpty()) {
            droppedImageImportActive_ = false; setBusy(false); selectQueueImage(queue_->currentRow()); showNotice(*error, true); return;
        }
        const QStringList additions = *checkedPaths;
        if (additions.isEmpty()) {
            droppedImageImportActive_ = false; setBusy(false);
            showNotice("拖入的图片已在队列中；相同本地路径只保留一次。"); return;
        }
        if (!droppedImageAppend_) {
            clearDatasetContext(); clearImageResults(); files_.clear(); queue_->clear();
            lastResult_ = {}; canvas_->clear(); predictionTable_->setRowCount(0); emptyResults_->show(); exportButton_->setEnabled(false);
        }
        sourceKind_ = vision::SourceKind::Images;
        droppedImageQueueChanged_ = true;
        const int first = files_.size();
        const quint64 population = ++queuePopulationGeneration_;
        queuePopulating_ = true;
        auto offset = std::make_shared<int>(0);
        auto *timer = queuePopulationTimer_ = new QTimer(this); timer->setInterval(0);
        const QIcon placeholder = ui::icon("image");
        static_cast<LazyImageQueue *>(queue_)->setColorMode(currentConfig().colorMode);
        connect(timer, &QTimer::timeout, this, [this, additions, cancelled, generation, population, offset, first, timer, placeholder] {
            if (cancelled->load() || generation != droppedImageImportGeneration_ || population != queuePopulationGeneration_) {
                timer->stop(); timer->deleteLater(); return;
            }
            QElapsedTimer budget; budget.start();
            {
                QSignalBlocker blocker(queue_); queue_->setUpdatesEnabled(false);
                int added = 0;
                while (*offset < additions.size() && added < 96 && budget.elapsed() < 5) {
                    const auto &path = additions.at((*offset)++);
                    files_.append(path);
                    auto *item = new QListWidgetItem(placeholder, QFileInfo(path).fileName());
                    item->setData(Qt::UserRole, path); item->setData(Qt::UserRole + 11, true);
                    item->setToolTip(path + "\n临时图片 · 无 GT，仅参与检测，不计入评估；下次启动不恢复");
                    queue_->addItem(item); ++added;
                }
                queue_->setUpdatesEnabled(true);
            }
            progress_->setRange(0, additions.size()); progress_->setValue(*offset);
            showNotice(QString("正在载入临时图片 · %1 / %2 张").arg(*offset).arg(additions.size()));
            if (*offset < additions.size()) return;
            timer->stop(); timer->deleteLater(); queuePopulationTimer_ = nullptr; queuePopulating_ = false;
            droppedImageImportActive_ = false;
            // New temporary pictures must be reachable even after an error-only review.
            if (evaluationReviewActive_) {
                if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
                ++evaluationFilterGeneration_; evaluationReviewPending_ = false; evaluationReviewAvailable_ = false;
                evaluationReviewFilter_ = "all";
            }
            for (int row = 0; row < queue_->count(); ++row) queue_->item(row)->setHidden(false);
            { QSignalBlocker blocker(queue_); queue_->setCurrentRow(first); }
            updateSourceUi(); setBusy(false); selectQueueImage(first);
            static_cast<LazyImageQueue *>(queue_)->scheduleVisibleThumbnails();
            if (evaluationReviewActive_ && !evaluationReviewPending_) applyEvaluationFilter("all");
            showNotice(QString("已加入 %1 张临时图片 · 按拖入顺序排列，相同路径去重；不参与数据集评估").arg(additions.size()));
        });
        timer->start();
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater); thread->start();
}

void MainWindow::cancelDroppedImageImport()
{
    if (!droppedImageImportActive_) return;
    if (droppedImageImportCancellation_) droppedImageImportCancellation_->store(true);
    ++droppedImageImportGeneration_; ++queuePopulationGeneration_;
    if (queuePopulationTimer_) { queuePopulationTimer_->stop(); queuePopulationTimer_->deleteLater(); queuePopulationTimer_ = nullptr; }
    queuePopulating_ = false; droppedImageImportActive_ = false;
    if (droppedImageQueueChanged_) {
        const QSignalBlocker blocker(queue_);
        const int keep = droppedImageAppend_ ? droppedImageOriginalCount_ : 0;
        while (files_.size() > keep) { files_.removeLast(); delete queue_->takeItem(queue_->count() - 1); }
        queue_->setCurrentRow(droppedImageAppend_ ? droppedImageOriginalRow_ : -1);
    }
    setBusy(false); updateSourceUi(); selectQueueImage(queue_->currentRow());
    showNotice("已取消图片载入 · 本次追加的临时图片已撤销，原数据集和启动恢复记录保留。");
    if (closing_) QTimer::singleShot(0, this, &QWidget::close);
}

void MainWindow::chooseVideo()
{
    if (busy_) return;
    cancelPendingDatasetSessionRestore();
    const QString p = QFileDialog::getOpenFileName(
        this, "选择视频", projectRoot_, "视频 (*.mp4 *.avi *.mkv *.mov *.webm *.m4v);;全部文件 (*)");
    if (p.isEmpty())
        return;
    loadVideo(p);
}

void MainWindow::loadVideo(const QString &p)
{
    clearDatasetContext();
    clearImageResults();
    sourceKind_ = vision::SourceKind::Video;
    streamPath_ = p;
    files_.clear();
    queue_->clear();
    auto *i = new QListWidgetItem(ui::icon("video"), QFileInfo(p).fileName());
    i->setToolTip(p);
    queue_->addItem(i);
    lastResult_ = {};
    canvas_->clear();
    canvasTitle_->setText(QFileInfo(p).fileName());
    predictionTable_->setRowCount(0);
    emptyResults_->show();
    exportButton_->setEnabled(false);
    updateSourceUi();
    updateRecordingUi();
}
void MainWindow::chooseCamera()
{
    if (busy_) return;
    cancelPendingDatasetSessionRestore();
    clearDatasetContext();
    clearImageResults();
    sourceKind_ = vision::SourceKind::Camera;
    files_.clear();
    queue_->clear();
    queue_->addItem(new QListWidgetItem(ui::icon("camera"), "实时摄像头"));
    lastResult_ = {};
    canvas_->clear();
    canvasTitle_->setText("摄像头预览");
    predictionTable_->setRowCount(0);
    emptyResults_->show();
    exportButton_->setEnabled(false);
    updateSourceUi();
}
void MainWindow::updateSourceUi()
{
    const bool analysisWorkspace = pages_ && pages_->currentIndex() == ui::routeIndex(ui::Route::Workbench) &&
        workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    if (workspaceHeader_) workspaceHeader_->setVisible(!analysisWorkspace);
    cameraIndex_->setVisible(sourceKind_ == vision::SourceKind::Camera);
    const bool streaming = sourceKind_ != vision::SourceKind::Images;
    const bool datasetContext = !datasetManifest_.isEmpty();
    // Dataset reports carry their own metrics. Reclaim their duplicated header
    // and the image-only recording placeholder so the canvas fits a small window.
    for (QLabel *value : {countMetric_, latencyMetric_, classMetric_, sizeMetric_})
        value->parentWidget()->setVisible(!datasetContext && (streaming || !files_.isEmpty()));
    countMetric_->parentWidget()->parentWidget()->setVisible(!datasetContext && (streaming || !files_.isEmpty()) &&
        (!workbenchTabs_ || workbenchTabs_->currentIndex() == 0));
    if (evaluation_) evaluation_->controlsWidget()->setVisible(datasetContext || !evaluation_->datasetYaml().isEmpty());
    sourceLabel_->setVisible(!datasetContext);
    recordButton_->setVisible(streaming);
    recordingStatus_->setVisible(streaming);
    stereoLabel_->setVisible(streaming);
    stereoView_->setVisible(streaming);
    stereoView_->setEnabled(streaming && !busy_);
    const QString view = stereoName(stereoMode(stereoView_->currentData().toString()));
    if (sourceKind_ == vision::SourceKind::Images)
        sourceLabel_->setText(datasetManifest_.isEmpty()
            ? (files_.isEmpty() ? QStringLiteral("拖入图片 / 数据集，或连接实时输入") : QString("临时图片 · %1 张 · 仅检测").arg(files_.size()))
            : QString("YOLO 数据集 · %1 · %2 张标注图片%3").arg(datasetManifest_.value("split").toString())
                  .arg(datasetManifest_.value("samples").toArray().size())
                  .arg(files_.size() > datasetSamples_.size() ? QString(" + %1 张临时图片").arg(files_.size() - datasetSamples_.size()) : QString()));
    else if (sourceKind_ == vision::SourceKind::Video)
        sourceLabel_->setText("视频 · " + QFileInfo(streamPath_).fileName() + " · " + view);
    else
        sourceLabel_->setText(QString("摄像头 %1 · %2 · 点击开始连接").arg(cameraIndex_->value()).arg(view));
    if (lastResult_.image.isNull())
    {
        countMetric_->setText("—");
        latencyMetric_->setText("—");
        classMetric_->setText("—");
        if (sourceKind_ != vision::SourceKind::Images || files_.isEmpty())
            sizeMetric_->setText("—");
        resultInfo_->setText(sourceKind_ == vision::SourceKind::Images && files_.isEmpty()
                                 ? "请选择 YOLO 数据集、视频或摄像头。"
                                 : "输入已就绪，点击开始检测。");
    }
    sourceLabel_->setToolTip(sourceLabel_->text());
    updateRecordingUi();
    updateRangeUi();
}


QString MainWindow::modelFileIdentity() const
{
    return fileIdentity(modelPath_);
}

QString MainWindow::inferenceFingerprint(const QString &modelIdentity) const
{
    auto c = currentConfig();
    std::sort(c.classFilter.begin(), c.classFilter.end());
    QByteArray bytes;
    QDataStream out(&bytes, QIODevice::WriteOnly);
    out << (modelIdentity.isEmpty() ? modelFileIdentity() : modelIdentity) << int(c.task) << c.inputSize << c.confidence << c.iou
        << int(c.colorMode) << c.swapRB << c.scale << c.meanR << c.meanG << c.meanB
        << int(c.device) << c.deviceIndex << c.labels << c.classFilter << c.maxDetections << c.cpuThreads;
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

void MainWindow::invalidateConfiguredResults()
{
    if (initializing_ || busy_ || !canvas_)
        return;
    if (classFilterError_.isEmpty() && !imageResults_.isEmpty() &&
        imageConfigFingerprint_ == inferenceFingerprint()) return;
    updateInputPreview();
    updateStartEligibility();
}

QStringList MainWindow::selectedImageRange(QString *error) const
{
    if (error) error->clear();
    if (files_.isEmpty()) {
        if (error) *error = "请先拖入图片、导入 YOLO 数据集，或选择视频 / 摄像头";
        return {};
    }
    const int mode = detectionRange_->currentIndex();
    if (mode == 0)
        return files_;
    const int row = queue_->currentRow();
    if (row < 0 || row >= files_.size() || !queue_->selectionModel()->hasSelection()) {
        if (error) *error = "请先在输入队列中选中一张图片";
        return {};
    }
    return mode == 1 ? files_.mid(row) : QStringList{files_.at(row)};
}

QStringList MainWindow::selectedEvaluationRange(QString *error) const
{
    if (error) error->clear();
    if (datasetManifest_.isEmpty() || datasetSamples_.isEmpty()) {
        if (error) *error = "评估需要带标注的 YOLO 数据集；临时图片仅参与检测。";
        return {};
    }
    const int mode = detectionRange_->currentIndex();
    const int selected = queue_->currentRow();
    if (mode != 0 && (selected < 0 || selected >= files_.size() || !queue_->selectionModel()->hasSelection())) {
        if (error) *error = "请先选中一张原数据集图片，再选择此评估范围。";
        return {};
    }
    if (mode != 0 && !datasetSamples_.contains(files_.at(selected))) {
        if (error) *error = "当前选中的是临时图片，没有 GT；请选择原数据集图片，或评估整个原数据集。";
        return {};
    }
    QStringList range;
    const int first = mode == 0 ? 0 : selected;
    const int end = mode == 2 ? first + 1 : files_.size();
    for (int row = first; row < end; ++row)
        if (datasetSamples_.contains(files_.at(row))) range.append(files_.at(row));
    if (range.isEmpty() && error) *error = "本次范围没有可评估的原数据集图片。";
    return range;
}

void MainWindow::updateRangeUi()
{
    if (!detectionRangePanel_ || !rangeHint_ || !runButton_) return;
    const bool images = sourceKind_ == vision::SourceKind::Images;
    detectionRangePanel_->setVisible(images);
    detectionRange_->setEnabled(images && !busy_);
    QString error;
    const auto range = busy_ ? taskRangeFiles_ : selectedImageRange(&error);
    const int first = busy_ ? taskRangeFirst_ : (range.isEmpty() ? 0 : files_.indexOf(range.first()) + 1);
    const int last = busy_ ? taskRangeLast_ : (range.isEmpty() ? 0 : files_.indexOf(range.last()) + 1);
    rangeHint_->setText(!error.isEmpty() ? error
        : QString("本次检测 %1 张：第 %2～%3 张%4")
              .arg(range.size()).arg(first).arg(last)
              .arg(busy_ ? QString(" · 完成 %1/%2").arg(completed_).arg(range.size()) : QString()));
    rangeHint_->setStyleSheet(error.isEmpty() ? QString() : "color: #FFB4AB;");
    updateStartEligibility();
}

void MainWindow::updateClassFilterUi()
{
    if (!classFilter_ || !classFilterHint_) return;
    const auto task = static_cast<vision::ModelTask>(taskBox_->currentIndex());
    const bool classification = task == vision::ModelTask::Classification;
    classFilter_->setEnabled(!busy_ && !classification);
    classFilterError_.clear();
    QVector<int> ids;
    QString hint;
    if (classification) {
        hint = "图像分类不使用目标检测类别筛选";
    } else if (!vision::parseClassFilter(classFilter_->text(), &ids, &classFilterError_)) {
        hint = classFilterError_;
    } else if (ids.isEmpty()) {
        hint = "留空检测全部类别";
    } else {
        const bool current = !modelFileIdentity().isEmpty() && inspectedModelIdentity_ == modelFileIdentity();
        if (!current || modelInspectionPending_) {
            classFilterError_ = modelInspectionPending_ ? "正在后台读取模型类别，完成后可开始"
                : (modelInspectionError_.isEmpty() ? "尚未取得模型类别信息，请重新选择模型"
                                                  : "无法校验类别：" + modelInspectionError_.left(180));
        } else {
            vision::validateClassFilter(ids, modelClassCount_, task, &classFilterError_);
        }
        if (!classFilterError_.isEmpty()) hint = classFilterError_;
        else {
            QStringList values;
            for (int id : ids) values.append(QString::number(id));
            hint = QString("保留原始 ID %1 · 模型 %2 类").arg(values.join(",")).arg(modelClassCount_);
            if (ids.contains(32)) hint += "\nCOCO ID 32 是 sports ball，包含多种球类，并非只识别足球。";
        }
    }
    classFilterHint_->setText(hint);
    classFilterHint_->setStyleSheet(classFilterError_.isEmpty() ? QString() : "color: #FFB4AB;");
    updateStartEligibility();
}

void MainWindow::updateStartEligibility()
{
    if (!runButton_ || !classFilter_) return;
    QString rangeError;
    if (sourceKind_ == vision::SourceKind::Images) selectedImageRange(&rangeError);
    QString evaluationRangeError;
    const auto evaluationFiles = selectedEvaluationRange(&evaluationRangeError);
    const bool ready = !busy_ && !queuePopulating_ && !gpuService_->isPreparing() && !resultIoPending_ &&
        !modelPath_.isEmpty() && rangeError.isEmpty() && (!modelConversion_ || !modelConversion_->isBusy());
    const bool canEvaluate = evaluation_ && sourceKind_ == vision::SourceKind::Images &&
        !datasetManifest_.isEmpty() && evaluationRangeError.isEmpty() && !evaluationFiles.isEmpty() &&
        evaluation_->canEvaluate() && !modelInspectionPending_;
    const bool evaluationPage = workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    runButton_->setEnabled(ready && (evaluationPage ? canEvaluate : classFilterError_.isEmpty()));
    runButton_->setText("开始检测");
    if (evaluationStartButton_) {
        evaluationStartButton_->setEnabled(ready && canEvaluate);
        evaluationStartButton_->setVisible(!busy_);
        evaluationStopButton_->setVisible(busy_);
        evaluationStopButton_->setEnabled(busy_ && stopButton_->isEnabled());
        evaluationRange_->setEnabled(!busy_ && sourceKind_ == vision::SourceKind::Images && !datasetManifest_.isEmpty());
        evaluationTaskSummary_->setText(QString("%1 · %2 px · 原数据集 %3 张")
            .arg(modelDisplayName(modelPath_)).arg(inputSize_->value()).arg(datasetSamples_.size()));
        evaluationTaskSummary_->setToolTip(modelPath_);
        evaluationRangeHint_->setText(sourceKind_ != vision::SourceKind::Images || datasetManifest_.isEmpty()
            ? "评估需要带标注的 YOLO 数据集；视频与摄像头请在图像预览页运行。"
            : !evaluationRangeError.isEmpty() ? evaluationRangeError :
              QString("本次评估 %1 张：原数据集第 %2～%3 张 · 临时图片不计入评估%4")
                .arg(busy_ ? taskRangeFiles_.size() : evaluationFiles.size())
                .arg(busy_ ? taskRangeFirst_ : files_.indexOf(evaluationFiles.first()) + 1)
                .arg(busy_ ? taskRangeLast_ : files_.indexOf(evaluationFiles.last()) + 1)
                .arg(busy_ ? QString(" · 完成 %1/%2").arg(completed_).arg(taskRangeFiles_.size()) : QString()));
        evaluationRangeHint_->setStyleSheet(evaluationRangeError.isEmpty() ? QString() : "color: #FFB4AB;");
    }
}

void MainWindow::startWorkbenchTask()
{
    if (busy_ || queuePopulating_) return;
    const bool evaluate = workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    if (evaluate && !evaluation_->canEvaluate()) return;
    evaluation_->setMode(evaluate);
    if (evaluate) startEvaluation();
    else startInference();
}

void MainWindow::inspectSelectedModel()
{
    if (initializing_ || busy_) return;
    if (isPtModel(modelPath_) && vision::cpuRuntimeDownloadNeeded()) return;
    if (modelInspectionCancellation_) modelInspectionCancellation_->store(true);
    const quint64 generation = ++modelInspectionGeneration_;
    modelInspectionCancellation_ = std::make_shared<std::atomic_bool>(false);
    const auto cancelled = modelInspectionCancellation_;
    inspectedModelIdentity_.clear();
    modelInspectionError_.clear();
    modelClassCount_ = 0;
    const QString identity = modelFileIdentity();
    if (identity.isEmpty()) {
        modelInspectionPending_ = false;
        updateClassFilterUi();
        return;
    }
    modelInspectionPending_ = true;
    updateClassFilterUi();
    const auto config = currentConfig();
    struct Inspection { vision::ModelConfig config; QString error; };
    const auto output = std::make_shared<Inspection>();
    auto *thread = QThread::create([config, cancelled, output] {
        try {
            output->config = vision::VisionEngine::inspectModel(config, [cancelled] { return cancelled->load(); });
        } catch (const std::exception &error) { output->error = QString::fromUtf8(error.what()); }
    });
    thread->setObjectName("modelClassInspection");
    connect(thread, &QThread::finished, this, [this, generation, identity, cancelled, output] {
        if (generation != modelInspectionGeneration_ || cancelled->load()) return;
        modelInspectionPending_ = false;
        if (identity != modelFileIdentity()) {
            invalidateConfiguredResults();
            inspectSelectedModel();
            return;
        }
        modelInspectionError_ = output->error;
        if (output->error.isEmpty()) {
            modelClassCount_ = output->config.modelClassCount;
            modelInputChannels_ = output->config.inputChannels;
            inspectedModelIdentity_ = identity;
            if (isPtModel(modelPath_) || output->config.task == vision::ModelTask::Classification) {
                const auto previousTask = taskBox_->currentIndex();
                const QSignalBlocker block(taskBox_);
                if (isPtModel(modelPath_)) nativeLabels_ = output->config.labels;
                taskBox_->setCurrentIndex(int(output->config.task));
                if (previousTask != taskBox_->currentIndex()) invalidateConfiguredResults();
            }
            updateModelMeta("已识别");
            updateTaskUi();
        }
        updateClassFilterUi();
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
}

void MainWindow::addFiles(const QStringList &input)
{
    if (busy_) return;
    if (!loadingDataset_) clearDatasetContext();
    if (sourceKind_ != vision::SourceKind::Images)
    {
        clearImageResults();
        files_.clear();
        queue_->clear();
    }
    sourceKind_ = vision::SourceKind::Images;
    int first = -1;
    for (const QString &p : input)
    {
        const QString absolute = QFileInfo(p).absoluteFilePath();
        if (files_.contains(absolute))
            continue;
        QImageReader reader(absolute);
        if (!reader.canRead()) continue;
        if (first < 0) first = files_.size();
        files_.append(absolute);
        auto *item = new QListWidgetItem(ui::icon("image"), QFileInfo(p).fileName());
        item->setData(Qt::UserRole, absolute);
        item->setToolTip(absolute);
        queue_->addItem(item);
    }
    if (first >= 0)
        queue_->setCurrentRow(first);
    updateSourceUi();
}
void MainWindow::importModel()
{
    if (previewDropLocked()) { showModelImportHint("请先结束当前任务，再导入模型。", true); return; }
    const QStringList paths =
        QFileDialog::getOpenFileNames(this, "导入视觉模型", projectRoot_ + "/models",
                                     "视觉模型 (*.onnx *.pt *.torchscript);;ONNX 模型 (*.onnx);;"
                                     "PyTorch / TorchScript 模型 (*.pt *.torchscript)");
    if (!paths.isEmpty()) importModelFiles(paths);
}
void MainWindow::showModelImportHint(const QString &message, bool error)
{
    modelLibraryExternalDragActive_ = false;
    if (modelLibraryDropHint_) {
        modelLibraryDropHint_->setText(message);
        modelLibraryDropHint_->setStyleSheet(error ? "color:#FFB4BB;" : "color:#B8B8B8;");
    }
    showNotice(message, error);
}
bool MainWindow::validateModelImport(const QStringList &input, QStringList *paths, QString *error) const
{
    paths->clear();
    if (input.isEmpty()) { *error = "请拖入 ONNX / PT / TorchScript 本地模型文件。"; return false; }
    // Keep an existing library key (including its display name, notes and
    // profiles) when another spelling or symlink names the same real file.
    QHash<QString, QString> existing;
    for (const auto &path : models_) {
        const QString canonical = QFileInfo(path).canonicalFilePath();
        if (!canonical.isEmpty() && !existing.contains(canonical)) existing.insert(canonical, path);
    }
    QSet<QString> seen;
    for (const auto &path : input) {
        const QFileInfo file(path);
        const QString suffix = file.suffix().toLower();
        if (!file.isFile() || !file.isReadable() || file.size() <= 0 ||
            (suffix != "onnx" && suffix != "pt" && suffix != "torchscript")) {
            *error = "仅支持可读取且非空的 .onnx / .pt / .torchscript 文件，不能混入文件夹或其他文件；本次未导入任何模型。";
            return false;
        }
        const QString canonical = file.canonicalFilePath();
        if (canonical.isEmpty()) { *error = "无法确定模型文件的本地路径；本次未导入任何模型。"; return false; }
        const QString actualSuffix = QFileInfo(canonical).suffix().toLower();
        if (actualSuffix != "onnx" && actualSuffix != "pt" && actualSuffix != "torchscript") {
            *error = "模型文件实际路径的格式不受支持；本次未导入任何模型。"; return false;
        }
        if (!seen.contains(canonical)) {
            seen.insert(canonical);
            paths->append(existing.value(canonical, canonical));
        }
    }
    return true;
}
bool MainWindow::importModelFiles(const QStringList &input)
{
    if (previewDropLocked()) { showModelImportHint("请先结束当前任务，再导入模型；当前模型与任务保持不变。", true); return false; }
    QStringList paths; QString error;
    if (!validateModelImport(input, &paths, &error)) { showModelImportHint(error, true); return false; }
    const QString selected = paths.last();
    // Online-runtime preparation may be cancelled. Do not mutate the library
    // until it is available; the callback rechecks busy state and every file.
    if (isPtModel(selected) && !ui::ensureCpuRuntime(this, [this, paths] { importModelFiles(paths); })) return false;
    QStringList next = models_;
    int added = 0;
    for (const auto &path : paths)
        if (!next.contains(path)) { next.append(path); ++added; }
    // Only the final model is globally selected and preloaded. Selecting it
    // before committing the batch also leaves cancellation/failure unchanged.
    setModel(selected);
    if (modelPath_ != QFileInfo(selected).absoluteFilePath()) return false;
    models_ = next;
    refreshModelLibrary();
    persist();
    showModelImportHint(added ? QString("已添加 %1 个模型；按输入顺序追加，已选用 %2，原文件保留。").arg(added).arg(modelDisplayName(modelPath_))
                             : QString("模型已在库中，未重复添加；已选用 %1。").arg(modelDisplayName(modelPath_)));
    return true;
}
bool MainWindow::rememberConvertedModel(const QString &path, const QJsonObject &metadata)
{
    const QFileInfo file(path);
    if (!file.isFile() || file.isSymLink() || !validConvertedModelMetadata(file.absoluteFilePath(), metadata) ||
        metadata.value("bytes").toDouble(-1) != double(file.size()))
    {
        showNotice("转换结果的模型元数据或文件大小不匹配，请重新转换后再加入模型库。", true);
        return false;
    }
    convertedModelProfiles_.insert(file.absoluteFilePath(),
        QVariantMap{{"fileSize", file.size()}, {"fileModifiedMs", file.lastModified().toMSecsSinceEpoch()},
                    {"metadata", metadata.toVariantMap()}});
    persist();
    return true;
}
QJsonObject MainWindow::convertedModelProfile(const QString &path) const
{
    const QFileInfo file(path);
    const auto profile = convertedModelProfiles_.value(file.absoluteFilePath()).toMap();
    if (!file.isFile() || file.isSymLink() || profile.value("fileSize").toLongLong() != file.size() ||
        profile.value("fileModifiedMs").toLongLong() != file.lastModified().toMSecsSinceEpoch())
        return {};
    const auto metadata = QJsonObject::fromVariantMap(profile.value("metadata").toMap());
    return validConvertedModelMetadata(file.absoluteFilePath(), metadata) ? metadata : QJsonObject{};
}
bool MainWindow::applyConvertedModelProfile(const QString &path)
{
    const auto metadata = convertedModelProfile(path);
    if (metadata.isEmpty())
        return false;
    nativeLabels_ = convertedModelLabels(metadata);
    modelInputChannels_ = metadata.value("input_channels").toInt();
    taskBox_->setCurrentIndex(metadata.value("task").toString() == "classify"
                                  ? int(vision::ModelTask::Classification)
                                  : int(metadata.value("layout").toString() == "v5"
                                            ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8));
    inputSize_->setValue(metadata.value("shape").toArray()[2].toInt());
    if (modelInputChannels_ == 1)
        inputColorMode_->setCurrentIndex(inputColorMode_->findData("grayscale"));
    scale_->setValue(1.0 / 255.0);
    meanR_->setValue(0);
    meanG_->setValue(0);
    meanB_->setValue(0);
    return true;
}
void MainWindow::setModel(const QString &path, bool force)
{
    if (busy_)
        return;
    if (path.isEmpty())
    {
        actualDeviceKnown_ = false;
        actualDeviceNotice_.clear();
        modelPath_.clear();
        if (modelConversion_) modelConversion_->setSourceModel({});
        modelInputChannels_ = 0;
        modelClassCount_ = 0;
        inspectedModelIdentity_.clear();
        modelInspectionError_.clear();
        if (modelInspectionCancellation_) modelInspectionCancellation_->store(true);
        ++modelInspectionGeneration_;
        modelInspectionPending_ = false;
        nativeLabels_.clear();
        modelName_->setText("尚未选择模型");
        modelMeta_->setText("请在模型库中选择一个模型。");
        modelName_->setToolTip({});
        updateTaskUi();
        updateInputPreview();
        updateClassFilterUi();
        displayModelStructure();
        persist();
        return;
    }
    QFileInfo f(path);
    if (!f.isFile())
    {
        refreshModelLibrary();
        showNotice("模型文件不存在：" + path, true);
        return;
    }
    if (modelPath_ == f.absoluteFilePath() && !force &&
        (!convertedModelProfiles_.contains(modelPath_) || !convertedModelProfile(modelPath_).isEmpty())) {
        if (!initializing_ && !modelInspectionPending_ && inspectedModelIdentity_ != modelFileIdentity())
            inspectSelectedModel();
        return;
    }
    if (!initializing_ && isPtModel(f.absoluteFilePath()) &&
        !ui::ensureCpuRuntime(this, [this, path, force] { setModel(path, force); })) return;
    const bool convertedBgr =
        isPtModel(f.absoluteFilePath()) && inputColorMode_->currentData().toString() == "bgr";
    modelPath_ = f.absoluteFilePath();
    modelClassCount_ = 0;
    inspectedModelIdentity_.clear();
    modelInspectionError_.clear();
    actualDeviceKnown_ = false;
    actualDeviceNotice_.clear();
    modelInputChannels_ = 0;
    nativeLabels_.clear();
    modelName_->setText(modelDisplayName(modelPath_));
    if (modelConversion_) modelConversion_->setSourceModel(modelPath_, modelDisplayName(modelPath_));
    updateModelMeta("已选择");
    modelName_->setToolTip(modelPath_);
    if (!models_.contains(modelPath_))
        models_.append(modelPath_);
    const QString n = f.fileName().toLower();
    if (n.contains("yolov5"))
        taskBox_->setCurrentIndex(0);
    else if (n.contains("yolov8") || n.contains("yolo11"))
        taskBox_->setCurrentIndex(1);
    applyConvertedModelProfile(modelPath_);
    updateTaskUi();
    updateInputPreview();
    if (!initializing_) inspectSelectedModel();
    persist();
    if (modelList_->count() != models_.size())
        refreshModelLibrary();
    else
    {
        const QSignalBlocker blocker(modelList_);
        for (int row = 0; row < modelList_->count(); ++row)
            if (modelList_->item(row)->data(Qt::UserRole).toString() == modelPath_)
            {
                modelList_->setCurrentRow(row);
                break;
            }
    }
    displayModelStructure();
    showNotice(convertedBgr ? "PT 的彩色输入使用 RGB，已从 BGR 切换为 RGB；也可选择灰度。"
                            : "模型已全局选用，结构正在后台加载；开始检测时验证推理模型。");
}
void MainWindow::importLabels()
{
    if (busy_) return;
    const QString p = QFileDialog::getOpenFileName(this, "导入类别标签（每行一个名称）", projectRoot_,
                                                   "UTF-8 文本 (*.txt *.names);;全部文件 (*)");
    if (p.isEmpty())
        return;
    QFile f(p);
    if (!f.open(QIODevice::ReadOnly))
    {
        showNotice("标签读取失败：" + f.errorString(), true);
        return;
    }
    QString s = QString::fromUtf8(f.readAll());
    if (s.startsWith(QChar(0xfeff)))
        s.remove(0, 1);
    QStringList labels;
    for (const QString &line : s.split('\n'))
    {
        const QString v = line.trimmed();
        if (!v.isEmpty())
            labels.append(v);
    }
    if (labels.isEmpty())
    {
        showNotice("标签文件为空。", true);
        return;
    }
    labels_ = labels;
    labelsPath_ = p;
    labelButton_->setText(QString("类别标签 · %1 个").arg(labels_.size()));
    labelButton_->setToolTip(p);
    invalidateConfiguredResults();
    persist();
    showNotice("已导入 " + QString::number(labels_.size()) + " 个类别标签。");
}
vision::ModelConfig MainWindow::currentConfig() const
{
    vision::ModelConfig c;
    c.modelPath = modelPath_;
    c.device = computeMode(computeDevice_->currentData().toString());
    c.deviceIndex = gpuDeviceIndex_->currentData().toInt();
    c.task = static_cast<vision::ModelTask>(taskBox_->currentIndex());
    c.modelClassCount = modelClassCount_;
    if (c.task != vision::ModelTask::Classification)
        vision::parseClassFilter(classFilter_->text(), &c.classFilter);
    if (!labels_.isEmpty())
        c.labels = labels_;
    else if (isPtModel(modelPath_))
        c.labels = {};
    else if (!convertedModelProfile(modelPath_).isEmpty())
        c.labels = nativeLabels_;
    else
        c.labels = {}; // The backend maps COCO only when the actual model has 80 classes.
    c.inputSize = inputSize_->value();
    c.confidence = confidence_->value();
    c.iou = iou_->value();
    const bool grayscale = inputColorMode_->currentData().toString() == "grayscale";
    c.colorMode = grayscale ? vision::InputColorMode::Grayscale : vision::InputColorMode::Color;
    c.inputChannels = modelInputChannels_;
    c.swapRB = inputColorMode_->currentData().toString() != "bgr";
    c.scale = scale_->value();
    c.meanR = meanR_->value();
    c.meanG = grayscale ? c.meanR : meanG_->value();
    c.meanB = grayscale ? c.meanR : meanB_->value();
    if (isPtModel(modelPath_))
    {
        c.swapRB = true;
        c.scale = 1.0 / 255.0;
        c.meanR = c.meanG = c.meanB = 0;
    }
    const auto profile = modelEvaluationProfiles_.value(modelPath_).toMap();
    const QFileInfo file(modelPath_);
    const QString modelIdentity = file.isFile() ? file.absoluteFilePath() + "|" + QString::number(file.size()) + "|" +
        QString::number(file.lastModified().toMSecsSinceEpoch()) : QString();
    if (profile.value("adopted").toBool() && !modelIdentity.isEmpty() &&
        profile.value("model_identity").toString() == modelIdentity)
    {
        const auto adopted = profile.value("config").toMap();
        c.maxDetections = qBound(1, adopted.value("max_detections", 300).toInt(), 10000);
        c.cpuThreads = qBound(0, adopted.value("cpu_threads", 0).toInt(), 256);
    }
    return c;
}
void MainWindow::startInference()
{
    if (busy_ || gpuService_->isPreparing() || (modelConversion_ && modelConversion_->isBusy()))
        return;
    if (evaluation_ && evaluation_->evaluationMode()) { startEvaluation(); return; }
    if (isPtModel(modelPath_) && !ui::ensureCpuRuntime(this, [this] { startInference(); })) return;
    evaluationReviewActive_ = false;
    resetEvaluationReviewNavigation();
    if (resultIoPending_)
    {
        showNotice("正在后台保存已完成结果，保存完成后即可开始新任务。");
        return;
    }
    if (inspectedModelIdentity_ != modelFileIdentity() && !modelInspectionPending_)
        inspectSelectedModel();
    updateClassFilterUi();
    updateRangeUi();
    if (!classFilterError_.isEmpty())
        return;
    QString rangeError;
    const auto imageRange = sourceKind_ == vision::SourceKind::Images ? selectedImageRange(&rangeError) : QStringList{};
    if (!rangeError.isEmpty())
        return;
    if (modelPath_.isEmpty())
    {
        showNotice("请先导入 ONNX 或 PT 模型。", true);
        return;
    }
    if (sourceKind_ == vision::SourceKind::Images && files_.isEmpty())
    {
        showNotice("请先导入 YOLO 数据集或选择视频／摄像头。", true);
        return;
    }
    if (autoExport_->isChecked())
    {
        if (!QDir().mkpath(exportDir_))
        {
            showNotice("无法创建自动导出目录：" + exportDir_, true);
            return;
        }
    }
    vision::JobRequest req;
    req.config = currentConfig();
    req.sourceKind = sourceKind_;
    req.stereoView = sourceKind_ == vision::SourceKind::Images
                         ? vision::StereoView::Full
                         : stereoMode(stereoView_->currentData().toString());
    req.cameraIndex = cameraIndex_->value();
    req.files = sourceKind_ == vision::SourceKind::Video ? QStringList{streamPath_} : imageRange;
    taskRangeFiles_ = imageRange;
    activeModelIdentity_ = modelFileIdentity();
    activeImageFingerprint_ = inferenceFingerprint(activeModelIdentity_);
    taskSourceIdentities_.clear();
    for (const auto &source : imageRange) taskSourceIdentities_.insert(source, fileIdentity(source));
    taskRangeFirst_ = imageRange.isEmpty() ? 0 : files_.indexOf(imageRange.first()) + 1;
    taskRangeLast_ = imageRange.isEmpty() ? 0 : files_.indexOf(imageRange.last()) + 1;
    if (modelInspectionPending_) {
        if (modelInspectionCancellation_) modelInspectionCancellation_->store(true);
        ++modelInspectionGeneration_;
        modelInspectionPending_ = false;
    }
    failed_ = false;
    inferenceStopping_ = false;
    lastError_.clear();
    actualDeviceKnown_ = false;
    actualDeviceNotice_.clear();
    completed_ = 0;
    const QString fingerprint = activeImageFingerprint_;
    if (imageConfigFingerprint_ != fingerprint)
        clearImageResults();
    imageConfigFingerprint_ = fingerprint;
    // A same-configuration rerun preserves previously accepted results until
    // each new result atomically replaces its entry. Progress counts only this job.
    lastResult_ = {};
    lastConfig_ = req.config;
    predictionTable_->setRowCount(0);
    emptyResults_->show();
    resultInfo_->setText("模型正在加载，等待推理结果…");
    auto preview = canvas_->result();
    preview.predictions.clear();
    canvas_->setResult(preview);
    countMetric_->setText("—");
    latencyMetric_->setText("—");
    classMetric_->setText("—");
    setBusy(true);
    persist();
    detection_->prepareRecording(dataRoot_ + "/recordings");
    detection_->prepare();
    emit startRequested(req);
}
void MainWindow::stopInference()
{
    cancelPendingDatasetSessionRestore();
    if (!busy_)
        return;
    if (droppedImageImportActive_) { cancelDroppedImageImport(); return; }
    if (datasetImportCancellation_) datasetImportCancellation_->store(true);
    if (queuePopulating_) {
        ++queuePopulationGeneration_;
        if (queuePopulationTimer_) { queuePopulationTimer_->stop(); queuePopulationTimer_->deleteLater(); queuePopulationTimer_ = nullptr; }
        queuePopulating_ = false;
        clearDatasetContext(); files_.clear(); queue_->clear(); canvas_->clear(); lastResult_ = {};
        if (evaluationWorker_) evaluationWorker_->requestStop();
        else setBusy(false);
        updateSourceUi(); showNotice("已取消队列载入，可重新导入数据集。");
        if (closing_ && !evaluationWorker_) QTimer::singleShot(0, this, &QWidget::close);
        return;
    }
    if (evaluationWorker_) {
        evaluationWorker_->requestStop();
        stopButton_->setEnabled(false);
        updateStartEligibility();
        showNotice("停止请求已发送 · 已完成样本将保留，未完成任务不会作为完整评估成绩");
        return;
    }
    inferenceStopping_ = true;
    recordingRequested_ = false;
    recordingStopping_ = recordingActive_;
    detection_->requestStop();
    updateRecordingUi();
    stopButton_->setEnabled(false);
    updateStartEligibility();
    showNotice("停止请求已发送 · 正在结束当前推理，已完成结果保留");
}
void MainWindow::setBusy(bool busy)
{
    busy_ = busy;
    if (busy && evaluationReviewPending_) {
        if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
        ++evaluationFilterGeneration_;
        evaluationReviewPending_ = false;
        evaluationReviewAvailable_ = false;
    }
    if (busy && imagePreviewLoader_) { ++imagePreviewGeneration_; imagePreviewLoader_->cancel(); }
    if (evaluation_) evaluation_->setBusy(busy);
    if (modelConversion_) modelConversion_->setEnabled(!busy);
    if (!busy)
        inferenceStopping_ = false;
    for (auto *c : lockedControls_)
        c->setEnabled(!busy);
    static_cast<ModelLibraryList *>(modelList_)->setLibraryBusy(busy);
    runButton_->setVisible(!busy);
    stopButton_->setVisible(busy);
    stopButton_->setEnabled(busy);
    exportButton_->setEnabled(!busy && !lastResult_.image.isNull() && !lastResult_.demonstration);
    queue_->setEnabled(!busy);
    stereoView_->setEnabled(!busy && sourceKind_ != vision::SourceKind::Images);
    progress_->setVisible(busy);
    updateTaskUi();
    updateRecordingUi();
    updateDeviceUi();
    updateRangeUi();
    if (busy)
    {
        if (sourceKind_ == vision::SourceKind::Images) {
            progress_->setRange(0, taskRangeFiles_.size());
            progress_->setValue(0);
        } else progress_->setRange(0, 0);
    }
    updateEvaluationReviewUi();
    if (!busy && evaluationReviewActive_ && !evaluationReport_.isEmpty() && !evaluationReviewAvailable_ &&
        !evaluationReviewPending_ && evaluationReviewError_.isEmpty())
        applyEvaluationFilter(evaluationReviewFilter_);
}
void MainWindow::updateTaskUi()
{
    const bool pt = isPtModel(modelPath_);
    auto *colorItems = qobject_cast<QStandardItemModel *>(inputColorMode_->model());
    if (colorItems && colorItems->item(1))
        colorItems->item(1)->setEnabled(!pt);
    if (pt && inputColorMode_->currentData().toString() == "bgr")
    {
        const QSignalBlocker blocker(inputColorMode_);
        inputColorMode_->setCurrentIndex(0);
    }
    const bool grayscale = inputColorMode_->currentData().toString() == "grayscale";
    const bool classification = taskBox_->currentIndex() == 2;
    confidence_->setEnabled(!busy_ && !classification);
    iou_->setEnabled(!busy_ && !classification);
    taskBox_->setEnabled(!busy_ && !pt);
    taskBox_->setToolTip(pt ? "PT 模型的任务由文件内置架构自动识别" : "按照 ONNX 输出张量选择任务格式");
    inputColorMode_->setEnabled(!busy_);
    for (auto *control : QList<QWidget *>{scale_, meanR_})
        control->setEnabled(!busy_ && !pt);
    meanG_->setEnabled(!busy_ && !pt && !grayscale);
    meanB_->setEnabled(!busy_ && !pt && !grayscale);
    meanRLabel_->setText(grayscale ? "灰度均值" : "均值 R");
    meanR_->setAccessibleName(grayscale ? "灰度均值" : "均值 R");
    meanR_->setToolTip(grayscale ? "ONNX 灰度输入统一使用该均值；三通道模型的三个灰度通道使用相同值。"
                                 : "RGB / BGR 彩色输入的 R 通道均值");
    meanG_->setToolTip(grayscale ? "灰度模式统一使用灰度均值，忽略此项。" : "G 通道均值");
    meanB_->setToolTip(grayscale ? "灰度模式统一使用灰度均值，忽略此项。" : "B 通道均值");
    if (pt)
    {
        scale_->setValue(1.0 / 255.0);
        meanR_->setValue(0);
        meanG_->setValue(0);
        meanB_->setValue(0);
    }
    preprocessHint_->setText(
        pt ? (grayscale ? "灰度输入 · 模型原生归一化" : "PT 彩色使用 RGB · 原生预处理")
           : (grayscale ? "灰度输入 · 统一使用灰度均值" : "彩色输入 · RGB / BGR 通道顺序"));
    if (labels_.isEmpty())
        labelButton_->setText(pt ? (nativeLabels_.isEmpty()
                                        ? "类别标签 · 模型自动读取"
                                        : QString("类别标签 · 内置 %1 类").arg(nativeLabels_.size()))
                                 : (!convertedModelProfile(modelPath_).isEmpty()
                                        ? QString("类别标签 · 转换模型 %1 类").arg(nativeLabels_.size())
                                        : (classification ? "类别标签 · 数字类别"
                                            : (modelClassCount_ == 80 ? "类别标签 · 默认 COCO 80"
                                                : modelClassCount_ > 0 ? QString("类别标签 · 数字 %1 类").arg(modelClassCount_)
                                                                      : QString("类别标签 · 模型默认")))));
    else
        labelButton_->setText(QString("类别标签 · %1 个").arg(labels_.size()));
    updateEvaluationModel();
    updateClassFilterUi();
    updateDeviceUi();
}
void MainWindow::setComputeDevice(vision::ComputeDevice device, int index)
{
    index = qBound(0, index, 63);
    if (busy_ || gpuService_->isPreparing())
        return;
    {
        const QSignalBlocker choiceBlocker(computeDevice_), indexBlocker(gpuDeviceIndex_);
        const int row = computeDevice_->findData(vision::computeDeviceKey(device));
        computeDevice_->setCurrentIndex(row < 0 ? 0 : row);
        if (gpuDeviceIndex_->findData(index) < 0)
            gpuDeviceIndex_->addItem(QString("GPU %1 · 待检查").arg(index), index);
        gpuDeviceIndex_->setCurrentIndex(gpuDeviceIndex_->findData(index));
    }
    actualDeviceKnown_ = false;
    actualDeviceNotice_.clear();
    updateDeviceUi();
    updateInputPreview();
    persist();
}

void MainWindow::updateDeviceUi()
{
    if (!computeDevice_)
        return;
    const bool installing = gpuService_->isPreparing();
    const bool checking = gpuService_->isChecking();
    const auto requested = computeMode(computeDevice_->currentData().toString());
    computeDevice_->setEnabled(!busy_ && !installing);
    gpuDeviceIndex_->setEnabled(!busy_ && !installing && requested != vision::ComputeDevice::CPU);
    const bool showIndex = requested == vision::ComputeDevice::CUDA ||
                           (requested == vision::ComputeDevice::Auto && gpuDeviceIndex_->count() > 1);
    gpuDeviceLabel_->setVisible(showIndex);
    gpuDeviceIndex_->setVisible(showIndex);
    const bool ready = gpuService_->environment().value("ok").toBool() && gpuService_->environment().value("prepared").toBool() &&
                       gpuService_->environment().value("cuda_available").toBool();
    const QString reason = gpuService_->environment().value("reason").toString();
    QString hint;
    if (requested == vision::ComputeDevice::CPU)
        hint = QStringLiteral("使用 CPU · 无需 GPU 环境");
    else if (installing)
        hint = QStringLiteral("正在准备 GPU 支持 · 可在更多页取消");
    else if (checking)
        hint = QStringLiteral("正在检查 GPU 环境…");
    else if (ready)
        hint = QStringLiteral("GPU 环境已就绪 · 运行时验证所选设备");
    else
        hint = requested == vision::ComputeDevice::Auto
                   ? QStringLiteral("GPU 尚未就绪 · 自动模式可使用 CPU")
                   : QStringLiteral("GPU 尚未就绪 · 更多 → 准备 GPU 支持");
    deviceHint_->setText("●");
    deviceHint_->setStyleSheet(QString("color:%1; font-size:12px;").arg(ready || requested == vision::ComputeDevice::CPU ? "#A6E3B0" : "#FFB88A"));
    deviceHint_->setAccessibleName(hint);
    deviceHint_->setToolTip(reason.isEmpty() ? hint : hint + "\n" + reason);
    computeDevice_->setToolTip("自动模式优先使用 NVIDIA GPU；显式 GPU 失败时不回退。\n" + deviceHint_->toolTip());
    gpuDeviceIndex_->setToolTip(gpuDeviceIndex_->currentText() + "\n" + deviceHint_->toolTip());
    if (backendBadge_)
    {
        const QString device = actualDevice_ == vision::ComputeDevice::CUDA
                                   ? QString("GPU %1 · %2").arg(actualDeviceIndex_).arg(actualDeviceName_)
                                   : QStringLiteral("CPU");
        const QString actual = device + "\n" + actualBackend_;
        backendBadge_->setText(
            actualDeviceKnown_
                ? actual
                : QString("所选：%1 · %2").arg(computeName(requested)).arg(busy_ ? "正在验证" : "待运行"));
        backendBadge_->setToolTip(
            actualDeviceKnown_
                ? actual + (actualDeviceNotice_.isEmpty() ? QString() : "\n" + actualDeviceNotice_)
                : hint);
    }
    if (backendFooter_)
    {
        const QString footer =
            actualDeviceKnown_ ? QString("Qt 6.8.3  ·  %1  ·  %2")
                                     .arg(actualDevice_ == vision::ComputeDevice::CUDA ? "NVIDIA GPU" : "CPU",
                                          actualBackend_)
                               : QString("Qt 6.8.3  ·  %1待验证").arg(computeName(requested));
        backendFooter_->setText(backendFooter_->fontMetrics().elidedText(footer, Qt::ElideRight, 300));
        backendFooter_->setToolTip(footer + "\n" + actualDeviceName_ + "\n" + actualDeviceNotice_);
    }
    if (gpuCheckButton_)
    {
        gpuCheckButton_->setEnabled(!busy_ && !installing && !checking);
        gpuPrepareButton_->setEnabled(!busy_ && !installing && !checking);
        gpuCancelButton_->setVisible(installing);
        gpuCancelButton_->setEnabled(installing && !gpuService_->isCancelling());
    }
    if (runButton_)
        updateStartEligibility();
}



void MainWindow::appendGpuLog(const QString &message)
{
    if (message.trimmed().isEmpty())
        return;
    gpuLog_->show();
    gpuLog_->appendPlainText(message.left(16000).trimmed());
}

void MainWindow::checkGpuEnvironment()
{
    if (!busy_) gpuService_->checkEnvironment();
}



void MainWindow::prepareGpuEnvironment()
{
    if (!ui::ensureCpuRuntime(this, [this] { prepareGpuEnvironment(); })) return;
    if (!busy_) gpuService_->prepareEnvironment();
}

void MainWindow::cancelGpuPreparation()
{
    gpuService_->cancelPreparation();
}

void MainWindow::updateModelMeta(const QString &state)
{
    const QString channels =
        modelInputChannels_ > 0 ? QString("输入 %1 通道").arg(modelInputChannels_) : "输入通道待识别";
    modelMeta_->setText(QString("%1  ·  %2 MB  ·  %3  ·  %4")
                            .arg(modelFormat(modelPath_))
                            .arg(QFileInfo(modelPath_).size() / 1048576.0, 0, 'f', 1)
                            .arg(channels)
                            .arg(state));
}
void MainWindow::onResult(const vision::InferenceResult &input, quint64 ticket)
{
    auto r = input;
    attachGroundTruth(r);
    if (ticket && (!busy_ || inferenceStopping_))
    {
        detection_->acknowledgeImageResult(ticket);
        return;
    }
    if (r.modelClassCount > 0) {
        modelClassCount_ = r.modelClassCount;
        lastConfig_.modelClassCount = r.modelClassCount;
        updateClassFilterUi();
    }
    if (sourceKind_ == vision::SourceKind::Images) {
        ++completed_;
        cacheImageResult(r);
        updateRangeUi();
        progress_->setRange(0, taskRangeFiles_.size());
        progress_->setValue(completed_);
    }
    displayResult(r);
    if (sourceKind_ == vision::SourceKind::Images)
    {
        recordResult(r, false);
        const int row = files_.indexOf(r.source);
        if (row >= 0)
        {
            queue_->setCurrentRow(row);
            queue_->item(row)->setForeground(QColor("#60CDFF"));
        }
        saveImageResultInBackground(r, ticket);
    }
}

void MainWindow::displayResult(const vision::InferenceResult &input)
{
    auto r = input;
    if (!evaluationReviewActive_) attachGroundTruth(r);
    lastResult_ = r;
    if (r.evaluationReport.isEmpty()) {
    actualDeviceKnown_ = true;
    actualDevice_ = r.device;
    actualDeviceIndex_ = r.deviceIndex;
    actualDeviceName_ = r.deviceName;
    actualDeviceNotice_ = r.deviceNotice;
    actualBackend_ = r.backend;
    updateDeviceUi();
    updateModelMeta("已验证");
    }
    canvas_->setResult(r);
    canvasTitle_->setText(sourceKind_ == vision::SourceKind::Images
                              ? QFileInfo(r.source).fileName()
                              : QString("%1 · 第 %2 帧")
                                    .arg(sourceKind_ == vision::SourceKind::Camera
                                             ? "摄像头"
                                             : QFileInfo(streamPath_).fileName())
                                    .arg(r.frameNumber));
    if (r.stereoView != vision::StereoView::Full)
        canvasTitle_->setText(canvasTitle_->text() + " · " + stereoName(r.stereoView));
    countMetric_->setText(QString::number(r.predictions.size()));
    latencyMetric_->setText(r.hasInferenceTiming ? QString::number(r.inferenceMs, 'f', 1) : QStringLiteral("未记录"));
    QSet<int> cls;
    for (const auto &p : r.predictions)
        cls.insert(p.classId);
    classMetric_->setText(QString::number(cls.size()));
    sizeMetric_->setText(QString("%1 × %2").arg(r.image.width()).arg(r.image.height()));
    sizeMetric_->setToolTip(r.sourceFrameSize.isEmpty() ? "预览与导出图像尺寸"
                                                        : QString("原始输入 %1 × %2 · %3")
                                                              .arg(r.sourceFrameSize.width())
                                                              .arg(r.sourceFrameSize.height())
                                                              .arg(stereoName(r.stereoView)));
    resultInfo_->setText(QString("%1 · %2 个%3\n总处理 %4 ms")
                             .arg(r.hasCompleteInferenceConfig ? taskName(r.task) : QStringLiteral("目标检测 · 历史预测"))
                             .arg(r.predictions.size())
                             .arg(r.task == vision::ModelTask::Classification ? "分类结果" : "目标")
                             .arg(r.totalMs, 0, 'f', 1));
    QString deviceDescription;
    if (!r.hasInferenceTiming)
        resultInfo_->setText(QString("目标检测 · %1 个目标\n逐图耗时未记录").arg(r.predictions.size()));
    if (!r.hasVerifiedDevice)
        deviceDescription = "来源设备（未验证）· " + (r.sourceRuntimeDevice.isEmpty() ? QStringLiteral("未记录") : r.sourceRuntimeDevice);
    else if (r.device == vision::ComputeDevice::CPU)
    {
        QString backend = r.backend;
        backend.remove(QRegularExpression("\\s*/\\s*CPU\\b", QRegularExpression::CaseInsensitiveOption));
        deviceDescription = "CPU · " + backend;
    }
    else
    {
        QString name = r.deviceName;
        name.remove(
            QRegularExpression("^(NVIDIA\\s+)?(GeForce\\s+)?", QRegularExpression::CaseInsensitiveOption));
        deviceDescription = "GPU · " + (name.isEmpty() ? r.backend : name);
    }
    resultInfo_->setText(resultInfo_->text() + "\n" + deviceDescription);
    if (!r.deviceNotice.isEmpty())
        resultInfo_->setText(resultInfo_->text() + "\n" + r.deviceNotice.left(400));
    emptyResults_->setVisible(r.predictions.isEmpty());
    emptyResults_->setText("未发现符合阈值的目标\n\n可尝试降低置信度或检查模型配置。");
    if (r.hasGroundTruth) {
        resultInfo_->setText(resultInfo_->text() + (r.hasEvaluationMatches
            ? QString("\nGT 匹配 · TP %1 / FP %2 / FN %3\nGT 虚线 · 预测实线 · 原始类别 ID 保留")
                .arg(r.matchedPredictions.size())
                .arg(r.predictions.size() - r.ignoredPredictions.size() - r.matchedPredictions.size())
                .arg(r.groundTruth.size() - r.matchedGroundTruth.size())
            : QString("\nGT 标注预览 · 尚未完成该图的匹配评估")));
    }
    predictionTable_->setColumnCount(r.hasGroundTruth ? 4 : 3);
    predictionTable_->setHorizontalHeaderLabels(r.hasGroundTruth
        ? QStringList{"类别", "置信度", "类别 ID", "匹配"}
        : QStringList{"类别", "置信度", "类别 ID"});
    predictionTable_->setRowCount(r.predictions.size());
    predictionTable_->clearSelection();
    for (int i = 0; i < r.predictions.size(); ++i)
    {
        const auto &p = r.predictions[i];
        auto *name = new QTableWidgetItem(p.label);
        name->setForeground(QColor::fromHsv((p.classId * 53 + 156) % 360, 130, 225));
        name->setToolTip(QString("x: %1  y: %2\nw: %3  h: %4")
                             .arg(p.box.x(), 0, 'f', 1)
                             .arg(p.box.y(), 0, 'f', 1)
                             .arg(p.box.width(), 0, 'f', 1)
                             .arg(p.box.height(), 0, 'f', 1));
        predictionTable_->setItem(i, 0, name);
        auto *conf = new QTableWidgetItem(QString::number(p.confidence * 100, 'f', 1) + "%");
        conf->setTextAlignment(Qt::AlignCenter);
        predictionTable_->setItem(i, 1, conf);
        auto *id = new QTableWidgetItem(QString::number(p.classId));
        id->setTextAlignment(Qt::AlignCenter);
        id->setForeground(QColor("#A5A5A5"));
        id->setToolTip("模型原始 class_id，从 0 开始");
        predictionTable_->setItem(i, 2, id);
        if (r.hasGroundTruth) {
            const QString state = !r.hasEvaluationMatches ? "待评估" : r.ignoredPredictions.contains(i) ? "忽略" : r.matchedPredictions.contains(i) ? "TP" : "FP";
            auto *match = new QTableWidgetItem(state);
            match->setTextAlignment(Qt::AlignCenter);
            match->setForeground(QColor(state == "TP" ? "#6FE7B6" : state == "FP" ? "#FFB08F" : "#D2D2D2"));
            predictionTable_->setItem(i, 3, match);
        }
    }
    if (r.hasEvaluationMatches) {
        for (int g = 0; g < r.groundTruth.size(); ++g) if (!r.matchedGroundTruth.contains(g)) {
            const int row = predictionTable_->rowCount(); predictionTable_->insertRow(row);
            const auto &gt = r.groundTruth[g];
            const QStringList values = {"GT · " + gt.label, "—", QString::number(gt.classId), "FN"};
            for (int c = 0; c < values.size(); ++c) {
                auto *item = new QTableWidgetItem(values[c]); item->setForeground(QColor("#E6AAFF"));
                item->setToolTip("数据集原始类别 ID；此标注没有匹配的目标预测。");
                predictionTable_->setItem(row, c, item);
            }
        }
    }
    exportButton_->setEnabled(!busy_ && !r.image.isNull() && !r.demonstration);
}
void MainWindow::onFinished(bool cancelled)
{
    recordingRequested_ = recordingActive_ = recordingStopping_ = false;
    recordingClock_->stop();
    if (sourceKind_ != vision::SourceKind::Images && !lastResult_.image.isNull())
    {
        recordResult(lastResult_);
        if (autoExport_->isChecked())
        {
            QString error;
            if (!writeResult(lastResult_, exportDir_, &error))
            {
                failed_ = true;
                lastError_ = "自动导出失败：" + error;
                showNotice(lastError_, true);
            }
        }
    }
    setBusy(false);
    if (sourceKind_ == vision::SourceKind::Images) {
        // Restores the selected accepted result even when cancellation happened
        // before the first new frame and the current queue row never changed.
        selectQueueImage(queue_->currentRow());
    }
    if (activeModelIdentity_ != modelFileIdentity()) {
        invalidateConfiguredResults();
        inspectSelectedModel();
    }
    refreshHistory();
    if (failed_ && !lastError_.isEmpty())
        showNotice(lastError_, true);
    else if (!failed_)
        showNotice(cancelled ? (sourceKind_ == vision::SourceKind::Images
                                   ? QString("任务已停止 · 本次完成 %1 / %2 张 · 已完成结果可回看与导出")
                                         .arg(completed_).arg(taskRangeFiles_.size())
                                   : QString("任务已停止 · 已处理 %1 帧 · 当前结果可导出").arg(completed_))
                             : QString("检测完成 · 已处理 %1 %2")
                                   .arg(completed_)
                                   .arg(sourceKind_ == vision::SourceKind::Images ? "张图片" : "帧"));
    if (!smokeDir_.isEmpty())
    {
        const bool success = !failed_ && !lastResult_.image.isNull() && !lastResult_.predictions.isEmpty();
        QString error = lastError_;
        bool exported = success && writeResult(lastResult_, smokeDir_, &error);
        QTimer::singleShot(
            100, this,
            [this, success, exported, error]
            {
                saveScreenshot(smokeDir_ + "/workbench.png");
                const bool hasResult = !lastResult_.image.isNull();
                QJsonObject j{
                    {"success", success && exported},
                    {"qt", qVersion()},
                    {"predictions", lastResult_.predictions.size()},
                    {"inference_ms", lastResult_.inferenceMs},
                    {"error", error},
                    {"backend", hasResult ? lastResult_.backend : QString()},
                    {"requested_device",
                     vision::computeDeviceKey(hasResult ? lastResult_.requestedDevice : lastConfig_.device)},
                    {"actual_device", hasResult ? QJsonValue(vision::computeDeviceKey(lastResult_.device))
                                                : QJsonValue(QJsonValue::Null)},
                    {"device_index", hasResult ? lastResult_.deviceIndex : -1},
                    {"device_name", hasResult ? lastResult_.deviceName : QString()},
                    {"device_notice", hasResult ? lastResult_.deviceNotice : QString()}};
                atomicWrite(smokeDir_ + "/smoke-report.json", QJsonDocument(j).toJson());
                qInfo().noquote() << QJsonDocument(j).toJson(QJsonDocument::Compact);
                QApplication::exit(success && exported ? 0 : 2);
            });
    }
    if (closing_)
        QTimer::singleShot(0, this, &QWidget::close);
}

void MainWindow::runDemo()
{
    if (busy_)
        return;
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", projectRoot_);
    const QString model = fixtures + "/models/yolov5n.onnx", image = fixtures + "/assets/bus.jpg";
    if (!QFileInfo::exists(model) || !QFileInfo::exists(image))
    {
        showNotice("示例资源尚未准备好。请导入自己的 ONNX 模型与图片。", true);
        if (!smokeDir_.isEmpty())
            QApplication::exit(2);
        return;
    }
    setModel(model);
    taskBox_->setCurrentIndex(0);
    inputSize_->setValue(640);
    confidence_->setValue(0.25);
    iou_->setValue(0.45);
    scale_->setValue(1.0 / 255.0);
    meanR_->setValue(0);
    meanG_->setValue(0);
    meanB_->setValue(0);
    inputColorMode_->setCurrentIndex(0);
    labels_.clear();
    labelsPath_.clear();
    labelButton_->setText("类别标签 · 默认 COCO 80");
    clearImageResults();
    files_.clear();
    queue_->clear();
    sourceKind_ = vision::SourceKind::Images;
    addFiles({image});
    selectRoute(ui::Route::Workbench);
    startInference();
}
void MainWindow::runSmoke(const QString &dir)
{
    smokeDir_ = QFileInfo(dir).absoluteFilePath();
    QDir().mkpath(smokeDir_);
    runDemo();
}
void MainWindow::runPtDemo()
{
    if (busy_)
        return;
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", projectRoot_);
    const QString model = fixtures + "/models/yolov8n.pt", image = fixtures + "/assets/bus.jpg";
    if (!QFileInfo::exists(model) || !QFileInfo::exists(image))
    {
        showNotice("PT 示例资源不存在，请导入自己的 .pt 模型。", true);
        if (!smokeDir_.isEmpty())
            QApplication::exit(2);
        return;
    }
    labels_.clear();
    labelsPath_.clear();
    setModel(model);
    taskBox_->setCurrentIndex(1);
    inputSize_->setValue(640);
    confidence_->setValue(.25);
    iou_->setValue(.45);
    clearImageResults();
    files_.clear();
    queue_->clear();
    sourceKind_ = vision::SourceKind::Images;
    addFiles({image});
    selectRoute(ui::Route::Workbench);
    updateTaskUi();
    startInference();
}
void MainWindow::runPtSmoke(const QString &dir)
{
    smokeDir_ = QFileInfo(dir).absoluteFilePath();
    QDir().mkpath(smokeDir_);
    runPtDemo();
}
void MainWindow::showModelStructure(const QString &model)
{
    if (!model.isEmpty())
        setModel(model);
    selectRoute(ui::Route::ModelDisplay);
}

void MainWindow::runModelSmoke(const QString &dir, const QString &model)
{
    const QString output = QFileInfo(dir).absoluteFilePath();
    if (!QDir().mkpath(output)) { QApplication::exit(2); return; }
    const auto completed = std::make_shared<bool>(false);
    const auto capturing = std::make_shared<bool>(false);
    const auto views = std::make_shared<QJsonObject>();
    const auto yaml = std::make_shared<QJsonObject>();
    const auto cachePreserved = std::make_shared<bool>(false);
    const auto fileDigest = [](const QString &path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QString();
        QCryptographicHash hash(QCryptographicHash::Sha256);
        return hash.addData(&file) ? QString::fromLatin1(hash.result().toHex()) : QString();
    };
    const QString sourceHash = fileDigest(model);
    const auto writeReport = [this, output, completed, views, yaml, cachePreserved](bool success, const QString &error) {
        if (*completed) return;
        *completed = true;
        if (!success) saveScreenshot(output + "/model-display.png");
        const QJsonObject report{{"success", success}, {"error", error}, {"application", "Vision Studio"},
            {"version", QCoreApplication::applicationVersion()}, {"qt", qVersion()}, {"netron", "9.3.1"},
            {"model", modelViewer_->modelPath()}, {"nodes", modelViewer_->graphNodeCount()},
            {"hierarchy_items", modelViewer_->hierarchyItemCount()}, {"parameter_rows", modelViewer_->parameterCount()},
            {"display_modes", *views}, {"yaml", *yaml}, {"cache_preserved", *cachePreserved},
            {"onnx_hint_visible", structureHint_->isVisible()}, {"onnx_hint", structureHint_->text()}};
        QString errorText;
        const bool saved = atomicWrite(output + "/model-display-report.json", QJsonDocument(report).toJson(), &errorText);
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        QApplication::exit(success && saved ? 0 : 2);
    };
    const auto captureViews = [this, output, capturing, views, yaml, cachePreserved, writeReport, fileDigest, sourceHash] {
        if (*capturing) return;
        *capturing = true;
        if (modelViewer_->state() != ModelViewer::State::Ready || modelViewer_->graphNodeCount() <= 0) {
            writeReport(false, "模型没有生成可显示的结构。"); return;
        }
        const QPointer<QObject> browser = modelViewer_->findChild<QObject *>("netronWebView");
        const QPointer<QProcess> service = modelViewer_->findChild<QProcess *>();
        const qint64 processId = service ? service->processId() : 0;
        const QString path = modelViewer_->modelPath();
        const int items = modelViewer_->hierarchyItemCount(), parameters = modelViewer_->parameterCount();
        const QString text = modelViewer_->yamlText();
        const bool available = modelViewer_->yamlAvailable();
        const auto *yamlModeButton = modelViewer_->findChild<QPushButton *>("modelYamlModeButton");
        const bool yamlModeVisible = yamlModeButton && yamlModeButton->isVisible();
        QString exportError;
        const QString destination = output + "/model-architecture.yaml";
        const bool exported = available && modelViewer_->saveYaml(destination, &exportError);
        QString refusedError;
        const bool existingRefused = exported && !modelViewer_->saveYaml(destination, &refusedError) && !refusedError.isEmpty();
        *yaml = QJsonObject{{"available", available}, {"mode_visible", yamlModeVisible}, {"reason", modelViewer_->yamlReason()},
            {"bytes", text.toUtf8().size()}, {"sha256", QString::fromLatin1(QCryptographicHash::hash(text.toUtf8(), QCryptographicHash::Sha256).toHex())},
            {"exported", exported}, {"export_error", exportError}, {"existing_export_refused", existingRefused},
            {"export_sha256", exported ? fileDigest(destination) : QString()}, {"source_sha256_before", sourceHash}};
        auto *timer = new QTimer(this);
        auto step = std::make_shared<int>(-1);
        QList<ModelViewer::DisplayMode> modes{ModelViewer::DisplayMode::Graph, ModelViewer::DisplayMode::Hierarchy,
            ModelViewer::DisplayMode::Parameters};
        QStringList names{"graph", "hierarchy", "parameters"};
        QStringList screenshots{"model-display.png", "model-hierarchy.png", "model-parameters.png"};
        if (available) {
            modes.append(ModelViewer::DisplayMode::YoloYaml);
            names.append("yolo_yaml");
            screenshots.append("model-yaml.png");
        }
        connect(timer, &QTimer::timeout, this, [this, output, views, yaml, cachePreserved, writeReport, fileDigest,
            sourceHash, timer, step, modes, names, screenshots, browser, service, processId, path, items, parameters, text, available] {
            if (*step >= 0) {
                saveScreenshot(output + '/' + screenshots[*step]);
                views->insert(names[*step], modelViewer_->displayMode() == modes[*step]);
            }
            if (++*step < modes.size()) { modelViewer_->setDisplayMode(modes[*step]); return; }
            timer->stop(); timer->deleteLater();
            modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Graph);
            *cachePreserved = browser && service && processId > 0 &&
                modelViewer_->findChild<QObject *>("netronWebView") == browser &&
                modelViewer_->findChild<QProcess *>() == service && service->processId() == processId &&
                modelViewer_->state() == ModelViewer::State::Ready && modelViewer_->modelPath() == path &&
                modelViewer_->hierarchyItemCount() == items && modelViewer_->parameterCount() == parameters &&
                modelViewer_->yamlAvailable() == available && modelViewer_->yamlText() == text;
            yaml->insert("source_sha256_after", fileDigest(path));
            bool success = *cachePreserved && items > 0 && !sourceHash.isEmpty() && fileDigest(path) == sourceHash &&
                yaml->value("mode_visible").toBool() == available;
            for (const auto &name : names) success = success && views->value(name).toBool();
            success = success && (available ? yaml->value("exported").toBool() && yaml->value("existing_export_refused").toBool() &&
                yaml->value("export_sha256") == yaml->value("sha256") : !modelViewer_->yamlReason().isEmpty());
            writeReport(success, success ? QString() : "模型视图、架构另存或解析缓存自检失败。");
        });
        timer->start(300);
    };
    connect(modelViewer_, &ModelViewer::modelLoaded, this, [this, captureViews](const QString &) {
        QTimer::singleShot(300, this, captureViews); }, Qt::SingleShotConnection);
    connect(modelViewer_, &ModelViewer::loadFailed, this, [writeReport](const QString &error) {
        writeReport(false, error); }, Qt::SingleShotConnection);
    showModelStructure(model);
    if (modelViewer_->state() == ModelViewer::State::Ready) QTimer::singleShot(300, this, captureViews);
    else if (modelViewer_->state() == ModelViewer::State::Error)
        QTimer::singleShot(0, this, [this, writeReport] { writeReport(false, modelViewer_->errorString()); });
}
void MainWindow::saveScreenshot(const QString &p)
{
    QDir().mkpath(QFileInfo(p).absolutePath());
    if (!grab().save(p))
        qWarning().noquote() << "Cannot save screenshot" << p;
}

MainWindow::ResultExport MainWindow::prepareResultExport(const vision::InferenceResult &result,
    const QString &directory, const vision::ModelConfig &config)
{
    const auto historical = result.evaluationReport == evaluationDirectory_
        ? evaluationReport_.value("config").toObject() : QJsonObject{};
    return vision::app::ResultWriter::prepare(result, directory, config, historical);
}

bool MainWindow::writePreparedResult(const ResultExport &result, QString *error)
{
    return vision::app::ResultWriter::writePrepared(result, error);
}

bool MainWindow::writeResult(const vision::InferenceResult &result, const QString &directory, QString *error)
{
    return writePreparedResult(prepareResultExport(result, directory, lastConfig_), error);
}

void MainWindow::exportResult()
{
    if (lastResult_.image.isNull() || lastResult_.demonstration)
        return;
    const QString dir = QFileDialog::getExistingDirectory(this, "选择结果导出目录",
                                                          QDir(exportDir_).exists() ? exportDir_ : dataRoot_);
    if (dir.isEmpty())
        return;
    QString error;
    if (writeResult(lastResult_, dir, &error))
    {
        exportDir_ = dir;
        persist();
        showNotice("已导出标注 PNG、JSON 与 CSV 到：" + dir);
    }
    else
        showNotice("导出失败：" + error, true);
}
void MainWindow::recordResult(const vision::InferenceResult &result, bool writeToDisk)
{
    vision::app::HistoryRepository::prepend(history_, result);
    if (writeToDisk) {
        QString error;
        if (!vision::app::HistoryRepository(dataRoot_ + "/history.json").save(history_, &error))
            showNotice("运行记录保存失败：" + error, true);
    }
    if (!busy_ || (pages_ && pages_->currentIndex() == ui::routeIndex(ui::Route::History))) refreshHistory();
}

void MainWindow::refreshHistory()
{
    if (!historyTable_)
        return;
    historyTable_->setRowCount(history_.size());
    for (int i = 0; i < history_.size(); ++i)
    {
        auto o = history_[i].toObject();
        const QStringList values = {
            QDateTime::fromString(o["time"].toString(), Qt::ISODateWithMs).toString("MM-dd  HH:mm:ss"),
            QFileInfo(o["source"].toString()).fileName() +
                ((o["stereo_view"].toString() == "left" || o["stereo_view"].toString() == "right")
                     ? " · " + stereoName(stereoMode(o["stereo_view"].toString()))
                     : QString()),
            o["model"].toString() +
                (o.contains("actual_device")
                     ? " · " + (o["actual_device"].toString() == "cuda" ? QStringLiteral("GPU")
                                                                        : QStringLiteral("CPU"))
                     : QString()),
            QString::number(o["objects"].toInt()),
            QString::number(o["inference_ms"].toDouble(), 'f', 1) + " ms",
            o["task"].toString()};
        for (int c = 0; c < values.size(); ++c)
        {
            auto *item = new QTableWidgetItem(values[c]);
            item->setToolTip(c == 1   ? o["source"].toString()
                             : c == 2 ? values[c] + "\n" + o["backend"].toString() + "\n" +
                                            o["device_name"].toString() + "\n" + o["device_notice"].toString()
                                      : values[c]);
            historyTable_->setItem(i, c, item);
        }
    }
}
QString MainWindow::modelDisplayName(const QString &path) const
{
    const QString name = modelDisplayNames_.value(path).toString().trimmed();
    return name.isEmpty() ? QFileInfo(path).fileName() : name;
}
void MainWindow::showModelContextMenu(const QPoint &position)
{
    auto *item = modelList_->itemAt(position);
    if (!item && position.x() < 0 && position.y() < 0)
        item = modelList_->currentItem();
    if (!item)
        return;
    const QString path = item->data(Qt::UserRole).toString();
    auto *menu = new QMenu(modelList_);
    menu->setObjectName("modelContextMenu");
    menu->setAttribute(Qt::WA_DeleteOnClose);
    auto addAction = [this, menu, path](const QString &title, const QString &icon,
                                      const char *name, auto handler, bool mutation)
    {
        auto *action = menu->addAction(ui::icon(icon, QColor("#D2D2D2")), title);
        action->setObjectName(name);
        action->setData(path);
        action->setEnabled(!mutation || !busy_);
        connect(action, &QAction::triggered, this, [this, path, handler] { (this->*handler)(path); });
        return action;
    };
    addAction("打开模型位置", "folder", "modelOpenLocationAction", &MainWindow::openModelLocation,
              false);
    addAction(modelNotes_.value(path).toString().isEmpty() ? "添加备注" : "修改备注", "note",
              "modelNoteAction", &MainWindow::editModelNote, true);
    addAction("重命名", "edit", "modelRenameAction", &MainWindow::renameLibraryModel, true);
    addAction("从列表移除", "cross", "modelRemoveAction", &MainWindow::removeLibraryModel, true);
    const QPoint anchor = position.x() < 0 && position.y() < 0
                              ? modelList_->visualItemRect(item).center()
                              : position;
    menu->popup(modelList_->viewport()->mapToGlobal(anchor));
}
void MainWindow::openModelLocation(const QString &path)
{
    const QString directory = QFileInfo(path).absolutePath();
    if (!QFileInfo(directory).isDir())
    {
        showNotice("模型所在文件夹不存在：" + directory, true);
        return;
    }
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(directory)))
        showNotice("无法打开模型文件夹：" + directory, true);
}
void MainWindow::editModelNote(const QString &path)
{
    if (busy_ || !models_.contains(path))
        return;
    QInputDialog dialog(this);
    dialog.setObjectName("modelNoteDialog");
    dialog.setWindowTitle("模型备注");
    dialog.setInputMode(QInputDialog::TextInput);
    dialog.setOption(QInputDialog::UsePlainTextEditForTextInput);
    dialog.setLabelText(modelDisplayName(path) + "\n添加实验说明、数据集或用途；留空可清除备注。");
    for (auto *label : dialog.findChildren<QLabel *>())
        label->setTextFormat(Qt::PlainText);
    dialog.setTextValue(modelNotes_.value(path).toString());
    dialog.setOkButtonText("保存");
    dialog.setCancelButtonText("取消");
    dialog.resize(520, 260);
    if (dialog.exec() != QDialog::Accepted || busy_ || !models_.contains(path))
        return;
    const QString note = dialog.textValue().trimmed();
    if (note.isEmpty())
        modelNotes_.remove(path);
    else
        modelNotes_.insert(path, note);
    persist();
    refreshModelLibrary();
    showNotice(note.isEmpty() ? "模型备注已清除。" : "模型备注已保存。");
}
void MainWindow::renameLibraryModel(const QString &path)
{
    if (busy_ || !models_.contains(path))
        return;
    QInputDialog dialog(this);
    dialog.setObjectName("modelRenameDialog");
    dialog.setWindowTitle("重命名模型");
    dialog.setInputMode(QInputDialog::TextInput);
    dialog.setLabelText("模型库中的显示名称（原始文件名和路径不变）\n留空可恢复原始文件名：" +
                        QFileInfo(path).fileName());
    for (auto *label : dialog.findChildren<QLabel *>())
        label->setTextFormat(Qt::PlainText);
    dialog.setTextValue(modelDisplayNames_.value(path).toString());
    dialog.setOkButtonText("保存");
    dialog.setCancelButtonText("取消");
    dialog.resize(520, 180);
    if (dialog.exec() != QDialog::Accepted || busy_ || !models_.contains(path))
        return;
    const QString name = dialog.textValue().simplified();
    if (name.isEmpty())
        modelDisplayNames_.remove(path);
    else
        modelDisplayNames_.insert(path, name);
    persist();
    refreshModelLibrary();
    if (path == modelPath_)
    {
        modelName_->setText(modelDisplayName(path));
        if (modelConversion_) modelConversion_->setSourceModel(path, modelDisplayName(path));
        displayModelStructure();
    }
    showNotice(name.isEmpty() ? "已恢复模型的原始文件名显示。" : "模型显示名称已保存。");
}
void MainWindow::removeLibraryModel(const QString &path)
{
    if (busy_ || !models_.contains(path))
        return;
    models_.removeAll(path);
    refreshModelLibrary();
    if (path == modelPath_)
    {
        QString next;
        for (const QString &candidate : models_)
            if (QFileInfo(candidate).isFile())
            {
                next = candidate;
                break;
            }
        setModel(next);
    }
    persist();
    showNotice("已从模型库移除，原始模型文件保留。");
}
void MainWindow::refreshModelLibrary()
{
    if (!modelList_)
        return;
    const QSignalBlocker blocker(modelList_);
    modelList_->clear();
    for (const QString &p : models_)
    {
        QFileInfo f(p);
        const QString detail =
            f.exists() ? QString("%1 MB  ·  %2").arg(f.size() / 1048576.0, 0, 'f', 1).arg(modelFormat(p))
                       : "文件已移动或不存在";
        const QString note = modelNotes_.value(p).toString();
        QString caption = modelDisplayName(p) + "\n" + detail + "\n" + p;
        if (!note.isEmpty())
        {
            const QString compactNote = note.simplified();
            caption += "\n备注 · " + compactNote.left(140) +
                       (compactNote.size() > 140 ? QStringLiteral("…") : QString());
        }
        const auto profile = modelEvaluationProfiles_.value(p).toMap();
        if (!profile.isEmpty()) caption += profile.value("model_identity").toString() == fileIdentity(p)
            ? QString("\n评估 · %1 · %2%3").arg(profile.value("split").toString()).arg(profile.value("full_split").toBool() ? "完整集合" : "子集/未完成")
                .arg(profile.value("adopted").toBool() ? QString(" · 已采用阈值 %1").arg(profile.value("adopted_threshold").toDouble(), 0, 'f', 2) : QString())
            : QString("\n模型文件已变更 · 原评估仅供历史回看");
        auto *i = new QListWidgetItem(ui::icon("model", QColor("#60CDFF"), 34), caption);
        i->setData(Qt::UserRole, p);
        i->setData(Qt::UserRole + 1, modelDisplayName(p));
        i->setData(Qt::UserRole + 2, note);
        i->setToolTip(modelDisplayName(p) + "\n原始文件：" + f.fileName() + "\n" + p +
                       (note.isEmpty() ? QString() : "\n备注：" + note));
        i->setSizeHint(QSize(100, (note.isEmpty() ? 94 : 118) + (profile.isEmpty() ? 0 : 25)));
        modelList_->addItem(i);
        if (p == modelPath_)
            modelList_->setCurrentItem(i);
    }
    modelCount_->setText(QString("本地模型库 · %1 个模型").arg(models_.size()));
}
void MainWindow::selectRoute(int index)
{
    if (!pages_ || index < 0 || index >= pages_->count())
        return;
    if (evaluation_) evaluation_->hideDatasetSettings();
    pages_->setCurrentIndex(index);
    if (index != 5)
        stopRecordingPlayback();
    else
        refreshRecordings();
    if (index == 2)
        displayModelStructure();
    for (int i = 0; i < navButtons_.size(); ++i)
    {
        navButtons_[i]->setChecked(i == index);
        navButtons_[i]->setIcon(
            ui::icon(QStringList{"work", "model", "graph", "history", "help", "video", "convert", "more"}[i],
                     QColor(i == index ? "#60CDFF" : "#D2D2D2")));
    }
    const QStringList titles = {"检测工作台", "模型库",   "模型显示", "运行记录",
                                "使用指南",   "录制视频", "模型转换", "更多"};
    const QStringList descriptions = {
        "从输入到洞察，让每一次视觉推理清晰可见。",   "管理本地模型，让每一个实验都有清晰的起点。",
        "查看网络结构、输入输出与层参数。",           "回看每一次推理，沉淀可追溯的运行数据。",
        "从模型配置到结果导出，掌握完整的工作流程。", "查看、播放和导出已保存的检测录像。",
        "将当前模型导出为 ONNX 或 TorchScript，保留原始文件。",
        "管理启动偏好、GPU 环境与检测结果导出。"};
    pageTitle_->setText(titles[index]);
    pageSubtitle_->setText(descriptions[index]);
    workspaceHeader_->setVisible(index != 0 || !workbenchTabs_ || workbenchTabs_->currentIndex() != 1);
}
void MainWindow::showNotice(const QString &s, bool error)
{
    if (!statusLabel_)
        return;
    const QString message = "●  " + s.simplified();
    statusLabel_->setText(statusLabel_->fontMetrics().elidedText(message, Qt::ElideRight,
                                                                 std::max(200, statusLabel_->width())));
    statusLabel_->setToolTip(s);
    statusLabel_->setStyleSheet(error ? "color:#FFB4BB;" : "color:#B8B8B8;");
}
void MainWindow::persist()
{
    if (!settings_)
        return;
    settings_->setValue("models", models_);
    settings_->setValue("modelDisplayNames", modelDisplayNames_);
    settings_->setValue("modelNotes", modelNotes_);
    settings_->setValue("convertedModelProfiles", convertedModelProfiles_);
    settings_->setValue("evaluationProfiles", modelEvaluationProfiles_);
    settings_->setValue("activeModel", modelPath_);
    settings_->setValue("exportDirectory", exportDir_);
    settings_->setValue("labels", labels_);
    settings_->setValue("labelsPath", labelsPath_);
    if (taskBox_)
    {
        settings_->setValue("task", taskBox_->currentIndex());
        settings_->setValue("inputSize", inputSize_->value());
        settings_->setValue("confidence", confidence_->value());
        settings_->setValue("iou", iou_->value());
        settings_->setValue("scale", scale_->value());
        settings_->setValue("meanR", meanR_->value());
        settings_->setValue("meanG", meanG_->value());
        settings_->setValue("meanB", meanB_->value());
        settings_->setValue("colorMode", inputColorMode_->currentData());
        settings_->setValue("stereoView", stereoView_->currentData());
        settings_->setValue("cameraIndex", cameraIndex_->value());
        settings_->setValue("computeDevice", computeDevice_->currentData());
        settings_->setValue("deviceIndex", gpuDeviceIndex_->currentData());
        settings_->setValue("swapRB", inputColorMode_->currentData().toString() != "bgr");
        settings_->setValue("autoExport", autoExport_->isChecked());
        settings_->setValue("classFilter", classFilter_->text());
    }
    settings_->sync();
}
void MainWindow::closeEvent(QCloseEvent *e)
{
    cancelPendingDatasetSessionRestore();
    if (evaluationPreviewCancellation_) evaluationPreviewCancellation_->store(true);
    if (evaluationFilterCancellation_) evaluationFilterCancellation_->store(true);
    if (modelInspectionCancellation_) modelInspectionCancellation_->store(true);
    if (modelConversion_ && modelConversion_->isBusy())
    {
        closing_ = true;
        modelConversion_->cancel();
        e->ignore();
        return;
    }
    if (gpuService_->isPreparing())
    {
        closing_ = true;
        cancelGpuPreparation();
        e->ignore();
        return;
    }
    if (busy_ || resultIoPending_)
    {
        closing_ = true;
        if (busy_)
            stopInference();
        else
            showNotice("正在保存已完成的结果，保存完成后关闭窗口…");
        e->ignore();
        return;
    }
    persist();
    e->accept();
}
void MainWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
    {
        if (modelConversion_ && modelConversion_->isBusy() && pages_->currentIndex() == ui::routeIndex(ui::Route::ModelConversion))
        {
            modelConversion_->cancel();
            e->accept();
            return;
        }
        stopInference();
        e->accept();
        return;
    }
    if (e->modifiers() & Qt::ControlModifier)
    {
        if (e->key() == Qt::Key_O && !busy_)
        {
            evaluation_->chooseDataset();
            return;
        }
        if (e->key() == Qt::Key_M && !busy_)
        {
            importModel();
            return;
        }
        if (e->key() == Qt::Key_R)
        {
            startWorkbenchTask();
            return;
        }
        if (e->key() == Qt::Key_E && !busy_)
        {
            exportResult();
            return;
        }
    }
    QMainWindow::keyPressEvent(e);
}
