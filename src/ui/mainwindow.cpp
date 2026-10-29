#include "runtimesetup.h"
#include "app/historyrepository.h"
#include "app/gpuenvironmentservice.h"
#include "app/recordingcontroller.h"
#include "pages/modelslibrarypage.h"
#include "pages/modeldisplaypage.h"
#include "pages/historypage.h"
#include "pages/recordingspage.h"
#include "pages/morepage.h"
#include "pages/guidepage.h"
#include "core/cpuruntime.h"
#include "mainwindow.h"
#include "smoke/smokedrivers.h"
#include "core/previewdrop.h"
#include "pages/workspacedropfilter.h"
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
#include "pages/workbenchpage.h"
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
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHeaderView>
#include <QImageWriter>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMouseEvent>
#include <QMessageBox>
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

namespace
{
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
QString stereoName(vision::StereoView view)
{
    return view == vision::StereoView::Left ? "双目左目"
                                            : (view == vision::StereoView::Right ? "双目右目" : "完整画面");
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
QString fileIdentity(const QString &path)
{
    const QFileInfo file(path);
    return file.isFile() ? file.absoluteFilePath() + "|" + QString::number(file.size()) + "|" +
                              QString::number(file.lastModified().toMSecsSinceEpoch()) : QString();
}
template <typename Editor, typename Value> auto normalizedEditorValue(Editor *editor, Value requested)
{
    // Only an explicitly accepted setting/profile field passes this boundary;
    // its own Qt bounds and precision retain the established runtime value.
    const QSignalBlocker blocker(editor);
    editor->setValue(requested);
    return editor->value();
}
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    operationCoordinator_ = new vision::app::OperationCoordinator(this);
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
    modelSession_ = new vision::app::ModelSessionController(this);
    inputSession_ = new vision::app::InputSessionController(projectRoot_, datasetSessionFilePath(), this);
    configuration_ = new vision::app::WorkbenchConfiguration(this);
    configuration_->setExportDirectory(settings_->value("exportDirectory", dataRoot_ + "/results").toString());
    vision::app::ModelSessionSnapshot savedModels;
    savedModels.models = settings_->value("models").toStringList();
    const bool hasSavedModelLibrary = settings_->contains("models");
    savedModels.displayNames = settings_->value("modelDisplayNames").toMap();
    savedModels.notes = settings_->value("modelNotes").toMap();
    savedModels.convertedProfiles = settings_->value("convertedModelProfiles").toMap();
    savedModels.evaluationProfiles = settings_->value("evaluationProfiles").toMap();
    modelSession_->initialize(savedModels);
    historyRepository_ = std::make_unique<vision::app::HistoryRepository>(dataRoot_ + "/history.json");
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
        if (isWorkbenchBusy()) {
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
        if (!active && isClosing()) QTimer::singleShot(0, this, &QWidget::close);
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
    connectConfiguration();
    gpuService_ = new vision::app::GpuEnvironmentService(projectRoot_, this);
    connect(gpuService_, &vision::app::GpuEnvironmentService::environmentChanged, this,
        [this](const QJsonObject &environment) {
        const int selected = configurationState().parameters.deviceIndex;
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
        if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
    });
    connectWorker();
    detection_->results()->setHistoryRepository(historyRepository_.get());
    connect(detection_->results(), &vision::app::ResultDeliveryService::pendingChanged, this, [this](int count) {
        if (!isWorkbenchBusy() && !count) {
            updateStartEligibility();
            if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
        }
    });
    connect(detection_->results(), &vision::app::ResultDeliveryService::notice, this, &MainWindow::showNotice);
    connect(detection_->results(), &vision::app::ResultDeliveryService::historyChanged, this, [this] {
        if (!isWorkbenchBusy() || (pages_ && pages_->currentIndex() == ui::routeIndex(ui::Route::History))) refreshHistory();
    });
    connectInputSession();
    connectEvaluation();
    initializeOperationCoordinator();
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
        (!hasSavedModelLibrary || modelState().models.contains(saved)))
        setModel(saved);
    else
    {
        QString next;
        for (const QString &candidate : modelState().models)
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
            if (QFileInfo(example).isFile() && !modelState().models.contains(example))
                modelSession_->addKnownModel(example);
        }
    }
    auto restored = configurationState();
    restored.parameters.task = static_cast<vision::ModelTask>(qBound(0, savedTask, 2));
    restored.parameters.inputSize = normalizedEditorValue(inputSize_, savedInput);
    restored.confidence = normalizedEditorValue(confidence_, savedConfidence);
    restored.iou = normalizedEditorValue(iou_, savedIou);
    restored.parameters.scale = normalizedEditorValue(scale_, savedScale);
    restored.parameters.meanR = normalizedEditorValue(meanR_, savedR);
    restored.parameters.meanG = normalizedEditorValue(meanG_, savedG);
    restored.parameters.meanB = normalizedEditorValue(meanB_, savedB);
    restored.parameters.colorMode = savedColorMode == "grayscale" ? vision::InputColorMode::Grayscale : vision::InputColorMode::Color;
    restored.parameters.swapRB = savedColorMode != "bgr";
    restored.stereoView = savedStereo == "left" ? vision::StereoView::Left : savedStereo == "right" ? vision::StereoView::Right : vision::StereoView::Full;
    restored.cameraIndex = normalizedEditorValue(cameraIndex_, savedCameraIndex);
    restored.parameters.device = computeMode(savedDevice);
    restored.parameters.deviceIndex = savedDeviceIndex;
    restored.automaticExport = savedAuto;
    restored.classFilterText = settings_->value("classFilter").toString().left(classFilter_->maxLength());
    restored.parameters.labels = savedLabels; restored.labelsPath = savedLabelsPath;
    configuration_->replaceDraft(restored);
    if (!savedLabels.isEmpty())
    {
        labelButton_->setText(QString("类别标签 · %1 个").arg(savedLabels.size()));
        labelButton_->setToolTip(savedLabelsPath);
    }
    applyConvertedModelProfile(modelState().activeModel);
    canvas_->clear();
    updateSourceUi();
    refreshModelLibrary();
    refreshHistory();
    refreshRecordings();
    selectRoute(ui::Route::Workbench);
    updateTaskUi();
    updateDeviceUi();
    connect(modelSession_, &vision::app::ModelSessionController::metadataChanged, this, [this] {
        const auto state = modelState();
        if (!isWorkbenchBusy() && !state.inspectionPending && state.inspectionError.isEmpty() &&
            !state.inspectedIdentity.isEmpty() && state.inspectedIdentity == modelFileIdentity()) {
            if (isPtModel(state.activeModel) || (state.hasSuggestedTask && state.suggestedTask == vision::ModelTask::Classification)) {
                const int previous = taskBox_->currentIndex();
                configuration_->setTask(state.suggestedTask);
                if (previous != taskBox_->currentIndex()) invalidateConfiguredResults();
            }
            updateModelMeta("已识别");
            updateTaskUi();
        }
        updateClassFilterUi();
    });
    connect(modelSession_, &vision::app::ModelSessionController::inspectionRetryRequested, this, [this] {
        if (!isClosing()) { invalidateConfiguredResults(); inspectSelectedModel(); }
    });
    connect(modelSession_, &vision::app::ModelSessionController::shutdownReady, this, [this] {
        if (isClosing()) QTimer::singleShot(0, this, &QWidget::close);
    });
    initializing_ = false;
    projectConfigurationToView();
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
                 if (!isWorkbenchBusy())
                     evaluation_->chooseDataset();
             });
    shortcut(QKeySequence("Ctrl+M"),
             [this]
             {
                 if (!isWorkbenchBusy())
                     importModel();
             });
    shortcut(QKeySequence("Ctrl+R"), [this] { startWorkbenchTask(); });
    shortcut(QKeySequence("Ctrl+E"),
             [this]
             {
                 if (!isWorkbenchBusy())
                     exportResult();
             });
    shortcut(QKeySequence(Qt::Key_Escape),
             [this]
             {
                 if (modelConversion_ && modelConversion_->isBusy() && pages_->currentIndex() == ui::routeIndex(ui::Route::ModelConversion))
                     modelConversion_->cancel();
                 else if (isWorkbenchBusy())
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
    // Normal destruction follows OperationCoordinator's asynchronous barrier.
    // Owners retain their own defensive nonblocking teardown for early destruction.
    if (!isShutdownReady()) qWarning("Window destroyed before shutdown barrier completed");
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
        b->setObjectName(QString("navigation-%1").arg(i));
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
    auto *page = workbenchPage_ = new WorkbenchPage(exportDirectory());
    const auto &controls = page->controls();
    autoExport_ = controls.autoExport;
    backendBadge_ = controls.backendBadge;
    cameraIndex_ = controls.cameraIndex;
    canvasTitle_ = controls.canvasTitle;
    canvas_ = controls.canvas;
    classFilterHint_ = controls.classFilterHint;
    classFilter_ = controls.classFilter;
    classMetric_ = controls.classMetric;
    computeDevice_ = controls.computeDevice;
    confidence_ = controls.confidence;
    countMetric_ = controls.countMetric;
    detectionRangePanel_ = controls.detectionRangePanel;
    detectionRange_ = controls.detectionRange;
    deviceHint_ = controls.deviceHint;
    emptyResults_ = controls.emptyResults;
    evaluationPreviewFilter_ = controls.evaluationPreviewFilter;
    evaluationRangeHint_ = controls.evaluationRangeHint;
    evaluationRange_ = controls.evaluationRange;
    evaluationReviewBar_ = controls.evaluationReviewBar;
    evaluationReviewButton_ = controls.evaluationReviewButton;
    evaluationReviewNextButton_ = controls.evaluationReviewNextButton;
    evaluationReviewPreviousButton_ = controls.evaluationReviewPreviousButton;
    evaluationReviewSummaryLabel_ = controls.evaluationReviewSummaryLabel;
    evaluationStartButton_ = controls.evaluationStartButton;
    evaluationStopButton_ = controls.evaluationStopButton;
    evaluationTaskSummary_ = controls.evaluationTaskSummary;
    evaluation_ = controls.evaluation;
    gpuDeviceIndex_ = controls.gpuDeviceIndex;
    gpuDeviceLabel_ = controls.gpuDeviceLabel;
    inputColorMode_ = controls.inputColorMode;
    inputSize_ = controls.inputSize;
    iou_ = controls.iou;
    labelButton_ = controls.labelButton;
    latencyMetric_ = controls.latencyMetric;
    meanB_ = controls.meanB;
    meanG_ = controls.meanG;
    meanRLabel_ = controls.meanRLabel;
    meanR_ = controls.meanR;
    modelButton_ = controls.modelButton;
    modelMeta_ = controls.modelMeta;
    modelName_ = controls.modelName;
    predictionTable_ = controls.predictionTable;
    preprocessHint_ = controls.preprocessHint;
    queue_ = controls.queue;
    rangeHint_ = controls.rangeHint;
    recordButton_ = controls.recordButton;
    recordingStatus_ = controls.recordingStatus;
    resultInfo_ = controls.resultInfo;
    runButton_ = controls.runButton;
    scale_ = controls.scale;
    showBoxes_ = controls.showBoxes;
    showGroundTruth_ = controls.showGroundTruth;
    showLabels_ = controls.showLabels;
    sizeMetric_ = controls.sizeMetric;
    sourceLabel_ = controls.sourceLabel;
    stereoLabel_ = controls.stereoLabel;
    stereoView_ = controls.stereoView;
    stopButton_ = controls.stopButton;
    taskBox_ = controls.taskBox;
    workbenchTabs_ = controls.workbenchTabs;
    zoomLabel_ = controls.zoomLabel;
    lockedControls_ += controls.lockedControls;
    connect(page, &WorkbenchPage::videoRequested, this, &MainWindow::chooseVideo);
    connect(page, &WorkbenchPage::cameraRequested, this, &MainWindow::chooseCamera);
    connect(page, &WorkbenchPage::importModelRequested, this, &MainWindow::importModel);
    connect(page, &WorkbenchPage::importLabelsRequested, this, &MainWindow::importLabels);
    connect(page, &WorkbenchPage::defaultLabelsRequested, this, [this] {
        configuration_->setLabels({}, {});
        invalidateConfiguredResults(); updateTaskUi(); persist();
        showNotice("已恢复模型默认类别。");
    });
    connect(page, &WorkbenchPage::deviceChanged, this, [this] {
        if (isWorkbenchBusy()) return;
        workbenchPage_->clearDeviceVerification(true);
        updateDeviceUi(); updateInputPreview(); persist();
    });
    connect(page, &WorkbenchPage::classFilterChanged, this, [this] {
        if (!initializing_ && !isWorkbenchBusy() && !modelState().inspectionPending &&
            modelState().inspectedIdentity != modelFileIdentity()) inspectSelectedModel();
        updateClassFilterUi();
        if (!initializing_ && !isWorkbenchBusy()) invalidateConfiguredResults();
        persist();
    });
    connect(page, &WorkbenchPage::parametersChanged, this, [this] {
        if (!initializing_ && !isWorkbenchBusy()) { invalidateConfiguredResults(); persist(); }
    });
    connect(page, &WorkbenchPage::taskChanged, this, [this] {
        updateTaskUi();
        if (!initializing_ && !isWorkbenchBusy()) { invalidateConfiguredResults(); inspectSelectedModel(); }
    });
    connect(page, &WorkbenchPage::inputColorChanged, this, [this] {
        updateTaskUi(); updateInputPreview(); persist();
    });
    connect(page, &WorkbenchPage::stereoChanged, this, [this] {
        if (!isWorkbenchBusy() && inputState().sourceKind != vision::SourceKind::Images) {
            workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0);
            emptyResults_->show(); exportButton_->setEnabled(false);
        }
        updateSourceUi(); persist();
    });
    connect(page, &WorkbenchPage::cameraIndexChanged, this, [this] { updateSourceUi(); persist(); });
    connect(page, &WorkbenchPage::rangeChanged, this, &MainWindow::updateRangeUi);
    connect(page, &WorkbenchPage::startRequested, this, &MainWindow::startWorkbenchTask);
    connect(page, &WorkbenchPage::stopRequested, this, &MainWindow::stopInference);
    connect(page, &WorkbenchPage::openExportDirectoryRequested, this, [this] {
        workbenchPage_->setExportDirectory(exportDirectory());
        if (!QDir().mkpath(exportDirectory())) showNotice("无法创建保存文件夹：" + exportDirectory(), true);
        else if (!QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(exportDirectory()).absoluteFilePath())))
            showNotice("无法打开保存文件夹：" + exportDirectory(), true);
    });
    connect(page, &WorkbenchPage::recordRequested, this, &MainWindow::toggleRecording);
    connect(page, &WorkbenchPage::filesDropped, this, [this](const QStringList &paths) { handlePreviewDrop(paths); });
    previewDropFilter_ = new WorkspaceDropFilter(WorkspaceDropFilter::Surface::Preview,
        page->findChild<QWidget *>("imagePreviewPage"), canvas_, nullptr, operationCoordinator_, modelSession_, page);
    connect(previewDropFilter_, &WorkspaceDropFilter::noticeRequested, this, &MainWindow::showNotice);
    connect(previewDropFilter_, &WorkspaceDropFilter::previewDropRequested, this,
        [this](quint64 id, const QStringList &paths) {
        previewDropFilter_->resolveDrop(id, handlePreviewDrop(paths));
    }, Qt::DirectConnection);
    connect(page, &WorkbenchPage::clearQueueRequested, this, [this] {
        cancelPendingDatasetSessionRestore(); clearDatasetContext(); clearImageResults(); inputSession_->clear();
        workbenchPage_->clearDisplayedResult(); canvas_->clear(); predictionTable_->setRowCount(0);
        emptyResults_->show(); exportButton_->setEnabled(false); updateSourceUi();
    });
    connect(page, &WorkbenchPage::queueRowChanged, this, &MainWindow::selectQueueImage);
    connect(page, &WorkbenchPage::queueSelectionChanged, this, [this](int row, bool selected) {
        inputSession_->setSelection(row, selected); updateRangeUi();
    });
    connect(page, &WorkbenchPage::reviewFilterRequested, this, &MainWindow::applyEvaluationFilter);
    connect(page, &WorkbenchPage::reviewNavigateRequested, this, &MainWindow::navigateEvaluationReview);
    connect(page, &WorkbenchPage::reviewErrorsRequested, this, &MainWindow::showEvaluationErrors);
    connect(page, &WorkbenchPage::workbenchTabChanged, this, [this](int) { updateSourceUi(); });
    return page;
}

QWidget *MainWindow::buildModels()
{
    auto *page = modelsLibraryPage_ = new ModelsLibraryPage;
    page->setApplicationContext(modelSession_, operationCoordinator_);
    const auto &controls = page->controls();
    modelCount_ = controls.count;
    modelList_ = controls.list;
    modelLibraryDropFilter_ = new WorkspaceDropFilter(WorkspaceDropFilter::Surface::ModelLibrary,
        page, nullptr, modelList_, operationCoordinator_, modelSession_, page);
    connect(modelLibraryDropFilter_, &WorkspaceDropFilter::modelHintChanged, page, &ModelsLibraryPage::setDropHint);
    connect(modelLibraryDropFilter_, &WorkspaceDropFilter::noticeRequested, this, &MainWindow::showNotice);
    connect(modelLibraryDropFilter_, &WorkspaceDropFilter::modelDropRequested, this,
        [this](quint64 id, const QStringList &paths) {
        modelLibraryDropFilter_->resolveDrop(id, importModelFiles(paths));
    }, Qt::DirectConnection);
    lockedControls_.append(controls.import);
    lockedControls_.append(controls.activate);
    lockedControls_.append(controls.remove);
    connect(page, &ModelsLibraryPage::importRequested, this, &MainWindow::importModel);
    connect(page, &ModelsLibraryPage::noticeRequested, this, &MainWindow::showNotice);
    connect(page, &ModelsLibraryPage::noteRequested, this, [this](const QString &path, const QString &note) {
        if (previewDropLocked() || !modelSession_->setNote(path, note)) return;
        persist(); refreshModelLibrary();
        showNotice(note.isEmpty() ? "模型备注已清除。" : "模型备注已保存。");
    });
    connect(page, &ModelsLibraryPage::renameRequested, this, [this](const QString &path, const QString &name) {
        if (previewDropLocked() || !modelSession_->rename(path, name)) return;
        persist(); refreshModelLibrary();
        if (path == modelState().activeModel) {
            modelName_->setText(modelDisplayName(path));
            if (modelConversion_) modelConversion_->setSourceModel(path, modelDisplayName(path));
            displayModelStructure();
        }
        showNotice(name.isEmpty() ? "已恢复模型的原始文件名显示。" : "模型显示名称已保存。");
    });
    connect(page, &ModelsLibraryPage::selectionRequested, this, [this](const QString &path) {
        if (!isWorkbenchBusy()) setModel(path);
    });
    connect(page, &ModelsLibraryPage::orderChanged, this, [this](const QStringList &paths) {
        QString error;
        if (!modelSession_->reorder(paths, &error)) {
            refreshModelLibrary();
            if (!error.isEmpty()) showNotice(error, true);
            return;
        }
        persist();
    });
    connect(page, &ModelsLibraryPage::detectRequested, this, [this] { selectRoute(ui::Route::Workbench); });
    connect(page, &ModelsLibraryPage::structureRequested, this, [this] { selectRoute(ui::Route::ModelDisplay); });
    connect(page, &ModelsLibraryPage::removeRequested, this, &MainWindow::removeLibraryModel);
    connect(page, &ModelsLibraryPage::evaluationRequested, this, [this] {
        const auto state = modelState();
        const auto profile = state.evaluationProfiles.value(state.activeModel).toMap();
        if (profile.value("report_directory").toString().isEmpty()) {
            selectRoute(ui::Route::Workbench); evaluation_->chooseDataset();
        } else loadEvaluationReport(profile.value("report_directory").toString());
    });
    return page;
}
QWidget *MainWindow::buildModelDisplay()
{
    auto *page = modelDisplayPage_ = new ModelDisplayPage;
    const auto &controls = page->controls();
    structureModelName_ = controls.modelName;
    structureModelMeta_ = controls.metadata;
    structureHint_ = controls.hint;
    structureStatus_ = controls.status;
    modelViewer_ = controls.viewer;
    connect(page, &ModelDisplayPage::modelLoaded, this, [this](const QString &path) {
        if (!isWorkbenchBusy() && (pages_->currentIndex() == ui::routeIndex(ui::Route::Models) ||
                       pages_->currentIndex() == ui::routeIndex(ui::Route::ModelDisplay)))
            showNotice(QString("模型结构、层级树与参数表已缓存 · %1").arg(QFileInfo(path).fileName()));
    });
    connect(page, &ModelDisplayPage::loadFailed, this, [this](const QString &error) {
        if (pages_->currentIndex() == ui::routeIndex(ui::Route::ModelDisplay)) showNotice(error, true);
    });
    return page;
}

void MainWindow::displayModelStructure()
{
    const QString path = modelState().activeModel;
    const bool needsRuntime = !path.isEmpty() && vision::cpuRuntimeDownloadNeeded();
    modelDisplayPage_->setSelectedModelHeader(path, modelDisplayName(path), needsRuntime);
    if (needsRuntime) {
        modelViewer_->stop();
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
    auto *page = historyPage_ = new HistoryPage(dataRoot_ + "/history.json");
    historyTable_ = page->controls().table;
    connect(page, &HistoryPage::exportRequested, this, [this] {
        const QString path = QFileDialog::getSaveFileName(this, "导出运行记录",
            dataRoot_ + "/history-export.json", "JSON (*.json)");
        if (path.isEmpty()) return;
        QString error;
        if (historyRepository_->exportTo(path, &error))
            showNotice("运行记录已导出：" + path);
        else showNotice("导出失败：" + error, true);
    });
    return page;
}
QWidget *MainWindow::buildMore()
{
    auto *page = morePage_ = new MorePage(settings_->value("restoreDatasetOnStartup", true).toBool(),
                                         vision::cpuRuntimeDownloadNeeded());
    const auto &controls = page->controls();
    restoreDatasetOnStartup_ = controls.restoreDataset;
    gpuEnvironmentStatus_ = controls.gpuEnvironmentStatus;
    gpuCheckButton_ = controls.gpuCheckButton;
    gpuPrepareButton_ = controls.gpuPrepareButton;
    gpuCancelButton_ = controls.gpuCancelButton;
    gpuSetupProgress_ = controls.gpuSetupProgress;
    gpuLog_ = controls.gpuLog;
    if (exportButton_ && !exportButton_->parent()) delete exportButton_;
    exportButton_ = controls.exportButton;
    connect(page, &MorePage::exportRequested, this, &MainWindow::exportResult);
    connect(page, &MorePage::restoreDatasetOnStartupChanged, this, [this](bool enabled) {
        settings_->setValue("restoreDatasetOnStartup", enabled);
        settings_->sync();
        inputSession_->setRestoreEnabled(enabled);
    });
    connect(page, &MorePage::gpuCheckRequested, this, [this] {
        if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).changeDevice) return;
        if (ui::ensureCpuRuntime(this, [this] { checkGpuEnvironment(); })) checkGpuEnvironment();
    });
    connect(page, &MorePage::gpuPrepareRequested, this, &MainWindow::prepareGpuEnvironment);
    connect(page, &MorePage::gpuCancelRequested, this, &MainWindow::cancelGpuPreparation);
    connect(page, &MorePage::cpuPrepareRequested, this, [this] {
        if (isClosing()) return;
        const auto ready = [this] {
            if (isClosing()) return;
            inspectSelectedModel(); displayModelStructure(); checkGpuEnvironment(); restoreDatasetSession();
            showNotice("CPU 运行环境已就绪，可离线使用 PT、转换、模型视图及评估。");
        };
        if (ui::ensureCpuRuntime(this, ready)) ready();
    });
    return page;
}

QWidget *MainWindow::buildRecordings()
{
    recordingsController_ = new vision::app::RecordingController(dataRoot_, this);
    auto *page = recordingsPage_ = new RecordingsPage(recordingsController_->directory());
    const auto &controls = page->controls();
    recordingsCount_ = controls.count;
    recordingsTable_ = controls.table;
    recordingCanvas_ = controls.canvas;
    recordingDetails_ = controls.details;
    recordingPlayButton_ = controls.play;
    exportRecordingButton_ = controls.exportVideo;
    connect(page, &RecordingsPage::refreshRequested, recordingsController_, &vision::app::RecordingController::refresh);
    connect(page, &RecordingsPage::selectionRequested, recordingsController_, &vision::app::RecordingController::selectRecording);
    connect(page, &RecordingsPage::playRequested, recordingsController_, &vision::app::RecordingController::togglePlayback);
    connect(page, &RecordingsPage::exportRequested, this, &MainWindow::exportRecording);
    connect(page, &RecordingsPage::openFolderRequested, this, [this] {
        const auto directory = recordingsController_->directory();
        if (!QDir().mkpath(directory) || !QDesktopServices::openUrl(QUrl::fromLocalFile(directory)))
            showNotice("无法打开录像文件夹：" + directory, true);
    });
    connect(recordingsController_, &vision::app::RecordingController::captureStateChanged, this,
            [this](const vision::app::RecordingCaptureState &) { updateRecordingUi(); });
    connect(recordingsController_, &vision::app::RecordingController::recordingsChanged, page, &RecordingsPage::setRecords);
    connect(recordingsController_, &vision::app::RecordingController::playbackChanged, page, &RecordingsPage::setPlayback);
    connect(recordingsController_, &vision::app::RecordingController::frameReady, page, &RecordingsPage::showFrame);
    connect(recordingsController_, &vision::app::RecordingController::notice, this, &MainWindow::showNotice);
    return page;
}

void MainWindow::clearImageResults()
{
    workbenchPage_->cancelImagePreview();
    detection_->results()->clearCache();
    if (queue_)
        for (int row = 0; row < queue_->count(); ++row)
            queue_->item(row)->setForeground(QBrush());
}

void MainWindow::selectQueueImage(int row)
{
    inputSession_->setSelection(row, queue_->selectionModel()->hasSelection());
    if (isWorkbenchBusy() || inputState().queuePopulating || inputState().sourceKind != vision::SourceKind::Images || row < 0 || row >= inputState().files.size())
        return;
    updateRangeUi();
    if (evaluationState().review.active && inputState().datasetSamples.contains(inputState().files.at(row))) {
        workbenchPage_->cancelImagePreview();
        reviewEvaluationImage(row); return;
    }
    if (detection_->results()->invalidateConfiguration(inferenceFingerprint())) {
        workbenchPage_->cancelImagePreview();
        for (int index = 0; index < queue_->count(); ++index) queue_->item(index)->setForeground(QBrush());
    }
    const QString source = inputState().files.at(row);
    const bool hadCache = imageResults().entries().contains(source);
    const auto *cached = detection_->results()->findCached(source, fileIdentity(source));
    if (hadCache && !cached) queue_->item(row)->setForeground(QBrush());
    if (cached && !cached->result.image.isNull()) {
        workbenchPage_->cancelImagePreview();
        workbenchPage_->setDisplayedConfig(cached->config); displayResult(cached->result); return;
    }
    if (cached && displayedResult().source == source && !displayedResult().image.isNull() &&
        canvas_->result().source == source && !canvas_->result().image.isNull()) return;
    workbenchPage_->clearDisplayedResult(); canvas_->clear();
    canvasTitle_->setText(QFileInfo(source).fileName());
    sizeMetric_->setText("—"); sizeMetric_->setToolTip("预览与导出图像尺寸");
    countMetric_->setText("—"); latencyMetric_->setText("—"); classMetric_->setText("—");
    predictionTable_->setRowCount(0); emptyResults_->show(); emptyResults_->setText("此图片尚无检测结果。");
    resultInfo_->setText("正在载入图片…"); exportButton_->setEnabled(false);
    const bool snapshot = cached && !cached->snapshotPath.isEmpty();
    workbenchPage_->requestImagePreview(snapshot ? cached->snapshotPath : source,
        snapshot ? source : QString(), cached ? cached->config.colorMode : currentConfig().colorMode);
}

void MainWindow::updateInputPreview()
{
    if (isWorkbenchBusy() || !canvas_)
        return;

    if (evaluationController_) evaluationController_->suspendReview();
    evaluationController_->setReviewActive(false);
    resetEvaluationReviewNavigation();
    for (int index = 0; index < queue_->count(); ++index) queue_->item(index)->setHidden(false);
    clearImageResults();
    const auto mode = currentConfig().colorMode;
    static_cast<LazyImageQueue *>(queue_)->setColorMode(mode);
    if (inputState().sourceKind == vision::SourceKind::Images) { selectQueueImage(queue_->currentRow()); return; }
    const auto previous = canvas_->result();
    if (previous.image.isNull())
        return;
    vision::InferenceResult preview;
    preview.source = previous.source;
    preview.task = currentConfig().task;
    preview.originalImage = previous.originalImage.isNull() ? previous.image : previous.originalImage;
    preview.sourceFrameSize = previous.sourceFrameSize;
    preview.image = vision::inputPreviewImage(preview.originalImage, mode);
    workbenchPage_->clearDisplayedResult();
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
    if (recordingsController_) recordingsController_->toggleRecording();
}

void MainWindow::updateRecordingUi()
{
    if (!recordButton_ || !recordingsController_) return;
    const auto &state = recordingsController_->captureState();
    const bool recording = state.requested || state.active;
    recordButton_->setText(state.buttonText);
    recordButton_->setIcon(ui::icon(recording ? "stop" : "record",
                                  recording ? QColor("#ffb0b0") : QColor("#D2D2D2")));
    recordButton_->setEnabled(state.enabled);
    recordButton_->setStyleSheet(recording ? "background:#432831; border-color:#8b4855; color:#ffb7b7;" : "");
    recordingStatus_->setStyleSheet(recording ? "color:#ffacb0; font-size:11px;" : "");
    recordingStatus_->setText(state.statusText);
}

void MainWindow::refreshRecordings()
{
    if (recordingsController_) recordingsController_->refresh();
}

void MainWindow::selectRecording()
{
    if (!recordingsController_ || !recordingsTable_) return;
    const int row = recordingsTable_->currentRow();
    const auto *item = row < 0 ? nullptr : recordingsTable_->item(row, 0);
    recordingsController_->selectRecording(item ? item->data(Qt::UserRole).toString() : QString());
}

void MainWindow::toggleRecordingPlayback()
{
    if (recordingsController_) recordingsController_->togglePlayback();
}


void MainWindow::stopRecordingPlayback()
{
    if (recordingsController_) recordingsController_->stopPlayback();
}

void MainWindow::exportRecording()
{
    if (!recordingsController_) return;
    const QString path = recordingsController_->playbackState().path;
    if (path.isEmpty()) return;
    const QString destination = QFileDialog::getSaveFileName(
        this, "导出录像", QDir::homePath() + "/" + QFileInfo(path).fileName(), "AVI 视频 (*.avi)");
    if (!destination.isEmpty()) recordingsController_->exportRecording(destination);
}

QWidget *MainWindow::buildGuide()
{
    return guidePage_ = new GuidePage;
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
                workbenchPage_->setDisplayedConfig(config);
                modelSession_->acceptLoadedMetadata(config, detection_->snapshot().modelIdentity);
                updateModelMeta("已加载");
                if (isPtModel(modelState().activeModel) && !isClosing())
                {
                    configuration_->setTask(config.task);
                }
                detection_->setConfigurationFingerprint(inferenceFingerprint(detection_->snapshot().modelIdentity));
                workbenchPage_->showResolvedDevice(backend, config);
                updateTaskUi();
                updateDeviceUi();
            });
    connect(detection_, &vision::app::DetectionController::status, this,
            [this](const QString &message)
            {
                if (!detection_->snapshot().failed && !detection_->snapshot().stopping)
                    showNotice(message);
            });
    connect(detection_, &vision::app::DetectionController::progress, this,
            [this](int done, int total)
            {
                if (inputState().sourceKind == vision::SourceKind::Images) {
                    updateRangeUi();
                    progress_->setRange(0, taskFiles().size());
                    progress_->setValue(completedCount());
                    return;
                }
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
                recordingsController_->setDetectionState(detection_->snapshot().running,
                    inputState().sourceKind, detection_->snapshot().stopping, true);
                detection_->reportFailure(s);
                resultInfo_->setText("推理失败 · 请检查模型与输入配置。");
                showNotice(s, true);
                if (!diagnosticRun_ && !isClosing())
                    QMessageBox::warning(this, "推理未完成", s);
            });
    connect(this, &MainWindow::startRequested, recordingsController_,
        [this](const vision::JobRequest &request) {
            recordingsController_->setDetectionState(true, request.sourceKind);
        });
    connect(recordingsController_, &vision::app::RecordingController::startRecordingRequested,
            detection_, &vision::app::DetectionController::requestStartRecording);
    connect(recordingsController_, &vision::app::RecordingController::stopRecordingRequested,
            detection_, &vision::app::DetectionController::requestStopRecording);
    connect(detection_, &vision::app::DetectionController::recordingStarted,
            recordingsController_, &vision::app::RecordingController::recordingStarted);
    connect(detection_, &vision::app::DetectionController::recordingFinished,
            recordingsController_, &vision::app::RecordingController::recordingFinished);
    connect(detection_, &vision::app::DetectionController::recordingFailed,
            recordingsController_, &vision::app::RecordingController::recordingFailed);
    connect(detection_, &vision::app::DetectionController::finished, recordingsController_,
            [this](bool) { recordingsController_->detectionFinished(); });
    connect(detection_, &vision::app::DetectionController::finished, this, &MainWindow::onFinished);
}

bool MainWindow::previewDropLocked() const
{
    return !vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).importInput;
}

bool MainWindow::handlePreviewDrop(const QStringList &input)
{
    if (previewDropLocked()) { showNotice("请先停止当前任务，再拖入新的输入。", true); return false; }
    cancelPendingDatasetSessionRestore();
    QStringList paths; QString error; vision::PreviewDropKind kind;
    if (!vision::classifyPreviewDrop(input, &paths, &kind, &error)) { showNotice(error, true); return false; }
    if (kind == vision::PreviewDropKind::Dataset) {
        loadDataset(paths.first(), evaluation_->datasetSplit(), evaluation_->metadataPath()); return true;
    }
    if (kind == vision::PreviewDropKind::Video) { loadVideo(paths.first()); return true; }
    bool append = inputState().sourceKind == vision::SourceKind::Images && inputState().datasetManifest.isEmpty();
    if (!inputState().datasetManifest.isEmpty()) {
        // The modal decision must never apply to a source changed by a nested callback.
        const auto previousFiles = inputState().files;
        const auto previousManifest = inputState().datasetManifest;
        const auto importGeneration = inputState().importGeneration;
        const auto queueGeneration = inputState().projectionGeneration;
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
        if (previewDropLocked() || previousFiles != inputState().files || previousManifest != inputState().datasetManifest ||
            importGeneration != inputState().importGeneration || queueGeneration != inputState().projectionGeneration) {
            showNotice("输入状态已变化，请重新拖入图片。", true); return false;
        }
        append = choice == 2;
    }
    importDroppedImages(paths, append);
    return true;
}

void MainWindow::importDroppedImages(const QStringList &paths, bool append)
{
    inputSession_->requestImages(paths, append);
}

void MainWindow::cancelDroppedImageImport()
{
    inputSession_->cancel();
}

void MainWindow::chooseVideo()
{
    if (isWorkbenchBusy()) return;
    cancelPendingDatasetSessionRestore();
    const QString p = QFileDialog::getOpenFileName(
        this, "选择视频", projectRoot_, "视频 (*.mp4 *.avi *.mkv *.mov *.webm *.m4v);;全部文件 (*)");
    if (p.isEmpty())
        return;
    loadVideo(p);
}

void MainWindow::loadVideo(const QString &path)
{
    if (!inputSession_->selectVideo(path)) return;
    auto *item = new QListWidgetItem(ui::icon("video"), QFileInfo(path).fileName());
    item->setToolTip(path); queue_->addItem(item);
    canvasTitle_->setText(QFileInfo(path).fileName());
    updateSourceUi(); updateRecordingUi();
}

void MainWindow::chooseCamera()
{
    if (isWorkbenchBusy()) return;
    if (!inputSession_->selectCamera()) return;
    queue_->addItem(new QListWidgetItem(ui::icon("camera"), "实时摄像头"));
    canvasTitle_->setText("摄像头预览");
    updateSourceUi();
}

void MainWindow::updateSourceUi()
{
    const bool analysisWorkspace = pages_ && pages_->currentIndex() == ui::routeIndex(ui::Route::Workbench) &&
        workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    if (workspaceHeader_) workspaceHeader_->setVisible(!analysisWorkspace);
    cameraIndex_->setVisible(inputState().sourceKind == vision::SourceKind::Camera);
    const bool streaming = inputState().sourceKind != vision::SourceKind::Images;
    const bool datasetContext = !inputState().datasetManifest.isEmpty();
    // Dataset reports carry their own metrics. Reclaim their duplicated header
    // and the image-only recording placeholder so the canvas fits a small window.
    for (QLabel *value : {countMetric_, latencyMetric_, classMetric_, sizeMetric_})
        value->parentWidget()->setVisible(!datasetContext && (streaming || !inputState().files.isEmpty()));
    countMetric_->parentWidget()->parentWidget()->setVisible(!datasetContext && (streaming || !inputState().files.isEmpty()) &&
        (!workbenchTabs_ || workbenchTabs_->currentIndex() == 0));
    if (evaluation_) evaluation_->controlsWidget()->setVisible(datasetContext || !evaluation_->datasetYaml().isEmpty());
    sourceLabel_->setVisible(!datasetContext);
    recordButton_->setVisible(streaming);
    recordingStatus_->setVisible(streaming);
    stereoLabel_->setVisible(streaming);
    stereoView_->setVisible(streaming);
    stereoView_->setEnabled(streaming && !isWorkbenchBusy());
    const QString view = stereoName(configurationState().stereoView);
    if (inputState().sourceKind == vision::SourceKind::Images)
        sourceLabel_->setText(inputState().datasetManifest.isEmpty()
            ? (inputState().files.isEmpty() ? QStringLiteral("拖入图片 / 数据集，或连接实时输入") : QString("临时图片 · %1 张 · 仅检测").arg(inputState().files.size()))
            : QString("YOLO 数据集 · %1 · %2 张标注图片%3").arg(inputState().datasetManifest.value("split").toString())
                  .arg(inputState().datasetManifest.value("samples").toArray().size())
                  .arg(inputState().files.size() > inputState().datasetSamples.size() ? QString(" + %1 张临时图片").arg(inputState().files.size() - inputState().datasetSamples.size()) : QString()));
    else if (inputState().sourceKind == vision::SourceKind::Video)
        sourceLabel_->setText("视频 · " + QFileInfo(inputState().streamPath).fileName() + " · " + view);
    else
        sourceLabel_->setText(QString("摄像头 %1 · %2 · 点击开始连接").arg(configurationState().cameraIndex).arg(view));
    if (displayedResult().image.isNull())
    {
        countMetric_->setText("—");
        latencyMetric_->setText("—");
        classMetric_->setText("—");
        if (inputState().sourceKind != vision::SourceKind::Images || inputState().files.isEmpty())
            sizeMetric_->setText("—");
        resultInfo_->setText(inputState().sourceKind == vision::SourceKind::Images && inputState().files.isEmpty()
                                 ? "请选择 YOLO 数据集、视频或摄像头。"
                                 : "输入已就绪，点击开始检测。");
    }
    sourceLabel_->setToolTip(sourceLabel_->text());
    updateRecordingUi();
    updateRangeUi();
}


QString MainWindow::modelFileIdentity() const
{
    return fileIdentity(modelState().activeModel);
}

QString MainWindow::inferenceFingerprint(const QString &modelIdentity) const
{
    return vision::app::WorkbenchConfiguration::configurationFingerprint(
        currentConfig(), modelIdentity.isEmpty() ? modelFileIdentity() : modelIdentity);
}

void MainWindow::invalidateConfiguredResults()
{
    if (initializing_ || isWorkbenchBusy() || !canvas_)
        return;
    if (classFilterStatus().error.isEmpty() && detection_->results()->hasResultsForConfiguration(inferenceFingerprint())) return;
    updateInputPreview();
    updateStartEligibility();
}

QStringList MainWindow::selectedImageRange(QString *error) const
{
    return inputSession_->range(static_cast<vision::app::ImageRange>(detectionRange_->currentIndex()), false, error);
}

QStringList MainWindow::selectedEvaluationRange(QString *error) const
{
    return inputSession_->range(static_cast<vision::app::ImageRange>(detectionRange_->currentIndex()), true, error);
}

void MainWindow::updateRangeUi()
{
    if (!detectionRangePanel_ || !rangeHint_ || !runButton_) return;
    const bool images = inputState().sourceKind == vision::SourceKind::Images;
    detectionRangePanel_->setVisible(images);
    detectionRange_->setEnabled(images && vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).editParameters);
    QString error;
    const auto range = isWorkbenchBusy() ? taskFiles() : selectedImageRange(&error);
    const int first = isWorkbenchBusy() ? taskFirst() : (range.isEmpty() ? 0 : inputState().files.indexOf(range.first()) + 1);
    const int last = isWorkbenchBusy() ? taskLast() : (range.isEmpty() ? 0 : inputState().files.indexOf(range.last()) + 1);
    rangeHint_->setText(!error.isEmpty() ? error
        : QString("本次检测 %1 张：第 %2～%3 张%4")
              .arg(range.size()).arg(first).arg(last)
              .arg(isWorkbenchBusy() ? QString(" · 完成 %1/%2").arg(completedCount()).arg(range.size()) : QString()));
    rangeHint_->setStyleSheet(error.isEmpty() ? QString() : "color: #FFB4AB;");
    updateStartEligibility();
}

void MainWindow::updateClassFilterUi()
{
    if (!classFilter_ || !classFilterHint_) return;
    const auto task = configurationState().parameters.task;
    const bool classification = task == vision::ModelTask::Classification;
    classFilter_->setEnabled(vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).editParameters && !classification);
    const auto status = classFilterStatus();
    classFilterHint_->setText(status.hint);
    classFilterHint_->setStyleSheet(status.error.isEmpty() ? QString() : "color: #FFB4AB;");
    updateStartEligibility();
}

void MainWindow::updateStartEligibility()
{
    if (!runButton_ || !classFilter_) return;
    QString rangeError;
    if (inputState().sourceKind == vision::SourceKind::Images) selectedImageRange(&rangeError);
    QString evaluationRangeError;
    const auto evaluationFiles = selectedEvaluationRange(&evaluationRangeError);
    const bool ready = vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).start &&
        !modelState().activeModel.isEmpty() && rangeError.isEmpty();
    const bool canEvaluate = evaluation_ && inputState().sourceKind == vision::SourceKind::Images &&
        !inputState().datasetManifest.isEmpty() && evaluationRangeError.isEmpty() && !evaluationFiles.isEmpty() &&
        evaluation_->canEvaluate() && !modelState().inspectionPending;
    const bool evaluationPage = workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    runButton_->setEnabled(ready && (evaluationPage ? canEvaluate : classFilterStatus().error.isEmpty()));
    runButton_->setText("开始检测");
    if (evaluationStartButton_) {
        evaluationStartButton_->setEnabled(ready && canEvaluate);
        evaluationStartButton_->setVisible(!isWorkbenchBusy());
        evaluationStopButton_->setVisible(isWorkbenchBusy());
        evaluationStopButton_->setEnabled(isWorkbenchBusy() && stopButton_->isEnabled());
        evaluationRange_->setEnabled(!isWorkbenchBusy() && inputState().sourceKind == vision::SourceKind::Images && !inputState().datasetManifest.isEmpty());
        evaluationTaskSummary_->setText(QString("%1 · %2 px · 原数据集 %3 张")
            .arg(modelDisplayName(modelState().activeModel)).arg(configurationState().parameters.inputSize).arg(inputState().datasetSamples.size()));
        evaluationTaskSummary_->setToolTip(modelState().activeModel);
        evaluationRangeHint_->setText(inputState().sourceKind != vision::SourceKind::Images || inputState().datasetManifest.isEmpty()
            ? "评估需要带标注的 YOLO 数据集；视频与摄像头请在图像预览页运行。"
            : !evaluationRangeError.isEmpty() ? evaluationRangeError :
              QString("本次评估 %1 张：原数据集第 %2～%3 张 · 临时图片不计入评估%4")
                .arg(isWorkbenchBusy() ? taskFiles().size() : evaluationFiles.size())
                .arg(isWorkbenchBusy() ? taskFirst() : inputState().files.indexOf(evaluationFiles.first()) + 1)
                .arg(isWorkbenchBusy() ? taskLast() : inputState().files.indexOf(evaluationFiles.last()) + 1)
                .arg(isWorkbenchBusy() ? QString(" · 完成 %1/%2").arg(completedCount()).arg(taskFiles().size()) : QString()));
        evaluationRangeHint_->setStyleSheet(evaluationRangeError.isEmpty() ? QString() : "color: #FFB4AB;");
    }
}

void MainWindow::startWorkbenchTask()
{
    if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).start) return;
    const bool evaluate = workbenchTabs_ && workbenchTabs_->currentIndex() == 1;
    if (evaluate && !evaluation_->canEvaluate()) return;
    evaluation_->setMode(evaluate);
    if (evaluate) startEvaluation();
    else startInference();
}

void MainWindow::inspectSelectedModel()
{
    if (initializing_ || isWorkbenchBusy() || isClosing()) return;
    modelSession_->inspect(currentConfig());
}

void MainWindow::addFiles(const QStringList &paths)
{
    if (isWorkbenchBusy()) return;
    inputSession_->requestImages(paths, inputState().sourceKind == vision::SourceKind::Images &&
        inputState().datasetManifest.isEmpty());
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
    if (modelLibraryDropFilter_) modelLibraryDropFilter_->showModelHint(message, error);
}
bool MainWindow::validateModelImport(const QStringList &input, QStringList *paths, QString *error) const
{
    return modelSession_->validateImport(input, paths, error);
}

bool MainWindow::importModelFiles(const QStringList &input)
{
    if (previewDropLocked()) { showModelImportHint("请先结束当前任务，再导入模型；当前模型与任务保持不变。", true); return false; }
    QStringList paths; QString error;
    if (!validateModelImport(input, &paths, &error)) { showModelImportHint(error, true); return false; }
    const QString selected = paths.last();
    if (isPtModel(selected) && !ui::ensureCpuRuntime(this, [this, paths] { importModelFiles(paths); })) return false;
    const auto before = modelState();
    const bool changed = before.activeModel != QFileInfo(selected).absoluteFilePath() ||
        (before.convertedProfiles.contains(selected) && convertedModelProfile(selected).isEmpty());
    const bool convertedBgr = isPtModel(selected) && !configurationState().parameters.swapRB &&
        configurationState().parameters.colorMode == vision::InputColorMode::Color;
    int added = 0;
    if (!modelSession_->importOrdered(paths, &added, &error)) { showModelImportHint(error, true); return false; }
    if (changed) presentSelectedModel(convertedBgr);
    else if (!initializing_ && !modelState().inspectionPending && modelState().inspectedIdentity != modelFileIdentity())
        inspectSelectedModel();
    refreshModelLibrary(); persist();
    showModelImportHint(added ? QString("已添加 %1 个模型；按输入顺序追加，已选用 %2，原文件保留。").arg(added).arg(modelDisplayName(modelState().activeModel))
                             : QString("模型已在库中，未重复添加；已选用 %1。").arg(modelDisplayName(modelState().activeModel)));
    return true;
}

bool MainWindow::rememberConvertedModel(const QString &path, const QJsonObject &metadata)
{
    QString error;
    if (!modelSession_->rememberConverted(path, metadata, &error)) { showNotice(error, true); return false; }
    persist();
    return true;
}

QJsonObject MainWindow::convertedModelProfile(const QString &path) const
{
    return modelSession_->convertedProfile(path);
}

bool MainWindow::applyConvertedModelProfile(const QString &path)
{
    const auto metadata = convertedModelProfile(path);
    if (metadata.isEmpty())
        return false;
    auto draft = configurationState();
    draft.parameters.task = metadata.value("task").toString() == "classify" ? vision::ModelTask::Classification :
        metadata.value("layout").toString() == "v5" ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8;
    draft.parameters.inputSize = normalizedEditorValue(inputSize_, metadata.value("shape").toArray()[2].toInt());
    if (modelState().inputChannels == 1) {
        draft.parameters.colorMode = vision::InputColorMode::Grayscale;
        draft.parameters.swapRB = true;
    }
    draft.parameters.scale = normalizedEditorValue(scale_, 1.0 / 255.0);
    draft.parameters.meanR = draft.parameters.meanG = draft.parameters.meanB = 0;
    configuration_->replaceDraft(draft);
    return true;
}
void MainWindow::setModel(const QString &path, bool force)
{
    if (isWorkbenchBusy()) return;
    const auto before = modelState();
    const QString absolute = path.isEmpty() ? QString() : QFileInfo(path).absoluteFilePath();
    if (!path.isEmpty() && before.activeModel == absolute && !force &&
        (!before.convertedProfiles.contains(absolute) || !convertedModelProfile(absolute).isEmpty())) {
        if (!initializing_ && !before.inspectionPending && before.inspectedIdentity != modelFileIdentity()) inspectSelectedModel();
        return;
    }
    if (!path.isEmpty() && !QFileInfo(path).isFile()) {
        refreshModelLibrary(); showNotice("模型文件不存在：" + path, true); return;
    }
    if (!initializing_ && isPtModel(absolute) && !ui::ensureCpuRuntime(this, [this, path, force] { setModel(path, force); })) return;
    const bool convertedBgr = isPtModel(absolute) && !configurationState().parameters.swapRB &&
        configurationState().parameters.colorMode == vision::InputColorMode::Color;
    QString error;
    if (!modelSession_->select(path, force, &error)) { refreshModelLibrary(); showNotice(error, true); return; }
    presentSelectedModel(convertedBgr);
}

void MainWindow::presentSelectedModel(bool convertedBgr)
{
    workbenchPage_->clearDeviceVerification();
    const auto state = modelState();
    if (state.activeModel.isEmpty()) {
        if (modelConversion_) modelConversion_->setSourceModel({});
        modelName_->setText("尚未选择模型"); modelMeta_->setText("请在模型库中选择一个模型。"); modelName_->setToolTip({});
        updateTaskUi(); updateInputPreview(); updateClassFilterUi(); displayModelStructure(); persist();
        return;
    }
    modelName_->setText(modelDisplayName(state.activeModel));
    if (modelConversion_) modelConversion_->setSourceModel(state.activeModel, modelDisplayName(state.activeModel));
    updateModelMeta("已选择"); modelName_->setToolTip(state.activeModel);
    if (state.hasSuggestedTask) configuration_->setTask(state.suggestedTask);
    applyConvertedModelProfile(state.activeModel);
    updateTaskUi(); updateInputPreview();
    if (!initializing_) inspectSelectedModel();
    persist();
    if (modelList_->count() != state.models.size()) refreshModelLibrary();
    else {
        const QSignalBlocker blocker(modelList_);
        for (int row = 0; row < modelList_->count(); ++row)
            if (modelList_->item(row)->data(Qt::UserRole).toString() == state.activeModel) { modelList_->setCurrentRow(row); break; }
    }
    displayModelStructure();
    showNotice(convertedBgr ? "PT 的彩色输入使用 RGB，已从 BGR 切换为 RGB；也可选择灰度。"
                            : "模型已全局选用，结构正在后台加载；开始检测时验证推理模型。");
}

void MainWindow::importLabels()
{
    if (isWorkbenchBusy()) return;
    const QString p = QFileDialog::getOpenFileName(this, "导入类别标签（每行一个名称）", projectRoot_,
                                                   "UTF-8 文本 (*.txt *.names);;全部文件 (*)");
    if (p.isEmpty())
        return;
    QString error;
    if (!configuration_->importLabels(p, &error)) { showNotice(error, true); return; }
    const auto labels = configurationState().parameters.labels;
    labelButton_->setText(QString("类别标签 · %1 个").arg(labels.size()));
    labelButton_->setToolTip(p);
    invalidateConfiguredResults();
    persist();
    showNotice("已导入 " + QString::number(labels.size()) + " 个类别标签。");
}
void MainWindow::startInference()
{
    if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).start)
        return;
    if (evaluation_ && evaluation_->evaluationMode()) { startEvaluation(); return; }
    if (isPtModel(modelState().activeModel) && !ui::ensureCpuRuntime(this, [this] { startInference(); })) return;
    evaluationController_->setReviewActive(false);
    resetEvaluationReviewNavigation();
    if (pendingResultWrites())
    {
        showNotice("正在后台保存已完成结果，保存完成后即可开始新任务。");
        return;
    }
    if (modelState().inspectedIdentity != modelFileIdentity() && !modelState().inspectionPending)
        inspectSelectedModel();
    updateClassFilterUi();
    updateRangeUi();
    if (!classFilterStatus().error.isEmpty())
        return;
    const auto task = operationCoordinator_->prepareDetection(*inputSession_, *modelSession_,
        *configuration_, static_cast<vision::app::ImageRange>(detectionRange_->currentIndex()));
    if (!task.valid) { if (!task.error.isEmpty()) showNotice(task.error, true); return; }
    if (modelState().inspectionPending) modelSession_->cancelInspection();
    workbenchPage_->clearDeviceVerification();
    const auto activation = operationCoordinator_->beginDetection(task, *detection_);
    if (!activation.started) return;
    if (activation.cacheInvalidated) {
        workbenchPage_->cancelImagePreview();
        for (int index = 0; index < queue_->count(); ++index) queue_->item(index)->setForeground(QBrush());
    }
    const auto &req = task.request;
    // A same-configuration rerun preserves previously accepted results until
    // each new result atomically replaces its entry. Progress counts only this job.
    workbenchPage_->clearDisplayedResult();
    workbenchPage_->setDisplayedConfig(req.config);
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
    emit startRequested(req);
}
void MainWindow::stopInference()
{
    const auto target = operationCoordinator_->stopWorkbench(*inputSession_, *evaluationController_,
        *detection_, *recordingsController_);
    if (target == vision::app::StopTarget::None || target == vision::app::StopTarget::Input) return;
    updateRecordingUi();
    stopButton_->setEnabled(false);
    updateStartEligibility();
    showNotice(target == vision::app::StopTarget::Evaluation
        ? "停止请求已发送 · 已完成样本将保留，未完成任务不会作为完整评估成绩"
        : "停止请求已发送 · 正在结束当前推理，已完成结果保留");
}

void MainWindow::setBusy(bool busy)
{
    busy = busy || isClosing();
    if (busy && evaluationState().review.pending) evaluationController_->suspendReview();
    if (busy) workbenchPage_->cancelImagePreview();
    if (evaluation_) evaluation_->setBusy(busy);
    if (modelConversion_) modelConversion_->setEnabled(!busy);
    for (auto *c : lockedControls_)
        c->setEnabled(!busy);
    if (modelsLibraryPage_) modelsLibraryPage_->setBusy(busy);
    runButton_->setVisible(!busy);
    stopButton_->setVisible(busy);
    stopButton_->setEnabled(busy);
    exportButton_->setEnabled(!busy && !displayedResult().image.isNull() && !displayedResult().demonstration);
    queue_->setEnabled(!busy);
    stereoView_->setEnabled(!busy && inputState().sourceKind != vision::SourceKind::Images);
    progress_->setVisible(busy);
    updateTaskUi();
    updateRecordingUi();
    updateDeviceUi();
    updateRangeUi();
    if (busy)
    {
        if (inputState().sourceKind == vision::SourceKind::Images) {
            progress_->setRange(0, taskFiles().size());
            progress_->setValue(0);
        } else progress_->setRange(0, 0);
    }
    updateEvaluationReviewUi();
    if (!busy && evaluationState().review.active && !evaluationState().report.isEmpty() && !evaluationState().review.available &&
        !evaluationState().review.pending && evaluationState().review.error.isEmpty())
        applyEvaluationFilter(evaluationState().review.filter);
}
void MainWindow::updateTaskUi()
{
    const bool editable = vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).editParameters;
    const bool pt = isPtModel(modelState().activeModel);
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
    confidence_->setEnabled(editable && !classification);
    iou_->setEnabled(editable && !classification);
    taskBox_->setEnabled(editable && !pt);
    taskBox_->setToolTip(pt ? "PT 模型的任务由文件内置架构自动识别" : "按照 ONNX 输出张量选择任务格式");
    inputColorMode_->setEnabled(editable);
    for (auto *control : QList<QWidget *>{scale_, meanR_})
        control->setEnabled(editable && !pt);
    meanG_->setEnabled(editable && !pt && !grayscale);
    meanB_->setEnabled(editable && !pt && !grayscale);
    meanRLabel_->setText(grayscale ? "灰度均值" : "均值 R");
    meanR_->setAccessibleName(grayscale ? "灰度均值" : "均值 R");
    meanR_->setToolTip(grayscale ? "ONNX 灰度输入统一使用该均值；三通道模型的三个灰度通道使用相同值。"
                                 : "RGB / BGR 彩色输入的 R 通道均值");
    meanG_->setToolTip(grayscale ? "灰度模式统一使用灰度均值，忽略此项。" : "G 通道均值");
    meanB_->setToolTip(grayscale ? "灰度模式统一使用灰度均值，忽略此项。" : "B 通道均值");
    if (pt)
    {
        const QSignalBlocker scaleBlocker(scale_), rBlocker(meanR_), gBlocker(meanG_), bBlocker(meanB_);
        scale_->setValue(1.0 / 255.0);
        meanR_->setValue(0);
        meanG_->setValue(0);
        meanB_->setValue(0);
        if (!isClosing()) {
            auto draft = configurationState();
            draft.parameters.scale = 1.0 / 255.0;
            draft.parameters.meanR = draft.parameters.meanG = draft.parameters.meanB = 0;
            draft.parameters.swapRB = true;
            configuration_->replaceDraft(draft);
        }
    }
    preprocessHint_->setText(
        pt ? (grayscale ? "灰度输入 · 模型原生归一化" : "PT 彩色使用 RGB · 原生预处理")
           : (grayscale ? "灰度输入 · 统一使用灰度均值" : "彩色输入 · RGB / BGR 通道顺序"));
    if (configurationState().parameters.labels.isEmpty())
        labelButton_->setText(pt ? (modelState().nativeLabels.isEmpty()
                                        ? "类别标签 · 模型自动读取"
                                        : QString("类别标签 · 内置 %1 类").arg(modelState().nativeLabels.size()))
                                 : (!convertedModelProfile(modelState().activeModel).isEmpty()
                                        ? QString("类别标签 · 转换模型 %1 类").arg(modelState().nativeLabels.size())
                                        : (classification ? "类别标签 · 数字类别"
                                            : (modelState().classCount == 80 ? "类别标签 · 默认 COCO 80"
                                                : modelState().classCount > 0 ? QString("类别标签 · 数字 %1 类").arg(modelState().classCount)
                                                                      : QString("类别标签 · 模型默认")))));
    else
        labelButton_->setText(QString("类别标签 · %1 个").arg(configurationState().parameters.labels.size()));
    updateEvaluationModel();
    updateClassFilterUi();
    updateDeviceUi();
}
void MainWindow::setComputeDevice(vision::ComputeDevice device, int index)
{
    index = qBound(0, index, 63);
    if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).changeDevice)
        return;
    auto draft = configurationState();
    draft.parameters.device = device; draft.parameters.deviceIndex = index;
    configuration_->replaceDraft(draft);
    workbenchPage_->clearDeviceVerification();
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
    const bool changeDevice = vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).changeDevice;
    const auto requested = computeMode(computeDevice_->currentData().toString());
    computeDevice_->setEnabled(changeDevice);
    gpuDeviceIndex_->setEnabled(changeDevice && requested != vision::ComputeDevice::CPU);
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
        const QString device = workbenchPage_->devicePresentation().device == vision::ComputeDevice::CUDA
                                   ? QString("GPU %1 · %2").arg(workbenchPage_->devicePresentation().index).arg(workbenchPage_->devicePresentation().name)
                                   : QStringLiteral("CPU");
        const QString actual = device + "\n" + workbenchPage_->devicePresentation().backend;
        backendBadge_->setText(
            workbenchPage_->devicePresentation().known
                ? actual
                : QString("所选：%1 · %2").arg(computeName(requested)).arg(isWorkbenchBusy() ? "正在验证" : "待运行"));
        backendBadge_->setToolTip(
            workbenchPage_->devicePresentation().known
                ? actual + (workbenchPage_->devicePresentation().notice.isEmpty() ? QString() : "\n" + workbenchPage_->devicePresentation().notice)
                : hint);
    }
    if (backendFooter_)
    {
        const QString footer =
            workbenchPage_->devicePresentation().known ? QString("Qt 6.8.3  ·  %1  ·  %2")
                                     .arg(workbenchPage_->devicePresentation().device == vision::ComputeDevice::CUDA ? "NVIDIA GPU" : "CPU",
                                          workbenchPage_->devicePresentation().backend)
                               : QString("Qt 6.8.3  ·  %1待验证").arg(computeName(requested));
        backendFooter_->setText(backendFooter_->fontMetrics().elidedText(footer, Qt::ElideRight, 300));
        backendFooter_->setToolTip(footer + "\n" + workbenchPage_->devicePresentation().name + "\n" + workbenchPage_->devicePresentation().notice);
    }
    if (gpuCheckButton_)
    {
        gpuCheckButton_->setEnabled(changeDevice && !checking);
        gpuPrepareButton_->setEnabled(changeDevice && !checking);
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
    if (vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).changeDevice) gpuService_->checkEnvironment();
}



void MainWindow::prepareGpuEnvironment()
{
    if (!vision::app::OperationCoordinator::availability(operationCoordinator_->activity()).changeDevice) return;
    if (!ui::ensureCpuRuntime(this, [this] { prepareGpuEnvironment(); })) return;
    if (!isWorkbenchBusy()) gpuService_->prepareEnvironment();
}

void MainWindow::cancelGpuPreparation()
{
    gpuService_->cancelPreparation();
}

void MainWindow::updateModelMeta(const QString &state)
{
    const QString channels =
        modelState().inputChannels > 0 ? QString("输入 %1 通道").arg(modelState().inputChannels) : "输入通道待识别";
    modelMeta_->setText(QString("%1  ·  %2 MB  ·  %3  ·  %4")
                            .arg(modelFormat(modelState().activeModel))
                            .arg(QFileInfo(modelState().activeModel).size() / 1048576.0, 0, 'f', 1)
                            .arg(channels)
                            .arg(state));
}
void MainWindow::onResult(const vision::InferenceResult &input, quint64 ticket)
{
    auto r = input;
    attachGroundTruth(r);
    if (inputState().sourceKind == vision::SourceKind::Images && !detection_->acceptImageResult(r, ticket)) return;
    if (r.modelClassCount > 0) {
        modelSession_->acceptInferenceClassCount(r.modelClassCount);
        workbenchPage_->setDisplayedClassCount(r.modelClassCount);
        updateClassFilterUi();
    }
    if (inputState().sourceKind == vision::SourceKind::Images) {
        updateRangeUi();
        progress_->setRange(0, taskFiles().size());
        progress_->setValue(completedCount());
    }
    displayResult(r);
    if (inputState().sourceKind == vision::SourceKind::Images)
    {
        const int row = inputState().files.indexOf(r.source);
        if (row >= 0)
        {
            queue_->setCurrentRow(row);
            queue_->item(row)->setForeground(QColor("#60CDFF"));
        }
    }
}

void MainWindow::displayResult(const vision::InferenceResult &input)
{
    auto r = input;
    if (!evaluationState().review.active) attachGroundTruth(r);
    workbenchPage_->showResult(r, inputState().sourceKind, inputState().streamPath, isWorkbenchBusy());
    if (r.evaluationReport.isEmpty()) { updateDeviceUi(); updateModelMeta("已验证"); }
    exportButton_->setEnabled(!isClosing() && !isWorkbenchBusy() && !r.image.isNull() && !r.demonstration);
}
void MainWindow::onFinished(bool cancelled)
{
    setBusy(false);
    if (inputState().sourceKind == vision::SourceKind::Images) {
        // Restores the selected accepted result even when cancellation happened
        // before the first new frame and the current queue row never changed.
        selectQueueImage(queue_->currentRow());
    }
    if (detection_->snapshot().modelIdentity != modelFileIdentity()) {
        invalidateConfiguredResults();
        inspectSelectedModel();
    }
    refreshHistory();
    if (detection_->snapshot().failed && !detection_->snapshot().error.isEmpty())
        showNotice(detection_->snapshot().error, true);
    else if (!detection_->snapshot().failed)
        showNotice(cancelled ? (inputState().sourceKind == vision::SourceKind::Images
                                   ? QString("任务已停止 · 本次完成 %1 / %2 张 · 已完成结果可回看与导出")
                                         .arg(completedCount()).arg(taskFiles().size())
                                   : QString("任务已停止 · 已处理 %1 帧 · 当前结果可导出").arg(completedCount()))
                             : QString("检测完成 · 已处理 %1 %2")
                                   .arg(completedCount())
                                   .arg(inputState().sourceKind == vision::SourceKind::Images ? "张图片" : "帧"));

    if (isClosing())
        QTimer::singleShot(0, this, &QWidget::close);
}

void MainWindow::runDemo()
{
    if (isWorkbenchBusy())
        return;
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", projectRoot_);
    const QString model = fixtures + "/models/yolov5n.onnx", image = fixtures + "/assets/bus.jpg";
    if (!QFileInfo::exists(model) || !QFileInfo::exists(image))
    {
        showNotice("示例资源尚未准备好。请导入自己的 ONNX 模型与图片。", true);
        if (diagnosticRun_)
            requestExit(2);
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
    configuration_->setLabels({}, {});
    labelButton_->setText("类别标签 · 默认 COCO 80");
    clearImageResults();
    inputSession_->clear();
    connect(inputSession_, &vision::app::InputSessionController::finished, this, [this](bool cancelled) {
        if (!cancelled && !workbenchPage_->inputPresentation().failed) startInference();
    }, Qt::SingleShotConnection);
    addFiles({image});
    selectRoute(ui::Route::Workbench);
}
void MainWindow::runSmoke(const QString &directory)
{
    vision::ui::smoke::runInferenceSmoke(*this, directory, false);
}
void MainWindow::runPtDemo()
{
    if (isWorkbenchBusy())
        return;
    const QString fixtures = qEnvironmentVariable("VISION_STUDIO_SMOKE_INPUT_ROOT", projectRoot_);
    const QString model = fixtures + "/models/yolov8n.pt", image = fixtures + "/assets/bus.jpg";
    if (!QFileInfo::exists(model) || !QFileInfo::exists(image))
    {
        showNotice("PT 示例资源不存在，请导入自己的 .pt 模型。", true);
        if (diagnosticRun_)
            requestExit(2);
        return;
    }
    configuration_->setLabels({}, {});
    setModel(model);
    taskBox_->setCurrentIndex(1);
    inputSize_->setValue(640);
    confidence_->setValue(.25);
    iou_->setValue(.45);
    clearImageResults();
    inputSession_->clear();
    connect(inputSession_, &vision::app::InputSessionController::finished, this, [this](bool cancelled) {
        if (!cancelled && !workbenchPage_->inputPresentation().failed) startInference();
    }, Qt::SingleShotConnection);
    addFiles({image});
    selectRoute(ui::Route::Workbench);
    updateTaskUi();
}
void MainWindow::runPtSmoke(const QString &directory)
{
    vision::ui::smoke::runInferenceSmoke(*this, directory, true);
}
void MainWindow::showModelStructure(const QString &model)
{
    if (!model.isEmpty())
        setModel(model);
    selectRoute(ui::Route::ModelDisplay);
}

void MainWindow::runModelSmoke(const QString &directory, const QString &model)
{
    vision::ui::smoke::runModelDisplaySmoke(*this, directory, model);
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
    const auto historical = result.evaluationReport == evaluationState().directory
        ? evaluationState().report.value("config").toObject() : QJsonObject{};
    return vision::app::ResultWriter::prepare(result, directory, config, historical);
}

bool MainWindow::writePreparedResult(const ResultExport &result, QString *error)
{
    return vision::app::ResultWriter::writePrepared(result, error);
}

bool MainWindow::writeResult(const vision::InferenceResult &result, const QString &directory, QString *error)
{
    return writePreparedResult(prepareResultExport(result, directory, displayedConfig()), error);
}

void MainWindow::exportResult()
{
    if (isClosing()) return;
    if (displayedResult().image.isNull() || displayedResult().demonstration)
        return;
    const QString dir = QFileDialog::getExistingDirectory(this, "选择结果导出目录",
                                                          QDir(exportDirectory()).exists() ? exportDirectory() : dataRoot_);
    if (dir.isEmpty())
        return;
    QString error;
    if (writeResult(displayedResult(), dir, &error))
    {
        configuration_->setExportDirectory(dir);
        workbenchPage_->setExportDirectory(dir);
        persist();
        showNotice("已导出标注 PNG、JSON 与 CSV 到：" + dir);
    }
    else
        showNotice("导出失败：" + error, true);
}
void MainWindow::refreshHistory()
{
    if (historyPage_) historyPage_->setRecords(historySnapshot());
}
QString MainWindow::modelDisplayName(const QString &path) const
{
    return modelSession_->displayName(path);
}

void MainWindow::removeLibraryModel(const QString &path)
{
    if (isWorkbenchBusy() || !modelState().models.contains(path)) return;
    const bool active = modelState().activeModel == path;
    if (!modelSession_->remove(path)) return;
    refreshModelLibrary();
    if (active) presentSelectedModel();
    persist(); showNotice("已从模型库移除，原始模型文件保留。");
}

void MainWindow::refreshModelLibrary()
{
    if (!modelsLibraryPage_) return;
    const auto state = modelState();
    QVector<ModelLibraryRow> rows;
    for (const QString &path : state.models) {
        const QFileInfo file(path);
        ModelLibraryRow row;
        row.path = path;
        row.displayName = modelDisplayName(path);
        row.detail = file.exists()
            ? QString("%1 MB  ·  %2").arg(file.size() / 1048576.0, 0, 'f', 1).arg(modelFormat(path))
            : QStringLiteral("文件已移动或不存在");
        row.note = state.notes.value(path).toString();
        const auto profile = state.evaluationProfiles.value(path).toMap();
        if (!profile.isEmpty())
            row.evaluationSummary = profile.value("model_identity").toString() == fileIdentity(path)
                ? QString("评估 · %1 · %2%3").arg(profile.value("split").toString())
                    .arg(profile.value("full_split").toBool() ? "完整集合" : "子集/未完成")
                    .arg(profile.value("adopted").toBool() ? QString(" · 已采用阈值 %1")
                        .arg(profile.value("adopted_threshold").toDouble(), 0, 'f', 2) : QString())
                : QStringLiteral("模型文件已变更 · 原评估仅供历史回看");
        row.tooltip = row.displayName + "\n原始文件：" + file.fileName() + "\n" + path +
            (row.note.isEmpty() ? QString() : "\n备注：" + row.note);
        row.height = (row.note.isEmpty() ? 94 : 118) + (profile.isEmpty() ? 0 : 25);
        rows.append(row);
    }
    modelsLibraryPage_->setRows(rows, state.activeModel);
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
    if (!settings_ || !configuration_)
        return;
    settings_->saveWorkspace(modelState(), configurationState(), taskBox_ != nullptr);
}
void MainWindow::closeEvent(QCloseEvent *event)
{
    if (operationCoordinator_->isShutdownReady()) { persist(); event->accept(); return; }
    cancelPendingDatasetSessionRestore();
    showNotice("正在安全结束任务并保存已完成结果…");
    operationCoordinator_->requestShutdown();
    event->ignore();
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
        if (e->key() == Qt::Key_O && !isWorkbenchBusy())
        {
            evaluation_->chooseDataset();
            return;
        }
        if (e->key() == Qt::Key_M && !isWorkbenchBusy())
        {
            importModel();
            return;
        }
        if (e->key() == Qt::Key_R)
        {
            startWorkbenchTask();
            return;
        }
        if (e->key() == Qt::Key_E && !isWorkbenchBusy())
        {
            exportResult();
            return;
        }
    }
    QMainWindow::keyPressEvent(e);
}

vision::ModelConfig MainWindow::displayedConfig() const { return workbenchPage_->displayedConfig(); }
vision::InferenceResult MainWindow::displayedResult() const { return workbenchPage_->displayedResult(); }
bool MainWindow::exportDisplayedResult(const QString &directory, QString *base, QString *error)
{
    const auto prepared = prepareResultExport(displayedResult(), directory, displayedConfig());
    if (base) *base = prepared.base;
    return writePreparedResult(prepared, error);
}
