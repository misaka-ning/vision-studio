#include "mainwindow.h"
#include "core/gpuruntime.h"
#include "core/inferenceworker.h"
#include "icons.h"
#include "imagecanvas.h"
#include "modelviewer.h"
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDebug>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPalette>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
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
#include <QTextBrowser>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
QImage readImage(const QString &path)
{
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
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
    return QFileInfo(path).suffix().compare("pt", Qt::CaseInsensitive) == 0;
}
QString modelFormat(const QString &path)
{
    return isPtModel(path) ? QString("PT") : QString("ONNX");
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
    settings_ = new QSettings(dataRoot_ + "/preferences.ini", QSettings::IniFormat, this);
    exportDir_ = settings_->value("exportDirectory", dataRoot_ + "/results").toString();
    models_ = settings_->value("models").toStringList();
    QFile h(dataRoot_ + "/history.json");
    if (h.open(QIODevice::ReadOnly))
        history_ = QJsonDocument::fromJson(h.readAll()).array();
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
    auto *header = new QHBoxLayout;
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
    demoButton_ = button("运行示例", "play");
    demoButton_->setToolTip("选择 ONNX 或 PT 示例，使用真实模型检测示例图片");
    demoButton_->setObjectName("demoButton");
    auto *demoMenu = new QMenu(demoButton_);
    demoMenu->addAction("ONNX · YOLOv5 Nano", this, &MainWindow::runDemo);
    demoMenu->addAction("PT · YOLOv8 Nano", this, &MainWindow::runPtDemo);
    demoButton_->setMenu(demoMenu);
    exportButton_ = button("导出结果", "export");
    exportButton_->setEnabled(false);
    exportButton_->setObjectName("exportResultButton");
    connect(exportButton_, &QPushButton::clicked, this, &MainWindow::exportResult);
    w->addLayout(header);
    pages_ = new QStackedWidget;
    pages_->setObjectName("workspacePages");
    pages_->addWidget(buildWorkbench());
    pages_->addWidget(buildModels());
    pages_->addWidget(buildModelDisplay());
    pages_->addWidget(buildHistory());
    pages_->addWidget(buildGuide());
    pages_->addWidget(buildRecordings());
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
    connectWorker();
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
    if (!saved.isEmpty() && QFileInfo::exists(saved))
        setModel(saved);
    else if (QFileInfo::exists(bundled))
        setModel(bundled);
    for (const QString &name : {QStringLiteral("yolov8n.pt"), QStringLiteral("yolov5n.pt")})
    {
        const QString example = projectRoot_ + "/models/" + name;
        if (QFileInfo(example).isFile() && !models_.contains(example))
            models_.append(example);
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
    labels_ = savedLabels;
    labelsPath_ = savedLabelsPath;
    if (!labels_.isEmpty())
    {
        labelButton_->setText(QString("类别标签 · %1 个").arg(labels_.size()));
        labelButton_->setToolTip(labelsPath_);
    }
    const QString sample = projectRoot_ + "/assets/bus.jpg";
    if (QFileInfo::exists(sample))
        addFiles({sample});
    else
    {
        canvas_->setResult(ImageCanvas::createDemoResult());
        canvasTitle_->setText("示范场景 · 交互预览");
        resultInfo_->setText("示范框用于预览界面；导入模型后可执行真实推理。");
    }
    refreshModelLibrary();
    refreshHistory();
    refreshRecordings();
    selectRoute(0);
    updateTaskUi();
    updateDeviceUi();
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
                     chooseImages();
             });
    shortcut(QKeySequence("Ctrl+M"),
             [this]
             {
                 if (!busy_)
                     importModel();
             });
    shortcut(QKeySequence("Ctrl+R"), [this] { startInference(); });
    shortcut(QKeySequence("Ctrl+E"),
             [this]
             {
                 if (!busy_)
                     exportResult();
             });
    shortcut(QKeySequence(Qt::Key_Escape),
             [this]
             {
                 if (busy_)
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
    for (QProcess *process : {gpuProbeProcess_, gpuSetupProcess_})
        if (process)
        {
            process->disconnect(this);
            if (process->state() != QProcess::NotRunning)
            {
                process->terminate();
                if (!process->waitForFinished(300))
                {
                    process->kill();
                    process->waitForFinished(300);
                }
            }
        }
    if (modelViewer_)
        modelViewer_->stop();
    stopRecordingPlayback();
    worker_->requestStop();
    workerThread_->quit();
    workerThread_->wait();
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
    setPalette(colors);
    QFile theme(QStringLiteral(":/fluent-dark.qss"));
    if (theme.open(QIODevice::ReadOnly))
        setStyleSheet(QString::fromUtf8(theme.readAll()));
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
    auto *cap = text("2.0 BETA  ·  本地视觉工作空间", "brandCaption");
    cap->setContentsMargins(4, 4, 0, 0);
    l->addWidget(cap);
    l->addSpacing(24);
    l->addWidget(text("工作空间", "eyebrow"));
    l->addSpacing(5);
    const QStringList titles = {"检测工作台", "模型库",   "模型显示", "运行记录",
                                "使用指南",   "录制视频", "更多"};
    const QStringList icons = {"work", "model", "graph", "history", "help", "video", "more"};
    for (int i = 0; i < titles.size(); ++i)
    {
        auto *b = button(titles[i], icons[i], "nav");
        b->setCheckable(true);
        navButtons_.append(b);
        l->addWidget(b);
        connect(b, &QPushButton::clicked, this, [this, i] { selectRoute(i); });
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
    all->addLayout(metrics);
    auto *inputCommands = card("commandBar");
    auto *commandLayout = new QVBoxLayout(inputCommands);
    commandLayout->setContentsMargins(16, 6, 16, 6);
    commandLayout->setSpacing(6);
    auto *commands = new QHBoxLayout;
    commands->setContentsMargins(0, 0, 0, 0);
    commands->setSpacing(16);
    commands->addWidget(text("输入源", "sectionTitle"));
    auto *sources = new QHBoxLayout;
    sources->setSpacing(8);
    QStringList names = {"图片", "文件夹", "视频", "摄像头"};
    QStringList icons = {"image", "folder", "video", "camera"};
    for (int i = 0; i < 4; ++i)
    {
        auto *b = button(names[i], icons[i]);
        sources->addWidget(b);
        lockedControls_.append(b);
        if (i == 0)
            connect(b, &QPushButton::clicked, this, &MainWindow::chooseImages);
        if (i == 1)
            connect(b, &QPushButton::clicked, this, &MainWindow::chooseFolder);
        if (i == 2)
            connect(b, &QPushButton::clicked, this, &MainWindow::chooseVideo);
        if (i == 3)
            connect(b, &QPushButton::clicked, this, &MainWindow::chooseCamera);
    }
    sourceLabel_ = text("尚未选择输入", "tiny");
    sourceLabel_->setWordWrap(true);
    sourceLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    sourceLabel_->setMaximumHeight(40);
    commands->addLayout(sources);
    commands->addWidget(sourceLabel_, 1);
    commandLayout->addLayout(commands);
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
    auto *devices = new QHBoxLayout;
    devices->setSpacing(12);
    devices->addWidget(text("推理设备", "sectionTitle"));
    computeDevice_ = new QComboBox;
    computeDevice_->setObjectName("computeDevice");
    computeDevice_->setAccessibleName("推理设备");
    computeDevice_->addItem("自动 · 优先 NVIDIA GPU", "auto");
    computeDevice_->addItem("CPU", "cpu");
    computeDevice_->addItem("NVIDIA GPU", "cuda");
    computeDevice_->setToolTip("自动模式优先使用已就绪的 NVIDIA GPU，无法使用时说明原因并使用 CPU；"
                               "显式选择 NVIDIA GPU 时，不会静默回退到 CPU。");
    computeDevice_->setMinimumWidth(220);
    computeDevice_->setMaximumWidth(270);
    devices->addWidget(computeDevice_);
    gpuDeviceLabel_ = text("GPU 设备", "tiny");
    gpuDeviceIndex_ = new QComboBox;
    gpuDeviceIndex_->setObjectName("gpuDeviceIndex");
    gpuDeviceIndex_->setAccessibleName("GPU 设备编号");
    gpuDeviceIndex_->addItem("GPU 0 · 待检查", 0);
    gpuDeviceIndex_->setMinimumWidth(220);
    gpuDeviceIndex_->setMaximumWidth(300);
    devices->addWidget(gpuDeviceLabel_);
    devices->addWidget(gpuDeviceIndex_);
    deviceHint_ = text("正在检查 GPU 环境…", "tiny");
    deviceHint_->setObjectName("deviceEnvironmentHint");
    deviceHint_->setWordWrap(true);
    deviceHint_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    deviceHint_->setMaximumHeight(40);
    devices->addWidget(deviceHint_, 1);
    commandLayout->addLayout(devices);
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
                             updateTaskUi();
                             persist();
                             showNotice("已恢复模型默认类别。");
                         });
    labelButton_->setMenu(labelMenu);
    lockedControls_.append(labelButton_);
    fields->addSpacing(7);
    fields->addWidget(section("推理参数", "03"));
    auto *form = new QFormLayout;
    form->setSpacing(9);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inputSize_ = new QSpinBox;
    inputSize_->setRange(32, 2048);
    inputSize_->setSingleStep(32);
    inputSize_->setValue(640);
    inputSize_->setSuffix(" px");
    form->addRow("输入尺寸", inputSize_);
    confidence_ = new QDoubleSpinBox;
    confidence_->setRange(0.01, 1.0);
    confidence_->setDecimals(2);
    confidence_->setSingleStep(0.05);
    confidence_->setValue(0.25);
    form->addRow("置信度", confidence_);
    iou_ = new QDoubleSpinBox;
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
    lockedControls_.append(autoExport_);
    backendBadge_ = text("设备待验证", "chip");
    backendBadge_->setObjectName("actualDeviceBadge");
    backendBadge_->setWordWrap(true);
    backendBadge_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fields->addWidget(backendBadge_);
    fields->addStretch();
    scroll->setWidget(inner);
    cl->addWidget(scroll, 1);
    runButton_ = button("开始检测", "play", "primary");
    runButton_->setMinimumHeight(44);
    cl->addWidget(runButton_);
    connect(runButton_, &QPushButton::clicked, this, &MainWindow::startInference);
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
    toolbar->setObjectName("toolbar");
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
    canvas_->setMinimumHeight(280);
    cf->addWidget(canvas_, 1);
    auto *cb = new QHBoxLayout;
    cb->setContentsMargins(14, 3, 14, 3);
    cb->addWidget(text("滚轮缩放 · 拖动平移 · 点击目标查看", "tiny"), 1);
    zoomLabel_ = text("100%", "tiny");
    cb->addWidget(zoomLabel_);
    cf->addLayout(cb);
    ml->addWidget(canvasFrame, 1);
    connect(fit, &QPushButton::clicked, canvas_, &ImageCanvas::fitToView);
    connect(actual, &QPushButton::clicked, canvas_, &ImageCanvas::actualSize);
    connect(showBoxes_, &QCheckBox::toggled, canvas_, &ImageCanvas::setBoxesVisible);
    connect(showLabels_, &QCheckBox::toggled, canvas_, &ImageCanvas::setLabelsVisible);
    connect(canvas_, &ImageCanvas::zoomChanged, this,
            [this](int z) { zoomLabel_->setText(QString::number(z) + "%"); });
    connect(canvas_, &ImageCanvas::fileDropped, this,
            [this](const QString &p)
            {
                if (!busy_)
                    addFiles({p});
                else
                    showNotice("请先停止任务，再更换输入。");
            });
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
    queue_ = new QListWidget;
    queue_->setFlow(QListView::LeftToRight);
    queue_->setViewMode(QListView::IconMode);
    queue_->setIconSize(QSize(64, 36));
    queue_->setGridSize(QSize(100, 64));
    queue_->setWrapping(false);
    queue_->setFixedHeight(68);
    queue_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ql->addWidget(queue_, 1);
    ml->addWidget(queueFrame);
    connect(queue_, &QListWidget::currentRowChanged, this,
            [this](int row)
            {
                if (!busy_ && sourceKind_ == vision::SourceKind::Images && row >= 0 && row < files_.size())
                {
                    vision::InferenceResult r;
                    r.source = files_[row];
                    r.originalImage = readImage(r.source);
                    r.image = vision::inputPreviewImage(r.originalImage, currentConfig().colorMode);
                    if (!r.image.isNull())
                    {
                        lastResult_ = {};
                        canvas_->setResult(r);
                        canvasTitle_->setText(QFileInfo(r.source).fileName());
                        sizeMetric_->setText(QString("%1 × %2").arg(r.image.width()).arg(r.image.height()));
                        countMetric_->setText("—");
                        latencyMetric_->setText("—");
                        classMetric_->setText("—");
                        predictionTable_->setRowCount(0);
                        emptyResults_->show();
                        resultInfo_->setText("图片已就绪，点击开始检测。");
                        exportButton_->setEnabled(false);
                    }
                }
            });
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
    rl->addWidget(resultInfo_);
    predictionTable_ = new QTableWidget(0, 3);
    predictionTable_->setHorizontalHeaderLabels({"类别", "置信度", "编号"});
    tableStyle(predictionTable_);
    predictionTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    predictionTable_->setColumnWidth(1, 78);
    predictionTable_->setColumnWidth(2, 38);
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
    connect(taskBox_, &QComboBox::currentIndexChanged, this, [this] { updateTaskUi(); });
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
    all->addWidget(columns, 1);
    return page;
}

QWidget *MainWindow::buildModels()
{
    auto *page = new QWidget;
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
    modelList_ = new QListWidget;
    modelList_->setObjectName("globalModelList");
    lockedControls_.append(modelList_);
    modelList_->setIconSize(QSize(34, 34));
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
    actions->addStretch();
    fl->addLayout(actions);
    connect(structure, &QPushButton::clicked, this, [this] { selectRoute(2); });
    connect(activate, &QPushButton::clicked, this, [this] { selectRoute(0); });
    connect(modelList_, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem *item)
            {
                if (item && !busy_)
                    setModel(item->data(Qt::UserRole).toString());
            });
    connect(modelList_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *) { selectRoute(0); });
    connect(remove, &QPushButton::clicked, this,
            [this]
            {
                auto *i = modelList_->currentItem();
                if (i)
                {
                    const QString p = i->data(Qt::UserRole).toString();
                    models_.removeAll(p);
                    refreshModelLibrary();
                    if (p == modelPath_)
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
                                          : state == ModelViewer::State::Ready ? QString("三种视图已缓存")
                                          : state == ModelViewer::State::Error ? "无法显示"
                                                                               : "等待模型");
            });
    connect(modelViewer_, &ModelViewer::modelLoaded, this,
            [this](const QString &path)
            {
                if (!busy_ && (pages_->currentIndex() == 1 || pages_->currentIndex() == 2))
                    showNotice(
                        QString("模型结构、层级树与参数表已缓存 · %1").arg(QFileInfo(path).fileName()));
            });
    connect(modelViewer_, &ModelViewer::loadFailed, this,
            [this](const QString &error)
            {
                if (pages_->currentIndex() == 2)
                    showNotice(error, true);
            });
    auto *note = text("使用模型库中全局选中的模型，三种视图共用后台解析缓存。结构图可缩放与平移；层级树和参数"
                      "表支持搜索与详情查看。",
                      "tiny");
    note->setWordWrap(true);
    layout->addWidget(note);
    return page;
}

void MainWindow::displayModelStructure()
{
    const QString &path = modelPath_;
    structureModelName_->setText(path.isEmpty() ? "在模型库中选择模型" : QFileInfo(path).fileName());
    structureModelName_->setToolTip(path);
    structureModelMeta_->setText(path.isEmpty() ? "检测工作台与模型显示使用同一个全局模型。"
                                                : QFileInfo(path).suffix().toUpper() + " · " + path);
    structureModelMeta_->setToolTip(path);
    structureModelMeta_->setWordWrap(true);
    structureHint_->setVisible(!path.isEmpty() &&
                               QFileInfo(path).suffix().compare("onnx", Qt::CaseInsensitive) != 0);
    if (modelViewer_->modelPath() != path || modelViewer_->state() == ModelViewer::State::Empty)
        modelViewer_->openModel(path);
}

QWidget *MainWindow::buildHistory()
{
    auto *page = new QWidget;
    auto *l = new QVBoxLayout(page);
    l->setContentsMargins(0, 0, 0, 0);
    l->setSpacing(16);
    auto *row = new QHBoxLayout;
    row->addWidget(text("最近 200 次图片推理与视频任务", "sectionTitle"), 1);
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
    connect(gpuCheckButton_, &QPushButton::clicked, this, &MainWindow::checkGpuEnvironment);
    connect(gpuPrepareButton_, &QPushButton::clicked, this, &MainWindow::prepareGpuEnvironment);
    connect(gpuCancelButton_, &QPushButton::clicked, this, &MainWindow::cancelGpuPreparation);
    layout->addWidget(environment);
    auto *examples = card();
    auto *exampleLayout = new QVBoxLayout(examples);
    exampleLayout->setContentsMargins(24, 22, 24, 22);
    exampleLayout->setSpacing(12);
    exampleLayout->addWidget(text("运行示例", "modelName"));
    auto *exampleHint =
        text("选择本地 ONNX 或 PT 示例，体验真实 YOLO 检测。运行后自动返回检测工作台。", "muted");
    exampleHint->setWordWrap(true);
    exampleLayout->addWidget(exampleHint);
    exampleLayout->addWidget(demoButton_, 0, Qt::AlignLeft);
    layout->addWidget(examples);
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

void MainWindow::updateInputPreview()
{
    if (busy_ || !canvas_)
        return;
    const auto mode = currentConfig().colorMode;
    for (int index = 0; index < queue_->count(); ++index)
    {
        auto *item = queue_->item(index);
        const QImage thumbnail = item->data(Qt::UserRole + 1).value<QImage>();
        if (!thumbnail.isNull())
            item->setIcon(QIcon(QPixmap::fromImage(vision::inputPreviewImage(thumbnail, mode))));
    }
    auto preview = canvas_->result();
    if (preview.image.isNull())
        return;
    if (preview.originalImage.isNull())
        preview.originalImage = preview.image;
    preview.image = vision::inputPreviewImage(preview.originalImage, mode);
    preview.predictions.clear();
    lastResult_ = {};
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
        worker_->requestStopRecording();
    }
    else
    {
        recordingRequested_ = true;
        worker_->requestStartRecording();
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
    auto *l = new QVBoxLayout(f);
    l->setContentsMargins(32, 22, 32, 22);
    auto *html = new QTextBrowser;
    html->setOpenExternalLinks(true);
    html->setHtml(R"(
    <style>h1{color:#F5F5F5;font-size:28px}h2{color:#60CDFF;font-size:16px;margin-top:26px}p,li{line-height:1.8;color:#D2D2D2;font-size:13px}code{color:#D8ECF5}a{color:#60CDFF}</style>
    <h1>让你的视觉模型，真正运行起来。</h1><p>Vision Studio 是一个原生 C++ / Qt 桌面工作台。模型加载、图像处理与推理均在本机完成。</p>
    <h2>01 / 开始你的第一次检测</h2><p>“更多”页的“运行示例”可选择 ONNX / PT 示例。使用自己的模型时，导入 ONNX 或 PT，然后选择图片、文件夹、视频或摄像头，点击“开始检测”。PT 使用本机独立 PyTorch 环境直接推理，无需手动导出。</p>
    <h2>模型库与结构显示</h2><p>在“模型库”点击模型即可全局选用，工作台检测与模型显示使用同一模型。选择后在后台预加载，结构图、层级树、参数表共用一次解析缓存；进入或离开页面、切换展示模式不重复加载。结构图可以缩放、平移并点击节点查看输入输出及参数；层级树可以展开模块，参数表可以查看张量名称、类型与形状，两者支持搜索和详情。普通 PT 的树表示模块包含关系，只有权重的文件按参数名称分组，不补造计算连接。无法解析的文件显示文字说明，可点击“重试加载”。非 ONNX 文件显示：想看完整结构，建使用导出的 ONNX。</p>
    <h2>PT 模型</h2><p>Ultralytics YOLO 的 PT 检查点会自动读取任务和类别名称，预处理由原生后端执行。支持的旧版 YOLOv5 权重使用随附的本地兼容模块。只包含 state_dict 的任意 PT 文件无法单独重建网络，需要原始模型架构。分割、姿态和旋转框输出暂不支持。</p>
    <h2>02 / 正确匹配模型</h2><p>YOLOv5：原始输出 <code>[1,N,5+C]</code>，包含 objectness。YOLOv8 / YOLO11：原始输出 <code>[1,4+C,N]</code>。模型应为 batch=1、固定正方形输入、FP32、不包含 NMS。输入尺寸必须与导出模型一致。分割、姿态、旋转框和端到端输出暂不支持。</p>
    <p>默认 640 px、RGB、1/255 缩放、零均值，适合常见 YOLO 模型。自定义检测模型必须导入数量匹配的 UTF-8 标签文本，每行一个名称，并保持训练时类别顺序。未导入标签时按 COCO 80 类解释检测输出。分类模型未配置标签时显示数字类别。</p>
    <h2>03 / 调整结果与预处理</h2><p>置信度越高，保留的目标越少；NMS IoU 控制同类重叠框的抑制。不同类别独立执行 NMS。检测输入使用 letterbox 保持比例，并将框映射回原图。分类使用正方形缩放和 top-5 输出，可在“预处理设置”调整缩放和均值；本版不提供逐通道标准差除法。</p>
    <p>输入源区域可直接选择 RGB、BGR 或灰度。灰度转换为图像明度：单通道模型直接接收灰度，三通道模型接收三个相同的灰度通道。ONNX 灰度输入统一使用“灰度均值”，G/B 均值不再分别参与计算；PT 可选 RGB 或灰度，归一化和零均值由原生后端执行。灰度模式同步显示灰度预览，模型载入后会显示实际输入通道数。</p>
    <h2>04 / 计算设备与 GPU 准备</h2><p>工作台可选择“自动 · 优先 NVIDIA GPU”“CPU”或“NVIDIA GPU”，并指定 GPU 编号。自动模式在 GPU 不可用时说明原因并使用 CPU；显式 GPU 模式无法使用时会报错。运行状态与导出记录显示实际执行设备，不能仅凭选择框判断。切换设备不改变灰度、颜色或双目设置，运行期间设备选择锁定。</p>
    <p>首次使用 GPU 请到“更多”检查并点击“准备 GPU 支持”。需要一次联网下载约 4 GB，建议至少预留 20 GB 空间；依赖保存在个人数据目录，CPU 环境保留，不安装或修改显卡驱动。准备过程中可查看进度与日志、取消本窗口启动的任务。完成后重新检查环境，再运行模型验证。</p>
    <h2>05 / 批量、视频与摄像头</h2><p>文件夹模式扫描当前目录内的常见图片格式，逐张推理。视频与摄像头连续处理每帧；实际速度取决于所选设备与模型，界面预览限流。点击“停止运行”结束任务。“更多”页可导出当前帧。视频或摄像头检测时可点击“开始录制”，再次点击结束；“录制视频”页可查看、播放及导出带检测标注的录像。</p>
    <p>单设备左右并排（SBS）的双目摄像头或视频可选“完整画面”“双目左目”“双目右目”。选择单目时，沿水平中线裁出所选眼，再执行预览、推理和导出；结果坐标以该单目图像为基准，JSON 同时记录原始双目画面尺寸。普通图片始终使用完整图像。运行中画面选择锁定，停止后可切换。</p>
    <h2>05 / 保存你的洞察</h2><p>导出结果会生成标注 PNG、包含原始像素坐标的 JSON，以及可用于表格分析的 CSV。启用自动保存后，批量任务为每张图片保存结果，流式任务在结束时保存最后一帧。设置和运行记录自动保存在用户数据目录。</p>
    <h2>键盘与画布</h2><p><code>Ctrl+O</code> 添加图片　<code>Ctrl+M</code> 导入模型　<code>Ctrl+R</code> 开始　<code>Ctrl+E</code> 导出　<code>Esc</code> 停止<br>滚轮缩放，拖动画布平移，双击适应画布，点击检测框或结果列表定位目标。</p>
    <h2>运行环境与兼容性</h2><p>界面为本机 C++ / Qt 6.8.3。ONNX 的 CPU 路径使用 OpenCV DNN，GPU 路径使用 ONNX Runtime CUDA；PT 使用本地 PyTorch 环境。兼容性取决于实际后端与模型算子。CPU 与 Netron 环境位于 runtime，GPU 支持安装在个人数据目录的 gpu-runtime。结构显示使用内嵌 Qt WebEngine 与本机 Netron，与推理设备选择无关。依赖准备完成后可离线运行模型。</p>
    <p><a href="https://github.com/ultralytics/yolov5">YOLOv5 官方项目</a>　<a href="https://docs.ultralytics.com/modes/export/">Ultralytics ONNX 导出文档</a></p>
    <h2>开源许可</h2><p>Copyright © 2026 misaka_ning。Vision Studio 应用代码按 GNU AGPL v3 发布，您可以依据许可证复制、修改和再分发。程序不提供任何担保。完整许可证、第三方许可和对应源码资料随发行版本提供。</p>
    <p><a href="https://www.gnu.org/licenses/agpl-3.0.html">GNU AGPL v3 完整许可证</a>　联系维护者：1468549029@qq.com</p>
    )");
    l->addWidget(html);
    return f;
}

void MainWindow::connectWorker()
{
    workerThread_ = new QThread(this);
    worker_ = new vision::InferenceWorker;
    worker_->moveToThread(workerThread_);
    connect(workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(this, &MainWindow::startRequested, worker_, &vision::InferenceWorker::run, Qt::QueuedConnection);
    connect(worker_, &vision::InferenceWorker::resultReady, this, &MainWindow::onResult);
    connect(worker_, &vision::InferenceWorker::modelReady, this,
            [this](const QString &backend, const vision::ModelConfig &config)
            {
                lastConfig_ = config;
                modelInputChannels_ = config.inputChannels;
                updateModelMeta("已加载");
                if (isPtModel(modelPath_))
                {
                    nativeLabels_ = config.labels;
                    taskBox_->setCurrentIndex(static_cast<int>(config.task));
                }
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
    connect(worker_, &vision::InferenceWorker::status, this, [this](const QString &s) { showNotice(s); });
    connect(worker_, &vision::InferenceWorker::progress, this,
            [this](int done, int total)
            {
                completed_ = done;
                if (total > 0)
                {
                    progress_->setRange(0, total);
                    progress_->setValue(done);
                }
                else
                    progress_->setRange(0, 0);
            });
    connect(worker_, &vision::InferenceWorker::failed, this,
            [this](const QString &s)
            {
                failed_ = true;
                lastError_ = s;
                resultInfo_->setText("推理失败 · 请检查模型与输入配置。");
                showNotice(s, true);
                if (smokeDir_.isEmpty() && !closing_)
                    QMessageBox::warning(this, "推理未完成", s);
            });
    connect(worker_, &vision::InferenceWorker::finished, this, &MainWindow::onFinished);
    connect(worker_, &vision::InferenceWorker::recordingStarted, this,
            [this](const QString &)
            {
                recordingActive_ = true;
                recordingElapsed_.restart();
                recordingClock_->start();
                if (!recordingRequested_)
                {
                    recordingStopping_ = true;
                    worker_->requestStopRecording();
                }
                updateRecordingUi();
            });
    connect(worker_, &vision::InferenceWorker::recordingFinished, this,
            [this](const QString &path, qint64 frames, double)
            {
                recordingRequested_ = recordingActive_ = recordingStopping_ = false;
                recordingClock_->stop();
                refreshRecordings();
                updateRecordingUi();
                showNotice(QString("录制已保存 · %1 帧 · %2").arg(frames).arg(QFileInfo(path).fileName()));
            });
    connect(worker_, &vision::InferenceWorker::recordingFailed, this,
            [this](const QString &error)
            {
                recordingRequested_ = recordingActive_ = recordingStopping_ = false;
                recordingClock_->stop();
                updateRecordingUi();
                showNotice("录制未保存：" + error, true);
            });
    workerThread_->start();
}

void MainWindow::chooseImages()
{
    const QStringList p =
        QFileDialog::getOpenFileNames(this, "选择待检测图片", projectRoot_,
                                      "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;全部文件 (*)");
    if (!p.isEmpty())
        addFiles(p);
}
void MainWindow::chooseFolder()
{
    const QString p = QFileDialog::getExistingDirectory(this, "选择图片文件夹", projectRoot_);
    if (p.isEmpty())
        return;
    QStringList files;
    const QDir d(p);
    for (const QFileInfo &f : d.entryInfoList(QDir::Files, QDir::Name))
    {
        const QString ext = f.suffix().toLower();
        if (QStringList{"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}.contains(ext))
            files.append(f.absoluteFilePath());
    }
    if (files.isEmpty())
    {
        showNotice("该文件夹中没有支持的图片。", true);
        return;
    }
    files_.clear();
    queue_->clear();
    addFiles(files);
}
void MainWindow::chooseVideo()
{
    const QString p = QFileDialog::getOpenFileName(
        this, "选择视频", projectRoot_, "视频 (*.mp4 *.avi *.mkv *.mov *.webm *.m4v);;全部文件 (*)");
    if (p.isEmpty())
        return;
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
    cameraIndex_->setVisible(sourceKind_ == vision::SourceKind::Camera);
    const bool streaming = sourceKind_ != vision::SourceKind::Images;
    stereoLabel_->setVisible(streaming);
    stereoView_->setVisible(streaming);
    stereoView_->setEnabled(streaming && !busy_);
    const QString view = stereoName(stereoMode(stereoView_->currentData().toString()));
    if (sourceKind_ == vision::SourceKind::Images)
        sourceLabel_->setText(QString("图片输入 · %1 个文件").arg(files_.size()));
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
        resultInfo_->setText("输入已就绪，点击开始检测。");
    }
    updateRecordingUi();
}

void MainWindow::addFiles(const QStringList &input)
{
    if (sourceKind_ != vision::SourceKind::Images)
    {
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
        const QImage img = readImage(absolute);
        if (img.isNull())
        {
            showNotice("无法读取图片：" + QFileInfo(p).fileName(), true);
            continue;
        }
        if (first < 0)
            first = files_.size();
        files_.append(absolute);
        const QImage thumbnail = img.scaled(64, 46, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        auto *item = new QListWidgetItem(
            QIcon(QPixmap::fromImage(vision::inputPreviewImage(thumbnail, currentConfig().colorMode))),
            QFileInfo(p).fileName());
        item->setData(Qt::UserRole + 1, thumbnail);
        item->setToolTip(absolute);
        queue_->addItem(item);
    }
    if (first >= 0)
        queue_->setCurrentRow(first);
    updateSourceUi();
}
void MainWindow::importModel()
{
    const QString p =
        QFileDialog::getOpenFileName(this, "导入视觉模型", projectRoot_ + "/models",
                                     "视觉模型 (*.onnx *.pt);;ONNX 模型 (*.onnx);;PyTorch 模型 (*.pt)");
    if (!p.isEmpty())
        setModel(p);
}
void MainWindow::setModel(const QString &path)
{
    if (busy_)
        return;
    if (path.isEmpty())
    {
        actualDeviceKnown_ = false;
        actualDeviceNotice_.clear();
        modelPath_.clear();
        modelInputChannels_ = 0;
        nativeLabels_.clear();
        modelName_->setText("尚未选择模型");
        modelMeta_->setText("请在模型库中选择一个模型。");
        modelName_->setToolTip({});
        updateTaskUi();
        updateInputPreview();
        displayModelStructure();
        persist();
        return;
    }
    QFileInfo f(path);
    if (!f.isFile())
    {
        showNotice("模型文件不存在：" + path, true);
        return;
    }
    if (modelPath_ == f.absoluteFilePath())
        return;
    const bool convertedBgr =
        isPtModel(f.absoluteFilePath()) && inputColorMode_->currentData().toString() == "bgr";
    modelPath_ = f.absoluteFilePath();
    actualDeviceKnown_ = false;
    actualDeviceNotice_.clear();
    modelInputChannels_ = 0;
    nativeLabels_.clear();
    modelName_->setText(f.fileName());
    updateModelMeta("已选择");
    modelName_->setToolTip(modelPath_);
    if (!models_.contains(modelPath_))
        models_.append(modelPath_);
    const QString n = f.fileName().toLower();
    if (n.contains("yolov5"))
        taskBox_->setCurrentIndex(0);
    else if (n.contains("yolov8") || n.contains("yolo11"))
        taskBox_->setCurrentIndex(1);
    updateTaskUi();
    updateInputPreview();
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
    showNotice("已导入 " + QString::number(labels_.size()) + " 个类别标签。");
}
vision::ModelConfig MainWindow::currentConfig() const
{
    vision::ModelConfig c;
    c.modelPath = modelPath_;
    c.device = computeMode(computeDevice_->currentData().toString());
    c.deviceIndex = gpuDeviceIndex_->currentData().toInt();
    c.task = static_cast<vision::ModelTask>(taskBox_->currentIndex());
    c.labels = labels_.isEmpty() ? ((isPtModel(modelPath_) || c.task == vision::ModelTask::Classification)
                                        ? QStringList{}
                                        : vision::cocoLabels())
                                 : labels_;
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
    return c;
}
void MainWindow::startInference()
{
    if (busy_ || gpuSetupProcess_)
        return;
    if (modelPath_.isEmpty())
    {
        showNotice("请先导入 ONNX 或 PT 模型。", true);
        return;
    }
    if (sourceKind_ == vision::SourceKind::Images && files_.isEmpty())
    {
        showNotice("请先添加图片或选择其他输入源。", true);
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
    req.files = sourceKind_ == vision::SourceKind::Video ? QStringList{streamPath_} : files_;
    failed_ = false;
    inferenceStopping_ = false;
    lastError_.clear();
    actualDeviceKnown_ = false;
    actualDeviceNotice_.clear();
    completed_ = 0;
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
    worker_->prepareRecording(dataRoot_ + "/recordings");
    worker_->prepare();
    emit startRequested(req);
}
void MainWindow::stopInference()
{
    if (!busy_)
        return;
    inferenceStopping_ = true;
    recordingRequested_ = false;
    recordingStopping_ = recordingActive_;
    worker_->requestStop();
    updateRecordingUi();
    stopButton_->setEnabled(false);
    showNotice("正在停止，请等待当前帧处理完成…");
}
void MainWindow::setBusy(bool busy)
{
    busy_ = busy;
    if (!busy)
        inferenceStopping_ = false;
    for (auto *c : lockedControls_)
        c->setEnabled(!busy);
    demoButton_->setEnabled(!busy);
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
    if (busy)
    {
        progress_->setRange(0, 0);
    }
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
                                 : (classification ? "类别标签 · 数字类别" : "类别标签 · 默认 COCO 80"));
    else
        labelButton_->setText(QString("类别标签 · %1 个").arg(labels_.size()));
    updateDeviceUi();
}
void MainWindow::setComputeDevice(vision::ComputeDevice device, int index)
{
    index = qBound(0, index, 63);
    if (busy_ || gpuSetupProcess_)
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
    const bool installing = gpuSetupProcess_ != nullptr;
    const bool checking = gpuProbeProcess_ != nullptr;
    const auto requested = computeMode(computeDevice_->currentData().toString());
    computeDevice_->setEnabled(!busy_ && !installing);
    gpuDeviceIndex_->setEnabled(!busy_ && !installing && requested != vision::ComputeDevice::CPU);
    const bool showIndex = requested == vision::ComputeDevice::CUDA ||
                           (requested == vision::ComputeDevice::Auto && gpuDeviceIndex_->count() > 1);
    gpuDeviceLabel_->setVisible(showIndex);
    gpuDeviceIndex_->setVisible(showIndex);
    const bool ready = gpuEnvironment_.value("ok").toBool() && gpuEnvironment_.value("prepared").toBool() &&
                       gpuEnvironment_.value("cuda_available").toBool();
    const QString reason = gpuEnvironment_.value("reason").toString();
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
    deviceHint_->setText(hint);
    deviceHint_->setToolTip(reason.isEmpty() ? hint : reason);
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
        gpuCancelButton_->setEnabled(installing && !gpuSetupCancelling_);
    }
    if (runButton_)
        runButton_->setEnabled(!installing);
    if (demoButton_)
        demoButton_->setEnabled(!busy_ && !installing);
}

QString MainWindow::gpuLauncherPython() const
{
    const QString override = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    const QStringList candidates{override, projectRoot_ + "/runtime/bin/python", "/usr/bin/python3.10",
                                 QStandardPaths::findExecutable("python3")};
    for (const QString &candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isExecutable())
            return candidate;
    return {};
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
    if (busy_ || gpuSetupProcess_ || gpuProbeProcess_)
        return;
    const QString python = gpuLauncherPython();
    const QString script = vision::gpuScriptPath("gpu_probe.py");
    if (python.isEmpty() || !QFileInfo(script).isFile())
    {
        gpuProbeKnown_ = true;
        gpuEnvironment_ = {{"ok", false}, {"reason", "未找到 GPU 检查工具或 Python；CPU 仍可使用。"}};
        gpuEnvironmentStatus_->setText(gpuEnvironment_.value("reason").toString());
        updateDeviceUi();
        return;
    }
    auto *process = new QProcess(this);
    process->setObjectName("gpuProbeProcess");
    gpuProbeProcess_ = process;
    gpuProbeOutput_.clear();
    gpuProbeErrors_.clear();
    gpuEnvironmentStatus_->setText("正在检查显卡与本地 GPU 推理环境…");
    updateDeviceUi();
    connect(process, &QProcess::readyReadStandardOutput, this,
            [this, process]
            {
                if (gpuProbeProcess_ == process)
                {
                    gpuProbeOutput_ += process->readAllStandardOutput();
                    if (gpuProbeOutput_.size() > 512 * 1024)
                        process->kill();
                }
            });
    connect(process, &QProcess::readyReadStandardError, this,
            [this, process]
            {
                if (gpuProbeProcess_ == process)
                    gpuProbeErrors_ = (gpuProbeErrors_ + process->readAllStandardError()).right(16000);
            });
    const auto finish = [this, process](bool success)
    {
        if (gpuProbeProcess_ != process)
            return;
        gpuProbeOutput_ += process->readAllStandardOutput();
        const QJsonObject response = QJsonDocument::fromJson(gpuProbeOutput_).object();
        gpuProbeProcess_ = nullptr;
        gpuProbeKnown_ = true;
        gpuEnvironment_ =
            success && response.contains("cuda_available")
                ? response
                : QJsonObject{{"ok", false},
                              {"reason", QString("GPU 环境检查未完成：%1")
                                             .arg(gpuProbeErrors_.isEmpty()
                                                      ? process->errorString()
                                                      : QString::fromUtf8(gpuProbeErrors_).left(500))}};
        const bool ready = gpuEnvironment_.value("ok").toBool() &&
                           gpuEnvironment_.value("prepared").toBool() &&
                           gpuEnvironment_.value("cuda_available").toBool();
        const int selected = gpuDeviceIndex_->currentData().toInt();
        {
            const QSignalBlocker blocker(gpuDeviceIndex_);
            gpuDeviceIndex_->clear();
            QStringList hardware;
            for (const QJsonValue &value : gpuEnvironment_.value("devices").toArray())
            {
                const QJsonObject device = value.toObject();
                const int index = device.value("index").toInt();
                const QString name = device.value("name").toString().left(120);
                gpuDeviceIndex_->addItem(QString("GPU %1 · %2").arg(index).arg(name), index);
                hardware << QString("GPU %1 · %2 · %3 GB")
                                .arg(index)
                                .arg(name)
                                .arg(device.value("total_memory_mb").toDouble() / 1024, 0, 'f', 1);
            }
            if (gpuDeviceIndex_->findData(selected) < 0)
                gpuDeviceIndex_->addItem(QString("GPU %1 · 当前未检测到").arg(selected), selected);
            gpuDeviceIndex_->setCurrentIndex(gpuDeviceIndex_->findData(selected));
            const QString reason = gpuEnvironment_.value("reason").toString();
            gpuEnvironmentStatus_->setText(
                (ready ? QString("GPU 推理环境已就绪") : QString("GPU 推理环境尚未就绪")) +
                (hardware.isEmpty() ? QString("\n未检测到 NVIDIA 显卡") : "\n" + hardware.join("\n")) +
                (reason.isEmpty() ? QString() : "\n" + reason));
        }
        gpuEnvironmentStatus_->setToolTip(vision::gpuRuntimeDirectory());
        process->deleteLater();
        updateDeviceUi();
    };
    connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [finish](int code, QProcess::ExitStatus status)
            { finish(code == 0 && status == QProcess::NormalExit); });
    connect(process, &QProcess::errorOccurred, this,
            [finish](QProcess::ProcessError error)
            {
                if (error == QProcess::FailedToStart)
                    finish(false);
            });
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    process->setProcessEnvironment(environment);
    process->start(python, {script, "--runtime-dir", vision::gpuRuntimeDirectory()});
    QTimer::singleShot(45000, process,
                       [this, process]
                       {
                           if (gpuProbeProcess_ == process && process->state() != QProcess::NotRunning)
                           {
                               gpuProbeErrors_ = "检查超时，请重试；CPU 仍可使用。";
                               process->kill();
                           }
                       });
}

void MainWindow::readGpuSetupOutput()
{
    if (!gpuSetupProcess_)
        return;
    gpuSetupOutput_ += gpuSetupProcess_->readAllStandardOutput();
    while (gpuSetupOutput_.contains('\n'))
    {
        const int newline = gpuSetupOutput_.indexOf('\n');
        const QByteArray line = gpuSetupOutput_.left(newline).trimmed();
        gpuSetupOutput_.remove(0, newline + 1);
        const QJsonObject event = QJsonDocument::fromJson(line).object();
        if (event.isEmpty())
            appendGpuLog(QString::fromUtf8(line));
        else
        {
            const QString message = event.value("message").toString();
            appendGpuLog(message.isEmpty() ? event.value("event").toString() : message);
            if (!message.isEmpty())
                gpuEnvironmentStatus_->setText(message.left(600));
            const QJsonValue percent =
                event.value("percent").isDouble() ? event.value("percent") : event.value("progress");
            if (percent.isDouble())
            {
                gpuSetupProgress_->setRange(0, 100);
                gpuSetupProgress_->setValue(qBound(0, percent.toInt(), 100));
            }
        }
    }
    if (gpuSetupOutput_.size() > 256 * 1024)
    {
        appendGpuLog(QString::fromUtf8(gpuSetupOutput_.left(16000)));
        gpuSetupOutput_.clear();
    }
}

void MainWindow::prepareGpuEnvironment()
{
    if (busy_ || gpuSetupProcess_ || gpuProbeProcess_)
        return;
    const QString python = gpuLauncherPython(), script = vision::gpuScriptPath("gpu_setup.py");
    if (python.isEmpty() || !QFileInfo(script).isFile())
    {
        gpuEnvironmentStatus_->setText("未找到 GPU 准备工具或 Python。CPU 仍可使用。");
        return;
    }
    auto *process = new QProcess(this);
    process->setObjectName("gpuSetupProcess");
    gpuSetupProcess_ = process;
    gpuSetupCancelling_ = false;
    gpuSetupOutput_.clear();
    gpuLog_->clear();
    appendGpuLog("GPU 依赖将安装到：" + vision::gpuRuntimeDirectory());
    gpuEnvironmentStatus_->setText(
        "正在准备 GPU 支持。首次下载约 4 GB，耗时取决于网络；CPU 环境和显卡驱动保持原样。");
    gpuSetupProgress_->setRange(0, 0);
    gpuSetupProgress_->show();
    updateDeviceUi();
    connect(process, &QProcess::readyReadStandardOutput, this, &MainWindow::readGpuSetupOutput);
    connect(process, &QProcess::readyReadStandardError, this,
            [this, process]
            {
                if (gpuSetupProcess_ == process)
                    appendGpuLog(QString::fromUtf8(process->readAllStandardError()));
            });
    const auto finish = [this, process](bool success)
    {
        if (gpuSetupProcess_ != process)
            return;
        readGpuSetupOutput();
        if (!gpuSetupOutput_.trimmed().isEmpty())
            appendGpuLog(QString::fromUtf8(gpuSetupOutput_));
        const bool cancelled = gpuSetupCancelling_;
        gpuSetupProcess_ = nullptr;
        gpuSetupCancelling_ = false;
        gpuSetupProgress_->hide();
        const QString message = cancelled ? QStringLiteral("GPU 准备已取消，CPU 仍可使用。")
                                : success
                                    ? QStringLiteral("GPU 准备完成，正在重新检查环境…")
                                    : QStringLiteral("GPU 准备未完成。请查看日志后重试；CPU 仍可使用。");
        gpuEnvironmentStatus_->setText(message);
        appendGpuLog(message);
        process->deleteLater();
        updateDeviceUi();
        if (success && !cancelled)
            QTimer::singleShot(0, this, &MainWindow::checkGpuEnvironment);
        if (closing_)
            QTimer::singleShot(0, this, &QWidget::close);
    };
    connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [finish](int code, QProcess::ExitStatus status)
            { finish(code == 0 && status == QProcess::NormalExit); });
    connect(process, &QProcess::errorOccurred, this,
            [finish](QProcess::ProcessError error)
            {
                if (error == QProcess::FailedToStart)
                    finish(false);
            });
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    process->setProcessEnvironment(environment);
    process->start(python, {"-u", script, "--runtime-dir", vision::gpuRuntimeDirectory()});
}

void MainWindow::cancelGpuPreparation()
{
    if (!gpuSetupProcess_ || gpuSetupCancelling_)
        return;
    gpuSetupCancelling_ = true;
    gpuEnvironmentStatus_->setText("正在取消本窗口启动的 GPU 准备任务…");
    appendGpuLog("正在取消 GPU 准备任务。");
    const QPointer<QProcess> process(gpuSetupProcess_);
    process->terminate();
    updateDeviceUi();
    QTimer::singleShot(5000, this,
                       [process]
                       {
                           if (process && process->state() != QProcess::NotRunning)
                               process->kill();
                       });
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
void MainWindow::onResult(const vision::InferenceResult &r)
{
    lastResult_ = r;
    actualDeviceKnown_ = true;
    actualDevice_ = r.device;
    actualDeviceIndex_ = r.deviceIndex;
    actualDeviceName_ = r.deviceName;
    actualDeviceNotice_ = r.deviceNotice;
    actualBackend_ = r.backend;
    updateDeviceUi();
    updateModelMeta("已验证");
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
    latencyMetric_->setText(QString::number(r.inferenceMs, 'f', 1));
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
                             .arg(taskName(r.task))
                             .arg(r.predictions.size())
                             .arg(r.task == vision::ModelTask::Classification ? "分类结果" : "目标")
                             .arg(r.totalMs, 0, 'f', 1));
    QString deviceDescription;
    if (r.device == vision::ComputeDevice::CPU)
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
    predictionTable_->setRowCount(r.predictions.size());
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
        auto *id = new QTableWidgetItem(QString::number(i + 1).rightJustified(2, '0'));
        id->setTextAlignment(Qt::AlignCenter);
        id->setForeground(QColor("#A5A5A5"));
        predictionTable_->setItem(i, 2, id);
    }
    if (sourceKind_ == vision::SourceKind::Images)
    {
        recordResult(r);
        const int row = files_.indexOf(r.source);
        if (row >= 0)
        {
            queue_->setCurrentRow(row);
            queue_->item(row)->setForeground(QColor("#60CDFF"));
        }
        if (autoExport_->isChecked())
        {
            QString error;
            if (!writeResult(r, exportDir_, &error))
            {
                failed_ = true;
                worker_->requestStop();
                showNotice("自动导出失败：" + error, true);
            }
        }
    }
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
                showNotice("自动导出失败：" + error, true);
            }
        }
    }
    setBusy(false);
    if (!failed_)
        showNotice(cancelled ? "任务已停止 · 当前结果可导出"
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
    const QString model = projectRoot_ + "/models/yolov5n.onnx", image = projectRoot_ + "/assets/bus.jpg";
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
    files_.clear();
    queue_->clear();
    sourceKind_ = vision::SourceKind::Images;
    addFiles({image});
    selectRoute(0);
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
    const QString model = projectRoot_ + "/models/yolov8n.pt", image = projectRoot_ + "/assets/bus.jpg";
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
    files_.clear();
    queue_->clear();
    sourceKind_ = vision::SourceKind::Images;
    addFiles({image});
    selectRoute(0);
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
    selectRoute(2);
}

void MainWindow::runModelSmoke(const QString &dir, const QString &model)
{
    const QString output = QFileInfo(dir).absoluteFilePath();
    if (!QDir().mkpath(output))
    {
        qCritical().noquote() << "无法创建模型显示自检目录：" << output;
        QApplication::exit(2);
        return;
    }
    const auto completed = std::make_shared<bool>(false);
    const auto capturing = std::make_shared<bool>(false);
    const auto views = std::make_shared<QJsonObject>();
    const auto cachePreserved = std::make_shared<bool>(false);
    const auto writeReport =
        [this, output, completed, views, cachePreserved](bool success, const QString &error)
    {
        if (*completed)
            return;
        *completed = true;
        if (!success)
            saveScreenshot(output + "/model-display.png");
        const QJsonObject report{{"success", success},
                                 {"error", error},
                                 {"application", "Vision Studio"},
                                 {"version", QCoreApplication::applicationVersion()},
                                 {"qt", qVersion()},
                                 {"netron", "9.3.1"},
                                 {"model", modelViewer_->modelPath()},
                                 {"nodes", modelViewer_->graphNodeCount()},
                                 {"hierarchy_items", modelViewer_->hierarchyItemCount()},
                                 {"parameter_rows", modelViewer_->parameterCount()},
                                 {"display_modes", *views},
                                 {"cache_preserved", *cachePreserved},
                                 {"onnx_hint_visible", structureHint_->isVisible()},
                                 {"onnx_hint", structureHint_->text()}};
        QString writeError;
        const bool saved =
            atomicWrite(output + "/model-display-report.json", QJsonDocument(report).toJson(), &writeError);
        qInfo().noquote() << QJsonDocument(report).toJson(QJsonDocument::Compact);
        QApplication::exit(success && saved ? 0 : 2);
    };
    const auto captureViews = [this, output, capturing, views, cachePreserved, writeReport]
    {
        if (*capturing)
            return;
        *capturing = true;
        if (modelViewer_->state() != ModelViewer::State::Ready || modelViewer_->graphNodeCount() <= 0)
        {
            writeReport(false, "模型没有生成可显示的结构。");
            return;
        }
        const QPointer<QObject> browser = modelViewer_->findChild<QObject *>("netronWebView");
        const QPointer<QProcess> service = modelViewer_->findChild<QProcess *>();
        const qint64 processId = service ? service->processId() : 0;
        const QString path = modelViewer_->modelPath();
        const int items = modelViewer_->hierarchyItemCount();
        const int parameters = modelViewer_->parameterCount();
        modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Graph);
        QTimer::singleShot(
            300, this,
            [this, output, views, cachePreserved, writeReport, browser, service, processId, path, items,
             parameters]
            {
                saveScreenshot(output + "/model-display.png");
                views->insert("graph", modelViewer_->displayMode() == ModelViewer::DisplayMode::Graph);
                modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Hierarchy);
                QTimer::singleShot(
                    200, this,
                    [this, output, views, cachePreserved, writeReport, browser, service, processId, path,
                     items, parameters]
                    {
                        saveScreenshot(output + "/model-hierarchy.png");
                        views->insert("hierarchy",
                                      modelViewer_->displayMode() == ModelViewer::DisplayMode::Hierarchy);
                        modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Parameters);
                        QTimer::singleShot(
                            200, this,
                            [this, output, views, cachePreserved, writeReport, browser, service, processId,
                             path, items, parameters]
                            {
                                saveScreenshot(output + "/model-parameters.png");
                                views->insert("parameters", modelViewer_->displayMode() ==
                                                                ModelViewer::DisplayMode::Parameters);
                                modelViewer_->setDisplayMode(ModelViewer::DisplayMode::Graph);
                                *cachePreserved =
                                    browser && service && processId > 0 &&
                                    modelViewer_->findChild<QObject *>("netronWebView") == browser &&
                                    modelViewer_->findChild<QProcess *>() == service &&
                                    service->processId() == processId &&
                                    modelViewer_->state() == ModelViewer::State::Ready &&
                                    modelViewer_->modelPath() == path &&
                                    modelViewer_->hierarchyItemCount() == items &&
                                    modelViewer_->parameterCount() == parameters;
                                const bool success =
                                    *cachePreserved && items > 0 && views->value("graph").toBool() &&
                                    views->value("hierarchy").toBool() && views->value("parameters").toBool();
                                writeReport(success,
                                            success ? QString() : "模型视图切换或解析缓存自检失败。");
                            });
                    });
            });
    };
    connect(
        modelViewer_, &ModelViewer::modelLoaded, this, [this, captureViews](const QString &)
        { QTimer::singleShot(300, this, captureViews); }, Qt::SingleShotConnection);
    connect(
        modelViewer_, &ModelViewer::loadFailed, this,
        [writeReport](const QString &error) { writeReport(false, error); }, Qt::SingleShotConnection);
    showModelStructure(model);
    if (modelViewer_->state() == ModelViewer::State::Ready)
        QTimer::singleShot(300, this, captureViews);
    else if (modelViewer_->state() == ModelViewer::State::Error)
        QTimer::singleShot(0, this, [this, writeReport] { writeReport(false, modelViewer_->errorString()); });
}
void MainWindow::saveScreenshot(const QString &p)
{
    QDir().mkpath(QFileInfo(p).absolutePath());
    if (!grab().save(p))
        qWarning().noquote() << "Cannot save screenshot" << p;
}

bool MainWindow::writeResult(const vision::InferenceResult &r, const QString &dir, QString *error)
{
    if (r.image.isNull() || r.demonstration)
    {
        if (error)
            *error = "没有可导出的真实结果。";
        return false;
    }
    if (!QDir().mkpath(dir))
    {
        if (error)
            *error = "无法创建目录。";
        return false;
    }
    const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz");
    const QString base = dir + "/" + cleanName(r.source) + "-" + stamp +
                         (r.frameNumber > 0 ? QString("-f%1").arg(r.frameNumber) : "");
    // Render from the result being exported; the canvas can be showing another queue item.
    ImageCanvas renderer;
    renderer.setResult(r);
    const QImage image = renderer.annotatedImage();
    QSaveFile png(base + ".png");
    if (!png.open(QIODevice::WriteOnly) || !image.save(&png, "PNG") || !png.commit())
    {
        if (error)
            *error = png.errorString();
        return false;
    }
    QJsonArray predictions;
    QByteArray csv = "class_id,label,confidence,x,y,width,height\n";
    for (const auto &p : r.predictions)
    {
        QJsonObject pred{{"class_id", p.classId},   {"label", p.label}, {"confidence", double(p.confidence)},
                         {"x", p.box.x()},          {"y", p.box.y()},   {"width", p.box.width()},
                         {"height", p.box.height()}};
        predictions.append(pred);
        QString label = p.label;
        label.replace('"', "\"\"");
        if (!label.isEmpty() && QString("=+-@\t\r").contains(label.front()))
            label.prepend('\'');
        csv += QString("%1,\"%2\",%3,%4,%5,%6,%7\n")
                   .arg(p.classId)
                   .arg(label)
                   .arg(p.confidence, 0, 'f', 6)
                   .arg(p.box.x(), 0, 'f', 2)
                   .arg(p.box.y(), 0, 'f', 2)
                   .arg(p.box.width(), 0, 'f', 2)
                   .arg(p.box.height(), 0, 'f', 2)
                   .toUtf8();
    }
    const QJsonObject preprocess{
        {"mode", isPtModel(lastConfig_.modelPath) ? "model_native" : "configured"},
        {"color_mode", lastConfig_.colorMode == vision::InputColorMode::Grayscale
                           ? "grayscale"
                           : (lastConfig_.swapRB ? "rgb" : "bgr")},
        {"swap_rb", lastConfig_.swapRB},
        {"scale", lastConfig_.scale},
        {"mean_rgb", QJsonArray{lastConfig_.meanR, lastConfig_.meanG, lastConfig_.meanB}}};
    const QJsonObject config{{"input_size", lastConfig_.inputSize},
                             {"input_channels", lastConfig_.inputChannels},
                             {"color_mode", lastConfig_.colorMode == vision::InputColorMode::Grayscale
                                                ? "grayscale"
                                                : (lastConfig_.swapRB ? "rgb" : "bgr")},
                             {"confidence_threshold", double(lastConfig_.confidence)},
                             {"nms_iou", double(lastConfig_.iou)},
                             {"preprocess", preprocess}};
    QJsonObject root{
        {"application", "Vision Studio"},
        {"version", QCoreApplication::applicationVersion()},
        {"timestamp", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)},
        {"source", r.source},
        {"model", r.modelName},
        {"backend", r.backend},
        {"requested_device", vision::computeDeviceKey(r.requestedDevice)},
        {"actual_device", vision::computeDeviceKey(r.device)},
        {"device_index", r.deviceIndex},
        {"device_name", r.deviceName},
        {"device_notice", r.deviceNotice},
        {"model_file", lastConfig_.modelPath},
        {"config", config},
        {"task", taskName(r.task)},
        {"width", r.image.width()},
        {"height", r.image.height()},
        {"stereo_view", stereoCode(r.stereoView)},
        {"source_frame_size",
         QJsonObject{
             {"width", r.sourceFrameSize.isEmpty() ? r.image.width() : r.sourceFrameSize.width()},
             {"height", r.sourceFrameSize.isEmpty() ? r.image.height() : r.sourceFrameSize.height()}}},
        {"frame", r.frameNumber},
        {"inference_ms", r.inferenceMs},
        {"total_ms", r.totalMs},
        {"predictions", predictions}};
    return atomicWrite(base + ".json", QJsonDocument(root).toJson(), error) &&
           atomicWrite(base + ".csv", csv, error);
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
void MainWindow::recordResult(const vision::InferenceResult &r)
{
    QJsonObject o{
        {"time", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)},
        {"source", r.source},
        {"model", r.modelName},
        {"backend", r.backend},
        {"requested_device", vision::computeDeviceKey(r.requestedDevice)},
        {"actual_device", vision::computeDeviceKey(r.device)},
        {"device_index", r.deviceIndex},
        {"device_name", r.deviceName},
        {"device_notice", r.deviceNotice},
        {"objects", r.predictions.size()},
        {"inference_ms", r.inferenceMs},
        {"task", taskName(r.task)},
        {"width", r.image.width()},
        {"height", r.image.height()},
        {"stereo_view", stereoCode(r.stereoView)},
        {"source_frame_size",
         QJsonObject{
             {"width", r.sourceFrameSize.isEmpty() ? r.image.width() : r.sourceFrameSize.width()},
             {"height", r.sourceFrameSize.isEmpty() ? r.image.height() : r.sourceFrameSize.height()}}},
        {"frame", r.frameNumber}};
    history_.prepend(o);
    while (history_.size() > 200)
        history_.removeLast();
    QString error;
    if (!atomicWrite(dataRoot_ + "/history.json", QJsonDocument(history_).toJson(), &error))
        showNotice("运行记录保存失败：" + error, true);
    refreshHistory();
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
        auto *i = new QListWidgetItem(ui::icon("model", QColor("#60CDFF"), 34),
                                      f.fileName() + "\n" + detail + "\n" + p);
        i->setData(Qt::UserRole, p);
        i->setToolTip(p);
        i->setSizeHint(QSize(100, 94));
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
            ui::icon(QStringList{"work", "model", "graph", "history", "help", "video", "more"}[i],
                     QColor(i == index ? "#60CDFF" : "#D2D2D2")));
    }
    const QStringList titles = {"检测工作台", "模型库",   "模型显示", "运行记录",
                                "使用指南",   "录制视频", "更多"};
    const QStringList descriptions = {
        "从输入到洞察，让每一次视觉推理清晰可见。",   "管理本地模型，让每一个实验都有清晰的起点。",
        "查看网络结构、输入输出与层参数。",           "回看每一次推理，沉淀可追溯的运行数据。",
        "从模型配置到结果导出，掌握完整的工作流程。", "查看、播放和导出已保存的检测录像。",
        "示例体验与检测结果导出，集中在这里。"};
    pageTitle_->setText(titles[index]);
    pageSubtitle_->setText(descriptions[index]);
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
    }
    settings_->sync();
}
void MainWindow::closeEvent(QCloseEvent *e)
{
    if (gpuSetupProcess_)
    {
        closing_ = true;
        cancelGpuPreparation();
        e->ignore();
        return;
    }
    if (busy_)
    {
        closing_ = true;
        stopInference();
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
        stopInference();
        e->accept();
        return;
    }
    if (e->modifiers() & Qt::ControlModifier)
    {
        if (e->key() == Qt::Key_O && !busy_)
        {
            chooseImages();
            return;
        }
        if (e->key() == Qt::Key_M && !busy_)
        {
            importModel();
            return;
        }
        if (e->key() == Qt::Key_R)
        {
            startInference();
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
