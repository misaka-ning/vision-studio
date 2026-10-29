// SPDX-License-Identifier: AGPL-3.0-only
#include "workbenchpage.h"
#include "pagehelpers.h"
#include "ui/evaluationpanel.h"
#include "ui/imagecanvas.h"
#include "ui/lazyimagequeue.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMenu>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

using namespace vision::ui::pages;
namespace
{
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
QLabel *section(const QString &title, const QString &number)
{
    auto *l = text(number + "   " + title, "sectionTitle");
    l->setMinimumHeight(20);
    return l;
}
} // namespace

WorkbenchPage::WorkbenchPage(const QString &exportDirectory, QWidget *parent) : QWidget(parent)
{
    setObjectName("workbenchPage");
    auto *all = new QVBoxLayout(this);
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
    metric("检测目标", "eye", controls_.countMetric, "objects");
    metric("模型推理", "cpu", controls_.latencyMetric, "ms");
    metric("识别类别", "model", controls_.classMetric, "classes");
    metric("输入分辨率", "image", controls_.sizeMetric, "px");
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
    controls_.evaluation = new EvaluationPanel;
    auto *datasetButton = button("YOLO 数据集", "folder");
    datasetButton->setObjectName("openYoloDatasetButton");
    datasetButton->setToolTip("选择标准 YOLO 数据集文件夹或 YAML 配置 · Ctrl+O");
    controls_.evaluation->setModeControlVisible(false);
    controls_.datasetButton = datasetButton;
    commands->addWidget(datasetButton);
    controls_.lockedControls.append(datasetButton);
    connect(datasetButton, &QPushButton::clicked, controls_.evaluation, &EvaluationPanel::chooseDataset);
    auto *video = button("视频", "video");
    video->setObjectName("openVideoButton");
    auto *camera = button("摄像头", "camera");
    camera->setObjectName("openCameraButton");
    controls_.videoButton = video; controls_.cameraButton = camera;
    commands->addWidget(video); commands->addWidget(camera);
    controls_.lockedControls.append(video); controls_.lockedControls.append(camera);
    connect(video, &QPushButton::clicked, this, &WorkbenchPage::videoRequested);
    connect(camera, &QPushButton::clicked, this, &WorkbenchPage::cameraRequested);
    controls_.sourceLabel = text("选择数据集或连接实时输入", "tiny");
    controls_.sourceLabel->setObjectName("workbenchSourceSummary");
    controls_.sourceLabel->setWordWrap(false);
    controls_.sourceLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    commands->addWidget(controls_.sourceLabel, 1);
    commands->addStretch();
    commands->addSpacing(8);
    commands->addWidget(text("设备", "tiny"));
    controls_.computeDevice = new QComboBox;
    controls_.computeDevice->setObjectName("computeDevice");
    controls_.computeDevice->setAccessibleName("推理设备");
    controls_.computeDevice->addItem("自动", "auto");
    controls_.computeDevice->addItem("CPU", "cpu");
    controls_.computeDevice->addItem("NVIDIA GPU", "cuda");
    controls_.computeDevice->setFixedWidth(140);
    commands->addWidget(controls_.computeDevice);
    controls_.gpuDeviceLabel = text("GPU", "tiny");
    commands->addWidget(controls_.gpuDeviceLabel);
    controls_.gpuDeviceIndex = new QComboBox;
    controls_.gpuDeviceIndex->setObjectName("gpuDeviceIndex");
    controls_.gpuDeviceIndex->setAccessibleName("GPU 设备编号");
    controls_.gpuDeviceIndex->addItem("GPU 0 · 待检查", 0);
    controls_.gpuDeviceIndex->setFixedWidth(190);
    commands->addWidget(controls_.gpuDeviceIndex);
    controls_.deviceHint = text("●", "tiny");
    controls_.deviceHint->setObjectName("deviceEnvironmentHint");
    controls_.deviceHint->setFixedWidth(14);
    commands->addWidget(controls_.deviceHint);
    commandLayout->addLayout(commands);
    commandLayout->addWidget(controls_.evaluation->controlsWidget());
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
    controls_.lockedControls.append(controls_.computeDevice);
    controls_.lockedControls.append(controls_.gpuDeviceIndex);
    connect(controls_.computeDevice, &QComboBox::currentIndexChanged, this, &WorkbenchPage::deviceChanged);
    connect(controls_.gpuDeviceIndex, &QComboBox::currentIndexChanged, this, &WorkbenchPage::deviceChanged);
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
    controls_.cameraIndex = new QSpinBox;
    controls_.cameraIndex->setObjectName("cameraIndex");
    controls_.cameraIndex->setRange(0, 10);
    controls_.cameraIndex->setPrefix("摄像头编号  ");
    controls_.cameraIndex->hide();
    fields->addWidget(controls_.cameraIndex);
    controls_.lockedControls.append(controls_.cameraIndex);
    auto *sourceOptions = new QFormLayout;
    sourceOptions->setSpacing(7);
    sourceOptions->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    controls_.inputColorMode = new QComboBox;
    controls_.inputColorMode->setObjectName("inputColorMode");
    controls_.inputColorMode->addItem("RGB · 彩色", "rgb");
    controls_.inputColorMode->addItem("BGR · 彩色", "bgr");
    controls_.inputColorMode->addItem("灰度 · 明度", "grayscale");
    controls_.inputColorMode->setAccessibleName("输入颜色");
    controls_.inputColorMode->setToolTip("选择模型实际接收的颜色。灰度会转换为明度：单通道模型直接使用，三通道模型复制"
                                "为三个相同通道。预览同步显示所选颜色模式。");
    sourceOptions->addRow("输入颜色", controls_.inputColorMode);
    controls_.stereoView = new QComboBox;
    controls_.stereoView->setObjectName("stereoView");
    controls_.stereoView->addItem("完整画面", "full");
    controls_.stereoView->addItem("双目左目", "left");
    controls_.stereoView->addItem("双目右目", "right");
    controls_.stereoView->setAccessibleName("双目画面选择");
    controls_.stereoView->setToolTip("单设备左右并排（SBS）输入：沿水平中线裁出左目或右目，预览、推理和导出均使用所选"
                            "眼；完整画面不裁剪。");
    controls_.stereoLabel = text("画面选择");
    sourceOptions->addRow(controls_.stereoLabel, controls_.stereoView);
    controls_.stereoLabel->hide();
    controls_.stereoView->hide();
    controls_.stereoView->setEnabled(false);
    fields->addLayout(sourceOptions);
    controls_.preprocessHint = text("彩色输入 · RGB / BGR 通道顺序", "tiny");
    controls_.preprocessHint->setWordWrap(true);
    fields->addWidget(controls_.preprocessHint);
    controls_.lockedControls.append(controls_.inputColorMode);
    controls_.lockedControls.append(controls_.stereoView);
    fields->addSpacing(7);
    fields->addWidget(section("检测模型", "02"));
    controls_.modelName = text("选择视觉模型", "modelName");
    controls_.modelName->setObjectName("activeModelName");
    controls_.modelName->setWordWrap(true);
    fields->addWidget(controls_.modelName);
    controls_.modelMeta = text("支持 ONNX / PT 检测模型", "tiny");
    controls_.modelMeta->setObjectName("modelMetadata");
    controls_.modelMeta->setWordWrap(true);
    fields->addWidget(controls_.modelMeta);
    controls_.modelButton = button("导入模型", "plus");
    fields->addWidget(controls_.modelButton);
    connect(controls_.modelButton, &QPushButton::clicked, this, &WorkbenchPage::importModelRequested);
    controls_.lockedControls.append(controls_.modelButton);
    controls_.taskBox = new QComboBox;
    controls_.taskBox->setObjectName("modelTask");
    controls_.taskBox->addItems({"YOLOv5 · 目标检测", "YOLOv8 / 11 · 目标检测", "图像分类"});
    controls_.taskBox->setToolTip("按模型实际输出选择格式，分割、姿态与含 NMS 的模型暂不支持");
    fields->addWidget(controls_.taskBox);
    controls_.lockedControls.append(controls_.taskBox);
    controls_.labelButton = button("类别标签 · 默认 COCO 80", "help");
    controls_.labelButton->setToolTip("导入 UTF-8 文本，每行一个类别名。自定义模型应配置自己的标签。");
    fields->addWidget(controls_.labelButton);
    auto *labelMenu = new QMenu(controls_.labelButton);
    labelMenu->addAction("导入标签文件", this, &WorkbenchPage::importLabelsRequested);
    labelMenu->addAction("使用模型默认类别", this, &WorkbenchPage::defaultLabelsRequested);
    controls_.labelButton->setMenu(labelMenu);
    controls_.lockedControls.append(controls_.labelButton);
    auto *filterForm = new QFormLayout;
    filterForm->setContentsMargins(0, 0, 0, 0);
    controls_.classFilter = new QLineEdit;
    controls_.classFilter->setObjectName("classFilter");
    controls_.classFilter->setAccessibleName("类别筛选 ID");
    controls_.classFilter->setPlaceholderText("全部类别，或 0,2,32");
    controls_.classFilter->setMaxLength(4096);
    controls_.classFilter->setToolTip("按模型原始 class_id 筛选，ID 从 0 开始；逗号、中文逗号或空格分隔。\n"
                             "类别标签只映射名称，不改变 ID。COCO 的 ID 32 是 sports ball，包含多种运动球，筛选它不等于只识别足球。");
    filterForm->addRow("类别筛选", controls_.classFilter);
    fields->addLayout(filterForm);
    controls_.classFilterHint = text("留空检测全部类别", "tiny");
    controls_.classFilterHint->setObjectName("classFilterHint");
    controls_.classFilterHint->setWordWrap(true);
    controls_.classFilterHint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fields->addWidget(controls_.classFilterHint);
    controls_.lockedControls.append(controls_.classFilter);
    connect(controls_.classFilter, &QLineEdit::textChanged, this, &WorkbenchPage::classFilterChanged);
    fields->addSpacing(7);
    fields->addWidget(section("推理参数", "03"));
    auto *form = new QFormLayout;
    form->setSpacing(9);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    controls_.inputSize = new WheelSafeSpinBox<QSpinBox>;
    controls_.inputSize->setObjectName("inputSize");
    controls_.inputSize->setRange(32, 4096);
    controls_.inputSize->setSingleStep(32);
    controls_.inputSize->setValue(640);
    controls_.inputSize->setSuffix(" px");
    form->addRow("输入尺寸", controls_.inputSize);
    controls_.confidence = new WheelSafeSpinBox<QDoubleSpinBox>;
    controls_.confidence->setObjectName("confidence");
    controls_.confidence->setRange(0.001, 1.0);
    controls_.confidence->setDecimals(3);
    controls_.confidence->setSingleStep(0.05);
    controls_.confidence->setValue(0.25);
    form->addRow("置信度", controls_.confidence);
    controls_.iou = new WheelSafeSpinBox<QDoubleSpinBox>;
    controls_.iou->setObjectName("nmsIou");
    controls_.iou->setRange(0.01, 1.0);
    controls_.iou->setDecimals(2);
    controls_.iou->setSingleStep(0.05);
    controls_.iou->setValue(0.45);
    form->addRow("NMS IoU", controls_.iou);
    fields->addLayout(form);
    for (auto *p : QList<QWidget *>{controls_.inputSize, controls_.confidence, controls_.iou})
        controls_.lockedControls.append(p);
    auto *advanced = button("预处理设置 ▾", {}, "ghost");
    fields->addWidget(advanced);
    auto *adv = new QWidget;
    auto *af = new QFormLayout(adv);
    af->setContentsMargins(0, 5, 0, 0);
    af->setSpacing(6);
    controls_.scale = new QDoubleSpinBox;
    controls_.scale->setDecimals(8);
    controls_.scale->setRange(0.00000001, 10);
    controls_.scale->setValue(1.0 / 255.0);
    controls_.scale->setSingleStep(0.001);
    af->addRow("缩放系数", controls_.scale);
    auto mean = [&](QDoubleSpinBox *&m, const QString &label)
    {
        m = new QDoubleSpinBox;
        m->setRange(-1024, 1024);
        m->setDecimals(3);
        af->addRow(label, m);
        controls_.lockedControls.append(m);
    };
    mean(controls_.meanR, "均值 R");
    mean(controls_.meanG, "均值 G");
    mean(controls_.meanB, "均值 B");
    controls_.meanR->setObjectName("meanR");
    controls_.meanG->setObjectName("meanG");
    controls_.meanB->setObjectName("meanB");
    controls_.meanRLabel = qobject_cast<QLabel *>(af->labelForField(controls_.meanR));
    adv->hide();
    fields->addWidget(adv);
    connect(advanced, &QPushButton::clicked, this,
            [adv, advanced]
            {
                adv->setVisible(!adv->isVisible());
                advanced->setText(adv->isVisible() ? "预处理设置 ▴" : "预处理设置 ▾");
            });
    controls_.lockedControls.append(controls_.scale);
    controls_.autoExport = new QCheckBox("自动保存每张图片的结果");
    controls_.autoExport->setToolTip("保存标注 PNG、JSON 和 CSV；视频与摄像头在停止后保存最后一帧");
    fields->addWidget(controls_.autoExport);
    auto *openExportDirectory = button("打开保存文件夹", "folder", "ghost");
    openExportDirectory->setObjectName("openExportDirectoryButton");
    openExportDirectory->setToolTip(exportDirectory);
    controls_.openExportDirectory = openExportDirectory;
    fields->addWidget(openExportDirectory);
    connect(openExportDirectory, &QPushButton::clicked, this, &WorkbenchPage::openExportDirectoryRequested);
    controls_.lockedControls.append(controls_.autoExport);
    controls_.backendBadge = text("设备待验证", "chip");
    controls_.backendBadge->setObjectName("actualDeviceBadge");
    controls_.backendBadge->setWordWrap(true);
    controls_.backendBadge->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fields->addWidget(controls_.backendBadge);
    fields->addStretch();
    scroll->setWidget(inner);
    cl->addWidget(scroll, 1);
    controls_.detectionRangePanel = new QWidget;
    controls_.detectionRangePanel->setObjectName("detectionRangePanel");
    auto *rangeLayout = new QVBoxLayout(controls_.detectionRangePanel);
    rangeLayout->setContentsMargins(0, 0, 0, 0);
    rangeLayout->setSpacing(5);
    auto *rangeForm = new QFormLayout;
    rangeForm->setContentsMargins(0, 0, 0, 0);
    controls_.detectionRange = new QComboBox;
    controls_.detectionRange->setObjectName("detectionRange");
    controls_.detectionRange->setAccessibleName("检测范围");
    controls_.detectionRange->addItems({"检测当前集合", "从当前图片向后检测", "只检测当前图片"});
    controls_.detectionRange->setToolTip("点击开始时固定本次范围和队列顺序；检测当前集合始终从第一张开始。");
    rangeForm->addRow("检测范围", controls_.detectionRange);
    rangeLayout->addLayout(rangeForm);
    controls_.rangeHint = text("请先导入 YOLO 数据集", "tiny");
    controls_.rangeHint->setObjectName("detectionRangeHint");
    controls_.rangeHint->setWordWrap(true);
    rangeLayout->addWidget(controls_.rangeHint);
    cl->addWidget(controls_.detectionRangePanel);
    controls_.lockedControls.append(controls_.detectionRange);
    connect(controls_.detectionRange, &QComboBox::currentIndexChanged, this, &WorkbenchPage::rangeChanged);
    controls_.runButton = button("开始检测", "play", "primary");
    controls_.runButton->setObjectName("startDetectionButton");
    controls_.runButton->setMinimumHeight(44);
    cl->addWidget(controls_.runButton);
    connect(controls_.runButton, &QPushButton::clicked, this, &WorkbenchPage::startRequested);
    controls_.stopButton = button("停止运行", "stop", "danger");
    controls_.stopButton->hide();
    cl->addWidget(controls_.stopButton);
    connect(controls_.stopButton, &QPushButton::clicked, this, &WorkbenchPage::stopRequested);
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
    controls_.canvasTitle = text("视觉预览", "sectionTitle");
    tl->addWidget(controls_.canvasTitle, 1);
    controls_.showBoxes = new QCheckBox("检测框");
    controls_.showBoxes->setChecked(true);
    tl->addWidget(controls_.showBoxes);
    controls_.showLabels = new QCheckBox("标签");
    controls_.showLabels->setChecked(true);
    tl->addWidget(controls_.showLabels);
    controls_.showGroundTruth = new QCheckBox("GT 标注");
    controls_.showGroundTruth->setObjectName("showGroundTruth");
    controls_.showGroundTruth->setChecked(true);
    controls_.showGroundTruth->hide();
    tl->addWidget(controls_.showGroundTruth);
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
    controls_.recordingStatus = text("视频 / 摄像头检测时可录制", "tiny");
    controls_.recordingStatus->setObjectName("recordingStatus");
    recordingRow->addWidget(controls_.recordingStatus, 1);
    controls_.recordButton = button("开始录制", "record");
    controls_.recordButton->setObjectName("recordButton");
    controls_.recordButton->setEnabled(false);
    controls_.recordButton->setToolTip("先开始视频或摄像头检测，再点击录制。再次点击结束并保存到录制视频页。");
    recordingRow->addWidget(controls_.recordButton);
    cf->addLayout(recordingRow);
    connect(controls_.recordButton, &QPushButton::clicked, this, &WorkbenchPage::recordRequested);
    controls_.recordingClock = new QTimer(this);
    controls_.recordingClock->setInterval(250);
    connect(controls_.recordingClock, &QTimer::timeout, this, &WorkbenchPage::recordingTick);
    controls_.canvas = new ImageCanvas;
    controls_.canvas->setObjectName("imageCanvas");
    controls_.canvas->setMinimumHeight(280);
    cf->addWidget(controls_.canvas, 1);
    auto *cb = new QHBoxLayout;
    cb->setContentsMargins(14, 3, 14, 3);
    cb->addWidget(text("滚轮缩放 · 拖动平移 · 点击目标查看", "tiny"), 1);
    controls_.zoomLabel = text("100%", "tiny");
    cb->addWidget(controls_.zoomLabel);
    cf->addLayout(cb);
    ml->addWidget(canvasFrame, 1);
    connect(controls_.showGroundTruth, &QCheckBox::toggled, controls_.canvas, &ImageCanvas::setGroundTruthVisible);
    connect(fit, &QPushButton::clicked, controls_.canvas, &ImageCanvas::fitToView);
    connect(actual, &QPushButton::clicked, controls_.canvas, &ImageCanvas::actualSize);
    connect(controls_.showBoxes, &QCheckBox::toggled, controls_.canvas, &ImageCanvas::setBoxesVisible);
    connect(controls_.showLabels, &QCheckBox::toggled, controls_.canvas, &ImageCanvas::setLabelsVisible);
    connect(controls_.canvas, &ImageCanvas::zoomChanged, this,
            [this](int z) { controls_.zoomLabel->setText(QString::number(z) + "%"); });
    connect(controls_.canvas, &ImageCanvas::filesDropped, this, &WorkbenchPage::filesDropped);
    auto *queueFrame = card();
    auto *ql = new QHBoxLayout(queueFrame);
    ql->setContentsMargins(12, 6, 12, 6);
    ql->setSpacing(5);
    auto *qh = new QVBoxLayout;
    qh->addWidget(text("输入队列", "sectionTitle"), 1);
    auto *clear = button("清空", "cross", "ghost");
    clear->setMinimumHeight(25);
    clear->setMaximumHeight(27);
    controls_.clearQueue = clear;
    qh->addWidget(clear);
    controls_.lockedControls.append(clear);
    connect(clear, &QPushButton::clicked, this, &WorkbenchPage::clearQueueRequested);
    ql->addLayout(qh);
    controls_.queue = new LazyImageQueue;
    controls_.queue->setObjectName("inputQueue");
    controls_.queue->setFlow(QListView::LeftToRight);
    controls_.queue->setViewMode(QListView::IconMode);
    controls_.queue->setIconSize(QSize(64, 36));
    controls_.queue->setGridSize(QSize(100, 64));
    controls_.queue->setWrapping(false);
    controls_.queue->setFixedHeight(68);
    controls_.queue->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ql->addWidget(controls_.queue, 1);
    ml->addWidget(queueFrame);
    connect(controls_.queue, &QListWidget::currentRowChanged, this, &WorkbenchPage::queueRowChanged);
    connect(controls_.queue, &QListWidget::itemSelectionChanged, this, [this] { emit queueSelectionChanged(controls_.queue->currentRow(), controls_.queue->selectionModel()->hasSelection()); });
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
    controls_.resultInfo = text("运行模型后，目标和置信度将显示在这里。", "muted");
    controls_.resultInfo->setWordWrap(true);
    controls_.resultInfo->setMinimumHeight(44);
    controls_.resultInfo->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    rl->addWidget(controls_.resultInfo);
    controls_.predictionTable = new QTableWidget(0, 3);
    controls_.predictionTable->setHorizontalHeaderLabels({"类别", "置信度", "类别 ID"});
    tableStyle(controls_.predictionTable);
    controls_.predictionTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    controls_.predictionTable->setColumnWidth(1, 78);
    controls_.predictionTable->setColumnWidth(2, 58);
    rl->addWidget(controls_.predictionTable, 1);
    controls_.emptyResults = text("等待检测\n\n目标详情将在运行后呈现。", "muted");
    controls_.emptyResults->setAlignment(Qt::AlignCenter);
    controls_.emptyResults->setWordWrap(true);
    rl->addWidget(controls_.emptyResults);
    rl->addStretch(0);
    auto *hints = text("点击列表中的目标，可在画布中定位。\n检测框坐标以原始图像像素为准。", "tiny");
    hints->setWordWrap(true);
    rl->addWidget(hints);
    columns->addWidget(inspector);
    connect(controls_.predictionTable, &QTableWidget::itemSelectionChanged, this,
            [this]
            {
                controls_.canvas->setSelectedPrediction(
                    controls_.predictionTable->selectedItems().isEmpty() ? -1 : controls_.predictionTable->currentRow());
            });
    connect(controls_.canvas, &ImageCanvas::predictionSelected, this,
            [this](int row)
            {
                if (row >= 0 && row < controls_.predictionTable->rowCount())
                    controls_.predictionTable->selectRow(row);
                else
                    controls_.predictionTable->clearSelection();
            });
    connect(controls_.taskBox, &QComboBox::currentIndexChanged, this, &WorkbenchPage::taskChanged);
    connect(controls_.inputSize, &QSpinBox::valueChanged, this, &WorkbenchPage::parametersChanged);
    for (auto *control : {controls_.confidence, controls_.iou, controls_.scale, controls_.meanR, controls_.meanG, controls_.meanB})
        connect(control, &QDoubleSpinBox::valueChanged, this, &WorkbenchPage::parametersChanged);
    connect(controls_.inputColorMode, &QComboBox::currentIndexChanged, this, &WorkbenchPage::inputColorChanged);
    connect(controls_.stereoView, &QComboBox::currentIndexChanged, this, &WorkbenchPage::stereoChanged);
    connect(controls_.cameraIndex, &QSpinBox::valueChanged, this, &WorkbenchPage::cameraIndexChanged);
    columns->setStretchFactor(0, 0);
    columns->setStretchFactor(1, 1);
    columns->setStretchFactor(2, 0);
    columns->setSizes({280, 660, 270});
    auto *previewPage = new QWidget;
    previewPage->setObjectName("imagePreviewPage");
    auto *previewLayout = new QVBoxLayout(previewPage);
    previewLayout->setContentsMargins(0, 0, 0, 0);
    previewLayout->setSpacing(6);
    controls_.evaluationReviewBar = card("evaluationReviewBar");
    auto *reviewLayout = new QHBoxLayout(controls_.evaluationReviewBar);
    reviewLayout->setContentsMargins(10, 5, 10, 5);
    reviewLayout->setSpacing(6);
    controls_.evaluationReviewSummaryLabel = text({}, "muted");
    controls_.evaluationReviewSummaryLabel->setObjectName("evaluationReviewSummaryLabel");
    controls_.evaluationReviewSummaryLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    reviewLayout->addWidget(controls_.evaluationReviewSummaryLabel, 1);
    controls_.evaluationPreviewFilter = new QComboBox;
    controls_.evaluationPreviewFilter->setObjectName("evaluationPreviewFilterCombo");
    for (const auto &entry : QList<QPair<QString, QString>>{{"全部图片", "all"}, {"包含误检", "fp"},
             {"包含漏检", "fn"}, {"无目标误报", "negative_fp"}, {"全部错例", "errors"}})
        controls_.evaluationPreviewFilter->addItem(entry.first, entry.second);
    controls_.evaluationPreviewFilter->setMaximumWidth(120);
    reviewLayout->addWidget(controls_.evaluationPreviewFilter);
    connect(controls_.evaluationPreviewFilter, &QComboBox::currentIndexChanged, this, [this] { emit reviewFilterRequested(controls_.evaluationPreviewFilter->currentData().toString()); });
    controls_.evaluationReviewPreviousButton = button("上一张");
    controls_.evaluationReviewPreviousButton->setObjectName("evaluationReviewPreviousButton");
    controls_.evaluationReviewNextButton = button("下一张");
    controls_.evaluationReviewNextButton->setObjectName("evaluationReviewNextButton");
    reviewLayout->addWidget(controls_.evaluationReviewPreviousButton);
    reviewLayout->addWidget(controls_.evaluationReviewNextButton);
    connect(controls_.evaluationReviewPreviousButton, &QPushButton::clicked, this, [this] { emit reviewNavigateRequested(-1); });
    connect(controls_.evaluationReviewNextButton, &QPushButton::clicked, this, [this] { emit reviewNavigateRequested(1); });
    auto *allReview = button("全部图片", "image");
    allReview->setObjectName("evaluationReviewAllButton");
    reviewLayout->addWidget(allReview);
    connect(allReview, &QPushButton::clicked, this, [this] { emit reviewFilterRequested("all"); });
    auto *analysisReview = button("返回评估", "graph");
    analysisReview->setObjectName("evaluationReviewAnalysisButton");
    reviewLayout->addWidget(analysisReview);
    connect(analysisReview, &QPushButton::clicked, this, [this] { controls_.workbenchTabs->setCurrentIndex(1); });
    for (auto *control : {controls_.evaluationReviewPreviousButton, controls_.evaluationReviewNextButton, allReview, analysisReview}) {
        control->setMinimumHeight(30);
        control->setIconSize(QSize(14, 14));
    }
    controls_.evaluationReviewSummaryLabel->setStyleSheet("font-size:11px; color:#D2D2D2;");
    controls_.evaluationPreviewFilter->setStyleSheet("QComboBox { min-height:20px; padding:4px 6px; }");
    controls_.evaluationReviewBar->hide();
    previewLayout->addWidget(controls_.evaluationReviewBar);
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
    controls_.evaluationTaskSummary = text("导入带标注的 YOLO 数据集，开始模型评估", "muted");
    controls_.evaluationTaskSummary->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    taskActions->addWidget(controls_.evaluationTaskSummary, 1);
    controls_.evaluationRange = new QComboBox;
    controls_.evaluationRange->setObjectName("evaluationRangeCombo");
    for (int i = 0; i < controls_.detectionRange->count(); ++i) controls_.evaluationRange->addItem(controls_.detectionRange->itemText(i));
    taskActions->addWidget(controls_.evaluationRange);
    controls_.lockedControls.append(controls_.evaluationRange);
    connect(controls_.evaluationRange, &QComboBox::currentIndexChanged, controls_.detectionRange, &QComboBox::setCurrentIndex);
    connect(controls_.detectionRange, &QComboBox::currentIndexChanged, controls_.evaluationRange, &QComboBox::setCurrentIndex);
    auto *configure = button("推理配置", "settings");
    configure->setObjectName("evaluationConfigureButton");
    taskActions->addWidget(configure);
    connect(configure, &QPushButton::clicked, this, [this] { controls_.workbenchTabs->setCurrentIndex(0); });
    controls_.evaluationReviewButton = button("查看错例", "eye");
    controls_.evaluationReviewButton->setObjectName("evaluationReviewButton");
    controls_.evaluationReviewButton->setEnabled(false);
    taskActions->addWidget(controls_.evaluationReviewButton);
    connect(controls_.evaluationReviewButton, &QPushButton::clicked, this, &WorkbenchPage::reviewErrorsRequested);
    controls_.evaluationStartButton = button("开始评估", "play", "primary");
    controls_.evaluationStartButton->setObjectName("evaluationStartButton");
    taskActions->addWidget(controls_.evaluationStartButton);
    connect(controls_.evaluationStartButton, &QPushButton::clicked, this, &WorkbenchPage::startRequested);
    controls_.evaluationStopButton = button("停止评估", "stop", "danger");
    controls_.evaluationStopButton->setObjectName("evaluationStopButton");
    controls_.evaluationStopButton->hide();
    taskActions->addWidget(controls_.evaluationStopButton);
    connect(controls_.evaluationStopButton, &QPushButton::clicked, this, &WorkbenchPage::stopRequested);
    taskLayout->addLayout(taskActions);
    controls_.evaluationRangeHint = text("请先导入 YOLO 数据集；评估需要有效标注与类别映射。", "tiny");
    controls_.evaluationRangeHint->setObjectName("evaluationRangeHint");
    controls_.evaluationRangeHint->setWordWrap(true);
    taskLayout->addWidget(controls_.evaluationRangeHint);
    evaluationLayout->addWidget(taskCard);
    evaluationLayout->addWidget(controls_.evaluation, 1);
    controls_.workbenchTabs = new QTabWidget;
    controls_.workbenchTabs->setObjectName("workbenchAnalysisTabs");
    controls_.workbenchTabs->addTab(previewPage, "图像预览");
    controls_.workbenchTabs->addTab(evaluationPage, "评估与阈值");
    all->addWidget(controls_.workbenchTabs, 1);
    connect(controls_.workbenchTabs, &QTabWidget::currentChanged, this, &WorkbenchPage::workbenchTabChanged);
}

void WorkbenchPage::setExportDirectory(const QString &directory)
{
    controls_.openExportDirectory->setToolTip(directory);
}
