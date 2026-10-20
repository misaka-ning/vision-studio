// SPDX-License-Identifier: AGPL-3.0-only
#include "evaluationpanel.h"

#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHeaderView>
#include <QGridLayout>
#include <QJsonDocument>
#include <QSaveFile>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QRunnable>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QThreadPool>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>

namespace
{
template <class Base> class NoWheel final : public Base
{
protected:
    void wheelEvent(QWheelEvent *event) override { event->ignore(); }
};
class ElidedLabel final : public QLabel
{
public:
    using QLabel::QLabel;
protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setPen(palette().color(foregroundRole()));
        painter.drawText(contentsRect(), alignment() | Qt::TextSingleLine,
                         fontMetrics().elidedText(text(), Qt::ElideRight, contentsRect().width()));
    }
};
class ThresholdSpin final : public QDoubleSpinBox
{
public:
    void setRows(const QJsonArray &rows)
    {
        values_.clear();
        for (const auto &entry : rows) {
            const auto confidence = entry.toObject().value("confidence");
            if (confidence.isDouble() && std::isfinite(confidence.toDouble())) values_.append(confidence.toDouble());
        }
        std::sort(values_.begin(), values_.end());
        values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
        update();
    }
protected:
    void wheelEvent(QWheelEvent *event) override { event->ignore(); }
    int nearest() const
    {
        int result = 0;
        for (int i = 1; i < values_.size(); ++i)
            if (std::abs(values_[i] - value()) < std::abs(values_[result] - value())) result = i;
        return result;
    }
    void stepBy(int steps) override
    {
        if (!values_.isEmpty()) setValue(values_[std::clamp(nearest() + steps, 0, int(values_.size()) - 1)]);
    }
    StepEnabled stepEnabled() const override
    {
        StepEnabled enabled = StepNone;
        if (values_.isEmpty()) return enabled;
        const int index = nearest();
        if (index > 0) enabled |= StepDownEnabled;
        if (index + 1 < values_.size()) enabled |= StepUpEnabled;
        return enabled;
    }
private:
    QVector<double> values_;
};
QLabel *label(const QString &text, const char *name = "muted")
{
    auto *result = new QLabel(text);
    result->setObjectName(name);
    result->setWordWrap(true);
    result->setTextFormat(Qt::PlainText);
    return result;
}
QFrame *card()
{
    auto *result = new QFrame;
    result->setObjectName("card");
    return result;
}
QPushButton *button(const QString &text, const char *name)
{
    auto *result = new QPushButton(text);
    result->setObjectName(name);
    result->setMinimumHeight(32);
    result->setCursor(Qt::PointingHandCursor);
    return result;
}
QDoubleSpinBox *number(const char *name, double minimum, double maximum, double step,
                       double value, int decimals = 2)
{
    auto *result = new NoWheel<QDoubleSpinBox>;
    result->setObjectName(name);
    result->setDecimals(decimals);
    result->setRange(minimum, maximum);
    result->setSingleStep(step);
    result->setValue(value);
    return result;
}
bool numeric(const QJsonValue &value)
{
    return value.isDouble() && std::isfinite(value.toDouble());
}
QString percentage(const QJsonValue &value)
{
    return numeric(value) ? QString::number(value.toDouble() * 100, 'f', 1) + "%" : QStringLiteral("N/A");
}
QString metricText(const QJsonObject &object, const QString &key)
{
    return percentage(object.value(key));
}
QString className(const QJsonArray &names, int index)
{
    return index >= 0 && index < names.size() ? names.at(index).toString() : QString();
}
QJsonObject requestOf(const QJsonObject &report)
{
    const auto request = report.value("request").toObject();
    return request.isEmpty() ? report.value("config").toObject() : request;
}
double nearestConfidence(const QJsonArray &rows, double value)
{
    double best = value, distance = std::numeric_limits<double>::max();
    for (const auto &entry : rows) {
        const auto candidate = entry.toObject().value("confidence");
        if (!numeric(candidate)) continue;
        const double delta = std::abs(candidate.toDouble() - value);
        if (delta < distance) { distance = delta; best = candidate.toDouble(); }
    }
    return best;
}
QString markdownPath(const QString &directory)
{
    for (const QString &name : {QStringLiteral("REPORT.md"), QStringLiteral("report.md")}) {
        const QString path = directory + "/" + name;
        if (QFileInfo::exists(path)) return path;
    }
    return {};
}
QString copyReport(const QString &source, const QString &destination)
{
    // Reports are immutable products, not user-selected source trees. Refuse
    // links and special files, and never overwrite a prior export.
    const QString canonical = QFileInfo(source).canonicalFilePath();
    const QString destinationParent = QFileInfo(QFileInfo(destination).absolutePath()).canonicalFilePath();
    if (QFileInfo(source).isSymLink() || canonical.isEmpty() || destinationParent.isEmpty() ||
        destinationParent == canonical || destinationParent.startsWith(canonical + "/") ||
        QFileInfo::exists(destination))
        return QStringLiteral("导出目录无效，不能位于原报告内部或覆盖已有目录。");
    QStringList files, directories;
    QDirIterator iterator(canonical, QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden,
                          QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        iterator.next();
        const auto info = iterator.fileInfo();
        if (info.isSymLink() || (!info.isFile() && !info.isDir()))
            return QStringLiteral("报告中包含链接或非普通文件，已拒绝复制。");
        const QString relative = QDir(canonical).relativeFilePath(info.absoluteFilePath());
        (info.isDir() ? directories : files).append(relative);
    }
    if (!QDir().mkdir(destination)) return QStringLiteral("不能创建导出目录。");
    for (const QString &relative : directories)
        if (!QDir().mkpath(destination + "/" + relative))
            return QStringLiteral("创建子目录失败，已保留部分导出以便检查。");
    for (const QString &relative : files) {
        const QString origin = canonical + "/" + relative;
        if (QFileInfo(origin).isSymLink() || !QFile::copy(origin, destination + "/" + relative))
            return QStringLiteral("复制失败，已保留部分导出；原报告未修改。");
    }
    return {};
}
}

// A native, dependency-free curve. Every plotted point is a measured operating
// point from the report; missing values deliberately break the line.
class EvaluationCurve final : public QWidget
{
public:
    explicit EvaluationCurve(QWidget *parent = nullptr) : QWidget(parent)
    {
        setObjectName("evaluationThresholdCurve");
        setMouseTracking(true);
        setMinimumSize(380, 270);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    void setRows(const QJsonArray &rows) { rows_ = rows; hovered_ = -1; update(); }
    void setThreshold(double threshold) { selected_ = threshold; update(); }
    std::function<void(double)> selected;
protected:
    QRectF plot() const { const int top = width() < 560 ? 54 : 34; return QRectF(48, top, qMax(1, width() - 72), qMax(1, height() - top - 44)); }
    int nearest(double value) const
    {
        int best = -1;
        double distance = std::numeric_limits<double>::max();
        for (int i = 0; i < rows_.size(); ++i) {
            const auto confidence = rows_.at(i).toObject().value("confidence");
            if (!numeric(confidence)) continue;
            const double delta = std::abs(confidence.toDouble() - value);
            if (delta < distance) { distance = delta; best = i; }
        }
        return best;
    }
    void mouseMoveEvent(QMouseEvent *event) override
    {
        const QRectF area = plot();
        hovered_ = area.contains(event->position()) ? nearest((event->position().x() - area.left()) / area.width()) : -1;
        update();
    }
    void leaveEvent(QEvent *) override { hovered_ = -1; update(); }
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() != Qt::LeftButton || !plot().contains(event->position())) return;
        const int index = nearest((event->position().x() - plot().left()) / plot().width());
        if (index >= 0 && selected) selected(rows_.at(index).toObject().value("confidence").toDouble());
    }
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        const QRectF area = plot();
        painter.fillRect(rect(), QColor("#272727"));
        painter.setFont(QFont(font().family(), 9));
        for (int tick = 0; tick <= 5; ++tick) {
            const double ratio = tick / 5.0;
            painter.setPen(QPen(QColor("#414141"), 1, Qt::DotLine));
            const double x = area.left() + ratio * area.width();
            const double y = area.bottom() - ratio * area.height();
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()));
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y));
            painter.setPen(QColor("#BCBEC3"));
            painter.drawText(QRectF(x - 20, area.bottom() + 8, 40, 18), Qt::AlignCenter, QString::number(ratio, 'f', 1));
            painter.drawText(QRectF(0, y - 10, 38, 20), Qt::AlignRight | Qt::AlignVCenter, QString::number(ratio, 'f', 1));
        }
        painter.setPen(QColor("#D2D2D2"));
        painter.drawText(QRectF(area.left(), height() - 20, area.width(), 18), Qt::AlignCenter, "置信度阈值");
        const QStringList keys = {"precision", "recall", "F1", "no_ball_false_positive_rate"};
        const QStringList titles = {"Precision", "Recall", "F1", "无目标误报率"};
        const QList<QColor> colors = {QColor("#60CDFF"), QColor("#A6E3B0"), QColor("#D4B6FF"), QColor("#FFB88A")};
        int legendX = int(area.left()), legendY = 16;
        for (int series = 0; series < keys.size(); ++series) {
            painter.setPen(QPen(colors[series], 2.2));
            const int needed = painter.fontMetrics().horizontalAdvance(titles[series]) + 38;
            if (legendX + needed > width() - 10) { legendX = int(area.left()); legendY += 18; }
            painter.drawLine(legendX, legendY, legendX + 16, legendY);
            painter.setPen(QColor("#E4E4E4"));
            painter.drawText(legendX + 22, legendY + 4, titles[series]);
            legendX += needed;
            QPainterPath path;
            bool started = false;
            for (const auto &entry : rows_) {
                const auto row = entry.toObject();
                const auto x = row.value("confidence"), y = row.value(keys[series]);
                if (!numeric(x) || !numeric(y)) { started = false; continue; }
                const QPointF point(area.left() + std::clamp(x.toDouble(), 0.0, 1.0) * area.width(),
                                    area.bottom() - std::clamp(y.toDouble(), 0.0, 1.0) * area.height());
                if (started) path.lineTo(point); else { path.moveTo(point); started = true; }
            }
            painter.setPen(QPen(colors[series], 2.2));
            painter.drawPath(path);
        }
        if (rows_.isEmpty()) {
            painter.setPen(QColor("#BCBEC3"));
            painter.drawText(area.adjusted(15, 10, -15, -10), Qt::AlignCenter | Qt::TextWordWrap,
                             "完成评估后显示阈值曲线\n移动查看，点击选择分析阈值");
            return;
        }
        const int index = hovered_ >= 0 ? hovered_ : nearest(selected_);
        if (index < 0) return;
        const auto row = rows_.at(index).toObject();
        const double x = area.left() + row.value("confidence").toDouble() * area.width();
        painter.setPen(QPen(QColor("#E6E6E6"), 1, Qt::DashLine));
        painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()));
        if (hovered_ >= 0) {
            const qreal boxWidth = qMin(205.0, area.width());
            const qreal left = std::clamp(x + 12, area.left(), area.right() - boxWidth);
            const QRectF box(left, area.top() + 10, boxWidth, 114);
            painter.setPen(QColor("#666666")); painter.setBrush(QColor("#303030"));
            painter.drawRoundedRect(box, 5, 5);
            painter.setPen(QColor("#F5F5F5"));
            const QString text = QString("阈值 %1\nP %2    R %3\nF1 %4\n无目标误报率 %5")
                .arg(row.value("confidence").toDouble(), 0, 'f', 3)
                .arg(metricText(row, "precision"), metricText(row, "recall"), metricText(row, "F1"), metricText(row, "no_ball_false_positive_rate"));
            painter.drawText(box.adjusted(10, 8, -8, -8), Qt::AlignLeft | Qt::AlignVCenter, text);
        }
    }
private:
    QJsonArray rows_;
    int hovered_ = -1;
    double selected_ = .25;
};

EvaluationPanel::EvaluationPanel(QWidget *parent) : QWidget(parent)
{
    setObjectName("evaluationPanel");
    controls_ = new QWidget(this);
    controls_->setObjectName("evaluationControls");
    controls_->setStyleSheet("#evaluationControls { background:#272727; }");
    controls_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto *configuration = new QVBoxLayout(controls_);
    configuration->setContentsMargins(0, 2, 0, 2);
    configuration->setSpacing(2);
    auto *modeRow = new QHBoxLayout;
    modeRow->setSpacing(8);
    modeRow->addWidget(label("工作模式", "sectionTitle"));
    mode_ = new NoWheel<QComboBox>;
    mode_->setObjectName("evaluationModeCombo");
    mode_->addItem("检测预览", false); mode_->addItem("评估调优", true);
    mode_->setFixedWidth(126);
    modeRow->addWidget(mode_);
    datasetHint_ = new ElidedLabel("导入 YOLO 数据集，可叠加标注并评估模型。");
    datasetHint_->setObjectName("evaluationDatasetHint");
    datasetHint_->setTextFormat(Qt::PlainText);
    datasetHint_->setStyleSheet("color:#B8B8B8; font-size:11px;");
    datasetHint_->setWordWrap(false);
    datasetHint_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    datasetHint_->setMinimumWidth(0);
    datasetHint_->setMaximumHeight(32);
    modeRow->addWidget(datasetHint_, 1);
    split_ = new NoWheel<QComboBox>;
    split_->setObjectName("evaluationSplitCombo");
    split_->addItem("Val · 验证集", "val"); split_->addItem("Test · 测试集", "test"); split_->addItem("Train · 训练集", "train");
    split_->setFixedWidth(142);
    modeRow->addWidget(split_);
    reload_ = button("重新检查", "evaluationReloadDatasetButton");
    modeRow->addWidget(reload_);
    eligibilityStatus_ = new QToolButton;
    eligibilityStatus_->setObjectName("evaluationEligibilityStatusButton");
    eligibilityStatus_->setCursor(Qt::PointingHandCursor);
    eligibilityStatus_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    eligibilityStatus_->setMinimumHeight(30);
    modeRow->addWidget(eligibilityStatus_);
    const QString toggleStyle = "QToolButton { color:#D2D2D2; background:#303030; border:1px solid #454545; border-radius:4px; padding:6px 9px; } QToolButton:hover { background:#3B3B3B; } QToolButton:checked { background:#354C59; border-color:#4A6C80; } QToolButton:disabled { color:#858585; background:#292929; border-color:#3B3B3B; }";
    datasetSettingsButton_ = new QToolButton;
    datasetSettingsButton_->setObjectName("evaluationDatasetSettingsButton");
    datasetSettingsButton_->setText("数据集设置");
    datasetSettingsButton_->setCheckable(true);
    datasetSettingsButton_->setCursor(Qt::PointingHandCursor);
    datasetSettingsButton_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    datasetSettingsButton_->setArrowType(Qt::RightArrow);
    datasetSettingsButton_->setStyleSheet(toggleStyle);
    datasetSettingsButton_->setToolTip("设置模型与数据集类别映射、审核文件及阈值扫描参数。");
    modeRow->addWidget(datasetSettingsButton_);
    configuration->addLayout(modeRow);

    datasetSettings_ = new QWidget;
    datasetSettings_->setObjectName("evaluationDatasetSettings");
    datasetSettings_->setStyleSheet("#evaluationDatasetSettings { background:#242424; border:1px solid #353535; border-radius:5px; }");
    auto *settingsLayout = new QVBoxLayout(datasetSettings_);
    settingsLayout->setContentsMargins(8, 2, 8, 2);
    settingsLayout->setSpacing(4);
    configuration->addWidget(datasetSettings_);
    datasetSettings_->hide();
    connect(datasetSettingsButton_, &QToolButton::toggled, this, [this](bool expanded) {
        datasetSettings_->setVisible(expanded);
        datasetSettingsButton_->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    });
    connect(eligibilityStatus_, &QToolButton::clicked, this, [this] {
        datasetSettingsButton_->setChecked(true);
    });

    validation_ = label({}, "evaluationValidationHint");
    validation_->setStyleSheet("color:#FFB4AB; font-size:11px;");
    validation_->setMaximumHeight(44);
    validation_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    settingsLayout->addWidget(validation_);
    datasetControls_ = new QWidget;
    auto *datasetConfiguration = new QVBoxLayout(datasetControls_);
    datasetConfiguration->setContentsMargins(0, 0, 0, 0);
    datasetConfiguration->setSpacing(2);
    settingsLayout->addWidget(datasetControls_);
    auto *mapping = new QHBoxLayout;
    mapping->setSpacing(9);
    modelId_ = new NoWheel<QSpinBox>; modelId_->setObjectName("evaluationModelClassIdSpin"); modelId_->setRange(0, 0);
    datasetId_ = new NoWheel<QSpinBox>; datasetId_->setObjectName("evaluationDatasetClassIdSpin"); datasetId_->setRange(0, 0);
    modelId_->setFixedWidth(82); datasetId_->setFixedWidth(82);
    mapping->addWidget(label("模型 ID", "muted")); mapping->addWidget(modelId_);
    mapping->addWidget(label("→ 数据集 ID", "muted")); mapping->addWidget(datasetId_);
    mappingHint_ = label({}, "evaluationMappingHint");
    mappingHint_->setWordWrap(false);
    mappingHint_->setStyleSheet("color:#B8B8B8; font-size:11px;");
    mappingHint_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    mapping->addWidget(mappingHint_, 1);
    auto *advancedToggle = new QToolButton;
    advancedToggle->setObjectName("evaluationAdvancedButton");
    advancedToggle->setText("高级参数"); advancedToggle->setCheckable(true); advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    advancedToggle->setArrowType(Qt::RightArrow);
    advancedToggle->setStyleSheet(toggleStyle);
    mapping->addWidget(advancedToggle);
    datasetConfiguration->addLayout(mapping);

    auto *advanced = new QWidget;
    advanced->setObjectName("evaluationAdvancedControls");
    auto *advancedLayout = new QVBoxLayout(advanced); advancedLayout->setContentsMargins(0, 0, 0, 0); advancedLayout->setSpacing(6);
    auto *metadataRow = new QHBoxLayout;
    metadata_ = new QLineEdit; metadata_->setObjectName("evaluationMetadataEdit");
    metadata_->setPlaceholderText("审核 metadata.csv（可选；标准 YOLO 标签无需此文件）");
    metadata_->setClearButtonEnabled(true);
    metadataRow->addWidget(metadata_, 1);
    metadataBrowse_ = button("审核文件", "evaluationMetadataBrowseButton");
    metadataRow->addWidget(metadataBrowse_);
    auto *parameters = new QHBoxLayout;
    parameters->setSpacing(8);
    matchIou_ = number("evaluationMatchIouSpin", .05, 1, .05, .5);
    scanStart_ = number("evaluationScanStartSpin", .001, .999, .01, .05, 3);
    scanEnd_ = number("evaluationScanEndSpin", .001, .999, .01, .95, 3);
    scanStep_ = number("evaluationScanStepSpin", .001, .5, .01, .01, 3);
    for (auto *spin : {matchIou_, scanStart_, scanEnd_, scanStep_}) spin->setFixedWidth(88);
    parameters->addWidget(label("匹配 IoU", "muted")); parameters->addWidget(matchIou_);
    parameters->addWidget(label("扫描起点", "muted")); parameters->addWidget(scanStart_);
    parameters->addWidget(label("终点", "muted")); parameters->addWidget(scanEnd_);
    parameters->addWidget(label("步长", "muted")); parameters->addWidget(scanStep_);
    metadata_->setMinimumWidth(100);
    metadata_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    metadata_->setToolTip("可选审核 metadata.csv；标准 YOLO 标签无需此文件。扫描复用低分预测，NMS 使用检测配置。");
    parameters->addLayout(metadataRow, 1);
    advancedLayout->addLayout(parameters);
    datasetConfiguration->addWidget(advanced); advanced->hide();
    connect(advancedToggle, &QToolButton::toggled, this, [advanced, advancedToggle](bool enabled) {
        advanced->setVisible(enabled); advancedToggle->setArrowType(enabled ? Qt::DownArrow : Qt::RightArrow);
    });

    auto *outer = new QVBoxLayout(this); outer->setContentsMargins(0, 0, 0, 0);
    auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame);
    auto *content = new QWidget; content->setObjectName("evaluationAnalysisContent");
    content->setStyleSheet("#evaluationAnalysisContent { background:#202020; }");
    auto *layout = new QVBoxLayout(content); layout->setContentsMargins(0, 0, 9, 0); layout->setSpacing(12);
    reportStatus_ = label("尚无评估报告", "evaluationReportStatus");
    reportStatus_->setStyleSheet("font-size:16px; font-weight:600; color:#F5F5F5;");
    layout->addWidget(reportStatus_);
    reportIdentity_ = label("在左侧选择评估调优，检查数据集与类别映射后开始。", "evaluationReportIdentity");
    reportIdentity_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(reportIdentity_);
    auto *metricGrid = new QGridLayout; metricGrid->setSpacing(10);
    const QStringList titles = {"mAP50", "mAP50–95", "Precision", "Recall", "F1", "无目标图片误报率"};
    for (int i = 0; i < titles.size(); ++i) {
        auto *frame = card(); auto *inside = new QVBoxLayout(frame); inside->setContentsMargins(13, 12, 13, 12);
        inside->addWidget(label(titles[i]));
        auto *value = label("N/A", qPrintable(QString("evaluationMetric%1").arg(i)));
        value->setStyleSheet("font-size:24px; font-weight:600; color:#F5F5F5;");
        inside->addWidget(value); metrics_.append(value); metricGrid->addWidget(frame, i / 3, i % 3);
    }
    layout->addLayout(metricGrid);
    counts_ = label("TP / FP / FN 与样本数量将在评估后显示。", "evaluationCounts");
    layout->addWidget(counts_);
    auto *analysis = card(); auto *analysisLayout = new QVBoxLayout(analysis);
    auto *thresholdRow = new QHBoxLayout;
    thresholdRow->addWidget(label("分析阈值", "sectionTitle"));
    threshold_ = new ThresholdSpin;
    threshold_->setObjectName("evaluationAnalysisThresholdSpin");
    threshold_->setRange(0, 1); threshold_->setDecimals(3); threshold_->setSingleStep(.01); threshold_->setValue(.25);
    threshold_->setToolTip("按报告中已计算的工作点选择；输入数值会吸附到最近的实际扫描阈值。");
    thresholdRow->addWidget(threshold_); thresholdRow->addStretch();
    apply_ = button("采用检测配置", "evaluationApplyThresholdButton"); apply_->setProperty("fluentRole", "primary");
    apply_->setToolTip("采用报告保存的输入尺寸、颜色、预处理、NMS、目标原始类别与当前分析置信度，供工作台图片、视频或摄像头检测使用。");
    thresholdRow->addWidget(apply_); analysisLayout->addLayout(thresholdRow);
    curve_ = new EvaluationCurve;
    analysisLayout->addWidget(curve_);
    analysisLayout->addWidget(label("曲线来自保存的预测。点击曲线或表格更新分析，不会自动修改工作台配置。AP 来自低置信度候选，与当前工作阈值区分。", "tiny"));
    layout->addWidget(analysis);
    auto *recommendCard = card(); auto *recommendLayout = new QVBoxLayout(recommendCard);
    auto *strategyRow = new QHBoxLayout;
    strategy_ = new NoWheel<QComboBox>; strategy_->setObjectName("evaluationRecommendationStrategyCombo");
    strategy_->addItem("均衡 · F1 最大", "f1"); strategy_->addItem("最低 Precision 下的最大 Recall", "precision"); strategy_->addItem("最高误报率下的最大 Recall", "fpr");
    strategy_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    strategy_->setMinimumContentsLength(12); strategy_->setMinimumWidth(0);
    strategyRow->addWidget(strategy_, 1);
    constraint_ = number("evaluationRecommendationConstraintSpin", 0, 1, .01, .9, 3);
    strategyRow->addWidget(constraint_);
    useRecommended_ = button("查看推荐工作点", "evaluationUseRecommendationButton");
    recommendLayout->addLayout(strategyRow);
    recommendation_ = label({}, "evaluationRecommendationHint"); recommendLayout->addWidget(recommendation_);
    auto *recommendActions = new QHBoxLayout; recommendActions->addStretch(); recommendActions->addWidget(useRecommended_);
    recommendLayout->addLayout(recommendActions);
    layout->addWidget(recommendCard);
    auto *tableCard = card(); auto *tableLayout = new QVBoxLayout(tableCard);
    auto *filterRow = new QHBoxLayout;
    filterRow->addWidget(label("阈值明细与错例", "sectionTitle"), 1);
    filter_ = new NoWheel<QComboBox>; filter_->setObjectName("evaluationImageFilterCombo");
    filter_->addItem("全部图片", "all"); filter_->addItem("包含误检", "fp"); filter_->addItem("包含漏检", "fn"); filter_->addItem("无目标图片误报", "negative_fp");
    filterRow->addWidget(filter_); tableLayout->addLayout(filterRow);
    table_ = new QTableWidget(0, 8); table_->setObjectName("evaluationThresholdTable");
    table_->setHorizontalHeaderLabels({"置信度", "Precision", "Recall", "F1", "无目标误报率", "TP", "FP", "FN"});
    table_->setSelectionBehavior(QAbstractItemView::SelectRows); table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers); table_->verticalHeader()->hide();
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setStretchLastSection(true); table_->setMinimumHeight(230);
    tableLayout->addWidget(table_); layout->addWidget(tableCard);
    auto *footer = new QHBoxLayout;
    openFolder_ = button("打开保存文件夹", "evaluationOpenDirectoryButton");
    openReport_ = button("打开报告", "evaluationOpenReportButton");
    export_ = button("导出原始评估报告", "evaluationExportReportButton");
    footer->addWidget(openFolder_); footer->addWidget(openReport_); footer->addWidget(export_); footer->addStretch();
    layout->addLayout(footer); layout->addStretch();
    scroll->setWidget(content); outer->addWidget(scroll);

    connect(mode_, &QComboBox::currentIndexChanged, this, [this] {
        if (updating_) return;
        if (evaluationMode() && !canEvaluate()) { QSignalBlocker blocker(mode_); mode_->setCurrentIndex(0); }
        updateEligibility(); emit modeChanged(evaluationMode());
    });
    connect(split_, &QComboBox::currentIndexChanged, this, [this] { if (!updating_) requestDataset(); });
    connect(reload_, &QPushButton::clicked, this, &EvaluationPanel::requestDataset);
    connect(metadata_, &QLineEdit::editingFinished, this, [this] { if (!updating_) requestDataset(); });
    connect(metadataBrowse_, &QPushButton::clicked, this, [this] {
        const QString file = QFileDialog::getOpenFileName(this, "选择审核 metadata.csv", metadata_->text(), "CSV 文件 (*.csv)");
        if (file.isEmpty()) return;
        metadata_->setText(file); requestDataset();
    });
    for (auto *control : {modelId_, datasetId_})
        connect(control, &QSpinBox::valueChanged, this, [this] { if (!updating_) { updateMapping(); updateEligibility(); emit configChanged(); } });
    for (auto *control : {matchIou_, scanStart_, scanEnd_, scanStep_})
        connect(control, &QDoubleSpinBox::valueChanged, this, [this] { if (!updating_) { updateEligibility(); emit configChanged(); } });
    connect(threshold_, &QDoubleSpinBox::valueChanged, this, [this](double threshold) {
        if (updating_) return;
        const double actual = nearestConfidence(report_.value("rows").toArray(), threshold);
        if (std::abs(actual - threshold) > 1e-8) { QSignalBlocker blocker(threshold_); threshold_->setValue(actual); }
        operatingOverride_ = {};
        curve_->setThreshold(actual); updateMetrics(); emit thresholdChanged(actual);
    });
    curve_->selected = [this](double value) { setAnalysisThreshold(value); };
    connect(table_, &QTableWidget::cellClicked, this, [this](int row, int) {
        if (auto *item = table_->item(row, 0)) setAnalysisThreshold(item->data(Qt::UserRole).toDouble());
    });
    connect(apply_, &QPushButton::clicked, this, [this] { if (!busy_ && !report_.isEmpty()) emit applyRequested(analysisThreshold()); });
    connect(filter_, &QComboBox::currentIndexChanged, this, [this] { emit imageFilterChanged(filter_->currentData().toString()); });
    connect(strategy_, &QComboBox::currentIndexChanged, this, [this] {
        if (strategy_->currentData() == "fpr") constraint_->setValue(.01);
        else if (strategy_->currentData() == "precision") constraint_->setValue(.9);
        updateRecommendation();
    });
    connect(constraint_, &QDoubleSpinBox::valueChanged, this, &EvaluationPanel::updateRecommendation);
    connect(useRecommended_, &QPushButton::clicked, this, [this] { if (recommended_ >= 0) setAnalysisThreshold(recommended_); });
    connect(openFolder_, &QPushButton::clicked, this, [this] {
        if (!directory_.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(directory_));
    });
    connect(openReport_, &QPushButton::clicked, this, [this] {
        const QString path = markdownPath(directory_);
        if (QFileInfo::exists(path)) { emit reportOpenRequested(path); QDesktopServices::openUrl(QUrl::fromLocalFile(path)); }
    });
    connect(export_, &QPushButton::clicked, this, &EvaluationPanel::exportReport);
    updateMapping(); updateEligibility(); updateRecommendation();
}

void EvaluationPanel::chooseDataset()
{
    if (busy_) return;
    const QFileInfo current(yaml_);
    const QString initial = current.isDir() ? current.absoluteFilePath() : current.absolutePath();
    const QString directory = QFileDialog::getExistingDirectory(this, "选择标准 YOLO 数据集文件夹", initial);
    if (directory.isEmpty()) return;
    setDatasetSource(directory, datasetSplit(), metadataPath());
    emit datasetRequested(yaml_, datasetSplit(), metadataPath());
}
void EvaluationPanel::chooseDatasetFile()
{
    if (busy_) return;
    const QString file = QFileDialog::getOpenFileName(this, "选择 YOLO 数据集 YAML 配置", yaml_, "YOLO 数据集 (*.yaml *.yml *.YAML *.YML)");
    if (file.isEmpty()) return;
    setDatasetSource(file, datasetSplit(), metadataPath());
    emit datasetRequested(yaml_, datasetSplit(), metadataPath());
}
void EvaluationPanel::setDatasetSource(const QString &source, const QString &split, const QString &metadata)
{
    updating_ = true;
    yaml_ = QFileInfo(source).absoluteFilePath();
    importError_.clear();
    metadata_->setText(metadata);
    const int index = split_->findData(split);
    if (index >= 0) split_->setCurrentIndex(index);
    updating_ = false;
    if (manifest_.isEmpty()) {
        datasetHint_->setText("正在检查数据集配置与标注…");
        datasetHint_->setToolTip(yaml_);
    }
    updateEligibility();
}
void EvaluationPanel::showDatasetImportError(const QString &message)
{
    importError_ = message;
    datasetHint_->setText("导入未完成 · " + message.section('\n', 0, 0));
    datasetHint_->setToolTip(message + "\n可切换集合重试，或重新选择配置。\n" + yaml_);
    updateEligibility();
}
void EvaluationPanel::requestDataset()
{
    if (busy_ || updating_ || yaml_.isEmpty()) return;
    emit datasetRequested(yaml_, datasetSplit(), metadataPath());
}
void EvaluationPanel::setDataset(const QJsonObject &manifest)
{
    const bool changed = manifest_ != manifest;
    updating_ = true;
    manifest_ = manifest;
    importError_.clear();
    yaml_ = manifest.value("dataset_source").toString(manifest.value("data_yaml").toString(yaml_));
    metadata_->setText(manifest.value("metadata_path").toString());
    const QString split = manifest.value("split").toString("val");
    const int index = split_->findData(split);
    if (index >= 0) split_->setCurrentIndex(index);
    const auto names = manifest.value("names").toArray();
    datasetId_->setMaximum(qMax(datasetId_->value(), qMax(0, int(names.size()) - 1)));
    const auto validation = manifest.value("validation").toObject();
    const int samples = manifest.value("samples").toArray().size();
    const QString datasetFile = QFileInfo(manifest.value("data_yaml").toString(yaml_)).fileName();
    const QString detail = QString("%1 · %2\n有效 %3 张 · 含框 %4 · 空标签 %5 · 缺标签 %6 · 排除 %7")
        .arg(datasetFile, split.toUpper()).arg(samples)
        .arg(validation.value("positive_images").toInt()).arg(validation.value("negative_images").toInt())
        .arg(validation.value("missing_labels").toInt()).arg(validation.value("excluded").toInt());
    datasetHint_->setText(QString("%1 · %2 张").arg(datasetFile).arg(samples));
    const QString relocation = manifest.value("relocation_notice").toString();
    datasetHint_->setToolTip(detail + "\n" + yaml_ + (relocation.isEmpty() ? QString() : "\n已使用当前目录迁移路径：" + relocation));
    updating_ = false;
    updateMapping(); updateEligibility();
    if (changed) emit configChanged();
}
void EvaluationPanel::clearDataset()
{
    updating_ = true;
    manifest_ = {}; yaml_.clear(); importError_.clear(); metadata_->clear(); split_->setCurrentIndex(split_->findData("val"));
    updating_ = false;
    datasetHint_->setText("导入 YOLO 数据集，可叠加标注并评估模型。");
    datasetHint_->setToolTip({});
    setMode(false); updateMapping(); updateEligibility(); emit configChanged();
}
void EvaluationPanel::setModel(const vision::ModelConfig &config)
{
    updating_ = true; model_ = config;
    modelId_->setMaximum(qMax(modelId_->value(), qMax(0, config.modelClassCount - 1)));
    updating_ = false; updateMapping(); updateEligibility();
}
void EvaluationPanel::setBusy(bool busy)
{
    busy_ = busy; updateEligibility();
}
void EvaluationPanel::setMode(bool evaluation)
{
    if (busy_) return;
    mode_->setCurrentIndex(evaluation && canEvaluate() ? 1 : 0);
    updateEligibility();
}
bool EvaluationPanel::evaluationMode() const { return mode_->currentData().toBool(); }
QString EvaluationPanel::validationMessage() const
{
    if (manifest_.isEmpty()) return "需要先导入带有效标签的 YOLO 数据集。";
    if (manifest_.value("requires_attention").toBool() || manifest_.value("validation").toObject().value("missing_labels").toInt() > 0)
        return "数据集存在缺失或无效标签，请修复后重新检查，不能将缺标签当作负样本。";
    if (manifest_.value("samples").toArray().isEmpty()) return "当前集合没有可评估图片。";
    if (manifest_.value("names").toArray().isEmpty()) return "数据集没有有效类别信息。";
    if (model_.task == vision::ModelTask::Classification) return "评估调优仅用于目标检测，当前模型为分类任务。";
    if (model_.modelPath.isEmpty()) return "请先在模型库选用检测模型。";
    if (model_.modelClassCount <= 0) return "正在等待模型实际类别信息，取得后才能校验映射。";
    if (modelClassId() >= model_.modelClassCount) return "模型原始类别 ID 超出模型实际范围。";
    if (datasetClassId() >= manifest_.value("names").toArray().size()) return "数据集类别 ID 超出当前数据集范围，请重新确认映射。";
    if (scanStart() > scanEnd()) return "扫描起点不能大于终点。";
    if (scanStep() <= 0 || (scanEnd() - scanStart()) / scanStep() > 1000) return "扫描点过多，请增大步长（最多 1001 个工作点）。";
    return {};
}
bool EvaluationPanel::canEvaluate() const { return validationMessage().isEmpty(); }
int EvaluationPanel::modelClassId() const { return modelId_->value(); }
int EvaluationPanel::datasetClassId() const { return datasetId_->value(); }
double EvaluationPanel::analysisThreshold() const { return threshold_->value(); }
double EvaluationPanel::matchIou() const { return matchIou_->value(); }
double EvaluationPanel::scanStart() const { return scanStart_->value(); }
double EvaluationPanel::scanEnd() const { return scanEnd_->value(); }
double EvaluationPanel::scanStep() const { return scanStep_->value(); }
QString EvaluationPanel::datasetSplit() const { return split_->currentData().toString(); }
QString EvaluationPanel::datasetYaml() const { return yaml_; }
QString EvaluationPanel::metadataPath() const { return metadata_->text().trimmed(); }
void EvaluationPanel::setAnalysisThreshold(double threshold)
{
    if (std::isfinite(threshold)) threshold_->setValue(std::clamp(threshold, 0.0, 1.0));
}
void EvaluationPanel::updateMapping()
{
    const QString modelName = modelClassId() < model_.labels.size() ? model_.labels.value(modelClassId()) : QStringLiteral("名称未知");
    const QString dataName = className(manifest_.value("names").toArray(), datasetClassId());
    QString text = QString("模型 %1 · %2 → 数据集 %3 · %4\n原始模型 ID 保持不变；此映射只用于当前目标类评估。")
        .arg(modelClassId()).arg(modelName).arg(datasetClassId()).arg(dataName.isEmpty() ? QStringLiteral("待导入") : dataName);
    if (modelClassId() == 32 && modelName.compare("sports ball", Qt::CaseInsensitive) == 0)
        text += "\nCOCO 32 是 sports ball，包含多种运动球；映射到足球不代表模型只识别足球。";
    mappingHint_->setText(text.section('\n', 0, 0));
    mappingHint_->setToolTip(text);
}
void EvaluationPanel::updateEligibility()
{
    const QString problem = validationMessage();
    const bool hasSource = !yaml_.isEmpty() || !manifest_.isEmpty();
    const bool importFailed = manifest_.isEmpty() && !importError_.isEmpty();
    const bool invalidDataset = !manifest_.isEmpty() &&
        (manifest_.value("requires_attention").toBool() ||
         manifest_.value("validation").toObject().value("missing_labels").toInt() > 0 ||
         manifest_.value("samples").toArray().isEmpty() || manifest_.value("names").toArray().isEmpty());
    const QString detail = importFailed ? importError_ : problem;
    if (auto *items = qobject_cast<QStandardItemModel *>(mode_->model()))
        if (auto *item = items->item(1)) item->setEnabled(problem.isEmpty());
    mode_->setEnabled(!busy_);
    datasetControls_->setVisible(!manifest_.isEmpty());
    datasetSettingsButton_->setVisible(hasSource);
    datasetSettings_->setVisible(hasSource && datasetSettingsButton_->isChecked());
    split_->setVisible(!yaml_.isEmpty()); reload_->setVisible(!yaml_.isEmpty());
    for (QWidget *control : QList<QWidget *>{split_, reload_, metadata_, metadataBrowse_, modelId_, datasetId_, matchIou_, scanStart_, scanEnd_, scanStep_})
        control->setEnabled(!busy_);
    reload_->setEnabled(!busy_ && !yaml_.isEmpty());
    QString summary = detail.section('\n', 0, 0).trimmed();
    if (summary.size() > 160) summary = summary.left(160) + "…";
    if (summary != detail.trimmed() && !summary.endsWith("…")) summary += "…";
    validation_->setText(summary);
    validation_->setToolTip(detail);
    validation_->setAccessibleDescription(detail);
    validation_->setVisible(!detail.isEmpty() && hasSource);
    validation_->setStyleSheet(!importFailed && !invalidDataset &&
                              (model_.task == vision::ModelTask::Classification || model_.modelPath.isEmpty() || model_.modelClassCount <= 0)
                                  ? "color:#E8C688; font-size:11px;" : "color:#FFB4AB; font-size:11px;");
    QString status, color;
    if (importFailed) { status = "导入失败"; color = "#FFB4AB"; }
    else if (manifest_.isEmpty()) { status = busy_ ? "正在检查" : "待导入"; color = "#E8C688"; }
    else if (invalidDataset) { status = "需检查"; color = "#FFB4AB"; }
    else if (problem.isEmpty()) { status = "可评估"; color = "#A6E3B0"; }
    else if (model_.task == vision::ModelTask::Classification) { status = "仅预览"; color = "#E8C688"; }
    else if (model_.modelPath.isEmpty()) { status = "待选模型"; color = "#E8C688"; }
    else if (model_.modelClassCount <= 0) { status = "等待模型"; color = "#E8C688"; }
    else { status = "需检查"; color = "#FFB4AB"; }
    eligibilityStatus_->setText(status);
    eligibilityStatus_->setVisible(hasSource);
    eligibilityStatus_->setStyleSheet(QString("QToolButton { color:%1; background:transparent; border:1px solid transparent; border-radius:4px; padding:4px 6px; font-size:11px; } QToolButton:hover { background:#333333; border-color:#454545; }").arg(color));
    eligibilityStatus_->setToolTip((detail.isEmpty() ? QStringLiteral("数据集与目标类别映射有效，可以选择评估调优。") : detail) + "\n点击查看数据集设置。");
    eligibilityStatus_->setAccessibleName("评估状态：" + status);
    const bool rows = !report_.value("rows").toArray().isEmpty();
    apply_->setEnabled(!busy_ && rows);
    threshold_->setEnabled(rows); filter_->setEnabled(rows);
    const bool files = !directory_.isEmpty() && QFileInfo(directory_).isDir();
    openFolder_->setEnabled(files); export_->setEnabled(files && !busy_ && !exporting_);
    openReport_->setEnabled(files && !markdownPath(directory_).isEmpty());
    updateRecommendation();
}

void EvaluationPanel::clearReport()
{
    setReport({});
}
void EvaluationPanel::setReport(const QJsonObject &report, const QString &directory)
{
    report_ = report; directory_ = directory; operatingOverride_ = {};
    const auto rows = report.value("rows").toArray();
    const auto request = requestOf(report);
    updating_ = true;
    const double initial = nearestConfidence(rows, request.value("analysis_threshold").toDouble(request.value("confidence").toDouble(analysisThreshold())));
    static_cast<ThresholdSpin *>(threshold_)->setRows(rows);
    threshold_->setValue(initial);
    curve_->setRows(rows); curve_->setThreshold(initial);
    table_->setRowCount(rows.size());
    const QStringList keys = {"confidence", "precision", "recall", "F1", "no_ball_false_positive_rate", "TP", "FP", "FN"};
    for (int row = 0; row < rows.size(); ++row) {
        const auto values = rows.at(row).toObject();
        for (int column = 0; column < keys.size(); ++column) {
            const auto value = values.value(keys[column]);
            QString text;
            if (column == 0) text = QString::number(value.toDouble(), 'f', 3);
            else if (column < 5) text = percentage(value);
            else text = numeric(value) ? QString::number(value.toInt()) : QStringLiteral("N/A");
            auto *item = new QTableWidgetItem(text); item->setTextAlignment(Qt::AlignCenter);
            item->setData(Qt::UserRole, value.toDouble()); table_->setItem(row, column, item);
        }
    }
    updating_ = false;
    if (report.isEmpty()) {
        reportStatus_->setText("尚无评估报告");
        reportIdentity_->setText("在左侧选择评估调优，检查数据集与类别映射后开始。");
    } else {
        const bool complete = report.value("status").toString() == "complete";
        const bool full = report.value("full_split").toBool();
        reportStatus_->setText(complete ? (full ? "完整集合评估" : "子集评估 · 不代表完整集合成绩") : "未完成评估 · 仅展示已完成样本");
        const auto manifest = report.value("manifest").toObject();
        const auto config = request.value("config").toObject().isEmpty() ? request.value("model").toObject() : request.value("config").toObject();
        QString modelPath = request.value("model_path").toString(config.value("model_path").toString());
        if (modelPath.isEmpty()) modelPath = request.value("model").toString();
        const QString split = manifest.value("split").toString(request.value("split").toString()).toUpper();
        reportIdentity_->setText(QString("%1 · %2\n模型 ID %3 → 数据集 ID %4 · 匹配 IoU %5\n报告保存其运行时配置，当前工作台配置变化不会改写历史成绩。")
            .arg(QFileInfo(modelPath).fileName(), split)
            .arg(request.value("model_class_id").toInt()).arg(request.value("dataset_class_id").toInt())
            .arg(request.value("match_iou").toDouble(.5), 0, 'f', 2));
        const QString input = config.value("input_size").toInt() > 0 ? QString::number(config.value("input_size").toInt()) + " px" : "尺寸未记录";
        const QString color = config.value("color_mode").toString() == "grayscale" ? "灰度" : "彩色";
        const QString device = config.value("resolved_device").toString(config.value("device").toString("未记录"));
        reportIdentity_->setText(reportIdentity_->text() + QString("\n%1 · %2 · %3 · NMS IoU %4\nAP 仅针对映射的目标类别；无目标图片指没有此目标类 GT 的图片。").arg(input, color, device).arg(config.value("nms_iou").toDouble(.45), 0, 'f', 2));
    }
    updateMetrics(); updateEligibility();
}
void EvaluationPanel::setOperatingMetrics(const QJsonObject &metrics)
{
    operatingOverride_ = metrics;
    updateMetrics();
}
QJsonObject EvaluationPanel::selectedRow() const
{
    if (!operatingOverride_.isEmpty() && std::abs(operatingOverride_.value("confidence").toDouble(-1) - analysisThreshold()) < 1e-7) return operatingOverride_;
    const auto rows = report_.value("rows").toArray();
    QJsonObject best;
    double delta = std::numeric_limits<double>::max();
    for (const auto &entry : rows) {
        const auto row = entry.toObject();
        const auto confidence = row.value("confidence");
        if (!numeric(confidence)) continue;
        const double candidate = std::abs(confidence.toDouble() - analysisThreshold());
        if (candidate < delta) { delta = candidate; best = row; }
    }
    return best.isEmpty() ? report_.value("metrics").toObject() : best;
}
void EvaluationPanel::updateMetrics()
{
    const auto summary = report_.value("metrics").toObject();
    const auto row = selectedRow();
    const bool headlineAP = report_.value("status").toString() == "complete" && report_.value("full_split").toBool() && summary.value("images").toInt() > 1;
    metrics_[0]->setText(headlineAP ? metricText(summary, "mAP50") : "N/A");
    metrics_[1]->setText(headlineAP ? metricText(summary, "mAP50_95") : "N/A");
    for (int i = 0; i < 2; ++i) metrics_[i]->setToolTip(headlineAP ? QString() : "主指标仅用于完整完成且超过一张图片的集合；单图、子集与中断报告不代表完整模型成绩。");
    metrics_[2]->setText(metricText(row, "precision")); metrics_[3]->setText(metricText(row, "recall"));
    metrics_[4]->setText(metricText(row, "F1")); metrics_[5]->setText(metricText(row, "no_ball_false_positive_rate"));
    if (report_.isEmpty()) { counts_->setText("TP / FP / FN 与样本数量将在评估后显示。"); return; }
    counts_->setText(QString("分析工作点 %1 · TP %2 / FP %3 / FN %4\n已完成 %5 张 · GT %6 个 · 确认无目标 %7 张，其中误报 %8 张。分母为 0 的指标显示 N/A。")
        .arg(row.value("confidence").toDouble(analysisThreshold()), 0, 'f', 3)
        .arg(row.value("TP").toInt()).arg(row.value("FP").toInt()).arg(row.value("FN").toInt())
        .arg(row.value("images").toInt()).arg(row.value("gt_boxes").toInt())
        .arg(row.value("no_ball_images").toInt()).arg(row.value("no_ball_false_positive_images").toInt()));
    int index = -1; const auto rows = report_.value("rows").toArray();
    for (int i = 0; i < rows.size(); ++i)
        if (std::abs(rows.at(i).toObject().value("confidence").toDouble() - row.value("confidence").toDouble()) < 1e-8) { index = i; break; }
    if (index >= 0) { QSignalBlocker blocker(table_); table_->selectRow(index); }
}
void EvaluationPanel::updateRecommendation()
{
    recommended_ = -1;
    constraint_->setVisible(strategy_->currentData() != "f1");
    const auto manifest = report_.value("manifest").toObject();
    const QString split = manifest.value("split").toString(requestOf(report_).value("split").toString());
    QString message;
    if (report_.isEmpty()) message = "完成完整验证集评估后，可按明确目标选择工作点。";
    else if (split == "test") message = "Test 扫描仅作描述性分析。请在 Val 上选择阈值，再固定到 Test 验证。";
    else if (split != "val") message = "训练集扫描仅作拟合分析；请在完整 Val 集上选择阈值，再固定到 Test 验证。";
    else if (report_.value("status").toString() != "complete" || !report_.value("full_split").toBool() || report_.value("metrics").toObject().value("images").toInt() <= 1)
        message = "子集或未完成报告不自动推荐部署阈值；请完成整个验证集。";
    else {
        const QString strategy = strategy_->currentData().toString();
        QJsonObject winner;
        std::array<double, 4> bestRank{-1, -1, -1, -1};
        for (const auto &entry : report_.value("rows").toArray()) {
            const auto row = entry.toObject();
            if (!numeric(row.value("confidence"))) continue;
            if (strategy == "precision" && (!numeric(row.value("precision")) || row.value("precision").toDouble() + 1e-12 < constraint_->value())) continue;
            if (strategy == "fpr" && (!numeric(row.value("no_ball_false_positive_rate")) || row.value("no_ball_false_positive_rate").toDouble() > constraint_->value() + 1e-12)) continue;
            const auto score = row.value(strategy == "f1" ? "F1" : "recall");
            if (!numeric(score) || !numeric(row.value("F1"))) continue;
            const double precision = row.value("precision").toDouble();
            const double third = row.value(strategy == "f1" ? "recall" : "F1").toDouble();
            // Match the report's deterministic recommendation: after quality
            // ties, prefer the higher confidence on a flat operating plateau.
            const std::array<double, 4> rank{score.toDouble(), precision, third, row.value("confidence").toDouble()};
            if (rank > bestRank) { winner = row; bestRank = rank; }
        }
        if (winner.isEmpty()) message = "无满足条件的工作点（或所需指标为 N/A）。不会自动放宽限制。";
        else {
            recommended_ = winner.value("confidence").toDouble();
            message = QString("建议查看阈值 %1：P %2 · R %3 · F1 %4 · 无目标误报率 %5。\n点击查看后，可检查错例；只有“采用检测配置”才更新检测工作台。")
                .arg(recommended_, 0, 'f', 3).arg(metricText(winner, "precision"), metricText(winner, "recall"), metricText(winner, "F1"), metricText(winner, "no_ball_false_positive_rate"));
        }
    }
    recommendation_->setText(message); useRecommended_->setEnabled(recommended_ >= 0 && !busy_);
}
void EvaluationPanel::exportReport()
{
    if (busy_ || directory_.isEmpty() || !QFileInfo(directory_).isDir()) return;
    const QString parent = QFileDialog::getExistingDirectory(this, "选择报告导出位置");
    if (parent.isEmpty()) return;
    exportReportTo(parent);
}
void EvaluationPanel::exportReportTo(const QString &parent)
{
    if (busy_ || exporting_ || directory_.isEmpty() || parent.isEmpty() || !QFileInfo(directory_).isDir()) return;
    const QString source = directory_;
    const QString destination = QDir(parent).filePath(QFileInfo(source).fileName() + "-export-" + QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss-zzz"));
    exporting_ = true;
    export_->setEnabled(false);
    const QJsonObject selection{{"analysis_threshold", analysisThreshold()}, {"operating_metrics", selectedRow()},
        {"source_report", QFileInfo(source).canonicalFilePath()}, {"note", "原始报告及扫描表保持不变，此文件仅记录导出时的分析选择。"}};
    const QPointer<EvaluationPanel> guard(this);
    QThreadPool::globalInstance()->start(QRunnable::create([guard, source, destination, selection] {
        QString error = copyReport(source, destination);
        if (error.isEmpty()) {
            const QString selectionPath = destination + "/analysis-selection.json";
            if (QFileInfo::exists(selectionPath)) error = "导出报告中已有分析选择文件，拒绝覆盖。";
            else {
                QSaveFile snapshot(selectionPath);
                const QByteArray bytes = QJsonDocument(selection).toJson();
                if (!snapshot.open(QIODevice::WriteOnly) || snapshot.write(bytes) != bytes.size() || !snapshot.commit())
                    error = "报告已复制，但分析选择文件写入失败。";
            }
        }
        if (!guard) return;
        QMetaObject::invokeMethod(guard, [guard, error, destination] {
            if (!guard) return;
            guard->exporting_ = false;
            guard->updateEligibility();
            guard->recommendation_->setText(error.isEmpty() ? "原始报告已导出，并附带当前分析选择：" + destination : error);
        }, Qt::QueuedConnection);
    }));
}
