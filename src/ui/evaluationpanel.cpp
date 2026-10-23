// SPDX-License-Identifier: AGPL-3.0-only
#include "evaluationpanel.h"

#include <QAbstractItemView>
#include <QApplication>
#include <QComboBox>
#include <QCheckBox>
#include <QDialogButtonBox>
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
#include <QHash>
#include <QGridLayout>
#include <QJsonDocument>
#include <QSaveFile>
#include <QJsonArray>
#include <QLabel>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPushButton>
#include <QRunnable>
#include <QScrollArea>
#include <QScreen>
#include <QShortcut>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTabWidget>
#include <QTabBar>
#include <QTextDocument>
#include <QTextEdit>
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
// QFileDialog's built-in file modes treat a selected directory as navigation.
// Keep its normal browser and double-click navigation, but explicitly accept a
// directory when the user presses the primary button or Enter in the name edit.
class DatasetSourceDialog final : public QFileDialog
{
public:
    explicit DatasetSourceDialog(QWidget *parent, const QString &source) : QFileDialog(parent)
    {
        setObjectName("yoloDatasetSourceDialog");
        setWindowTitle("导入 YOLO 数据集 · 文件夹或 YAML");
        QPalette colors = parent ? parent->palette() : palette();
        colors.setColor(QPalette::Window, QColor("#202020"));
        colors.setColor(QPalette::WindowText, QColor("#F5F5F5"));
        colors.setColor(QPalette::Base, QColor("#303030"));
        colors.setColor(QPalette::Text, QColor("#F5F5F5"));
        colors.setColor(QPalette::Button, QColor("#333333"));
        colors.setColor(QPalette::ButtonText, QColor("#F5F5F5"));
        colors.setColor(QPalette::Highlight, QColor("#354C59"));
        colors.setColor(QPalette::HighlightedText, Qt::white);
        setPalette(colors);
        setOption(QFileDialog::DontUseNativeDialog);
        setOption(QFileDialog::ReadOnly);
        setFileMode(QFileDialog::AnyFile);
        setAcceptMode(QFileDialog::AcceptOpen);
        setNameFilter("YOLO 数据集配置 (*.yaml *.yml *.YAML *.YML)");
        setFilter(QDir::AllDirs | QDir::Files | QDir::NoDotAndDotDot);
        setLabelText(QFileDialog::FileName, "文件夹或配置：");
        setLabelText(QFileDialog::Accept, "导入所选项");
        const QFileInfo current(source);
        setDirectory(source.isEmpty() ? QDir::homePath() : current.isDir() ? current.absoluteFilePath() : current.absolutePath());
        if (current.isFile()) selectFile(current.fileName());
        hint_ = new QLabel("单击文件夹后点击导入，或选择 YAML 配置；双击文件夹可继续浏览。", this);
        hint_->setObjectName("yoloDatasetSourceHint");
        hint_->setTextFormat(Qt::PlainText);
        hint_->setWordWrap(true);
        hint_->setStyleSheet("color:#B8B8B8; font-size:12px; padding:4px 0;");
        if (auto *grid = qobject_cast<QGridLayout *>(layout())) grid->addWidget(hint_, grid->rowCount(), 0, 1, grid->columnCount());
        else if (layout()) layout()->addWidget(hint_);
        connect(this, &QFileDialog::currentChanged, this, [this](const QString &path) {
            selected_ = path; selectionDirectory_ = directory().absolutePath(); refresh(path);
        });
        connect(this, &QFileDialog::directoryEntered, this, [this](const QString &) { selected_.clear(); refresh({}); });
        if (auto *edit = findChild<QLineEdit *>("fileNameEdit"))
        {
            connect(edit, &QLineEdit::textChanged, this, [this] { refresh({}); });
            connect(edit, &QLineEdit::textEdited, this, [this] { selected_.clear(); });
        }
        refresh({});
        resize(840, 580);
    }
    QString source() const { return source_; }
    void accept() override
    {
        QString candidate;
        if (auto *edit = findChild<QLineEdit *>("fileNameEdit")) candidate = edit->text().trimmed();
        if (!selected_.isEmpty() && selectionDirectory_ == directory().absolutePath()) candidate = selected_;
        if (candidate.startsWith('"') && candidate.endsWith('"') && candidate.size() > 1) candidate = candidate.mid(1, candidate.size() - 2);
        if (candidate.isEmpty()) candidate = selectedFiles().value(0, directory().absolutePath());
        if (QDir::isRelativePath(candidate)) candidate = directory().absoluteFilePath(candidate);
        const QFileInfo info(candidate);
        const QString suffix = info.suffix().toLower();
        if ((!info.isDir() && (!info.isFile() || (suffix != "yaml" && suffix != "yml"))) || !info.isReadable()) {
            hint_->setText("请选择可读取的现有文件夹，或 .yaml／.yml 数据集配置文件。");
            hint_->setStyleSheet("color:#FFB4AB; font-size:12px; padding:4px 0;");
            return;
        }
        source_ = info.absoluteFilePath();
        QDialog::accept();
    }
protected:
    void showEvent(QShowEvent *event) override
    {
        QFileDialog::showEvent(event);
        refresh({});
    }
private:
    void refresh(const QString &selected)
    {
        const QString candidate = selected.isEmpty() ? selectedFiles().value(0, directory().absolutePath()) : selected;
        const QFileInfo info(candidate);
        hint_->setText(info.isDir() ? "将导入此文件夹；双击文件夹可继续浏览。也可直接选择 YAML 配置。"
                                    : "可导入现有文件夹或 YAML 配置；双击文件夹可继续浏览。");
        hint_->setStyleSheet("color:#B8B8B8; font-size:12px; padding:4px 0;");
        if (auto *buttons = findChild<QDialogButtonBox *>())
            if (auto *open = buttons->button(QDialogButtonBox::Open)) {
                open->setText(info.isDir() ? "导入此文件夹" : "导入所选项");
                open->setEnabled(true);
            }
    }
    QLabel *hint_ = nullptr;
    QString source_, selected_, selectionDirectory_;
};
class DatasetSettingsPopup final : public QFrame
{
public:
    explicit DatasetSettingsPopup(QWidget *parent) : QFrame(parent, Qt::Popup | Qt::FramelessWindowHint)
    {
        setAttribute(Qt::WA_NoMouseReplay);
        setFocusPolicy(Qt::StrongFocus);
    }
protected:
    void keyPressEvent(QKeyEvent *event) override
    {
        if (event->key() == Qt::Key_Escape) { hide(); event->accept(); }
        else QFrame::keyPressEvent(event);
    }
    void mousePressEvent(QMouseEvent *event) override
    {
        if (!rect().contains(event->position().toPoint())) {
            setAttribute(Qt::WA_NoMouseReplay);
            hide(); event->accept(); return;
        }
        QFrame::mousePressEvent(event);
    }
};
// A selectable plain-text error can contain one very long path or traceback
// token. QTextEdit provides wrapping and selection without inserting characters
// into the original text; the outer settings scroller owns vertical scrolling.
class WrappedErrorLabel final : public QLabel
{
public:
    WrappedErrorLabel()
    {
        editor_ = new QTextEdit(this);
        editor_->setReadOnly(true);
        editor_->setAcceptRichText(false);
        editor_->setFrameShape(QFrame::NoFrame);
        editor_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        editor_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        editor_->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
        editor_->document()->setDocumentMargin(0);
        editor_->setTabChangesFocus(true);
        QSizePolicy policy(QSizePolicy::Ignored, QSizePolicy::Minimum);
        policy.setHeightForWidth(true); setSizePolicy(policy);
    }
    void setContent(const QString &text, const QString &color)
    {
        QLabel::setText(text);
        editor_->setPlainText(text);
        editor_->setStyleSheet(QString("QTextEdit { color:%1; background:transparent; border:0; padding:0; font-size:11px; }").arg(color));
        editor_->ensurePolished();
        editor_->setAccessibleName("数据集校验详情");
        editor_->setAccessibleDescription(text);
        updateGeometry();
    }
    int heightForWidth(int width) const override
    {
        QTextDocument document;
        document.setDefaultFont(editor_->font());
        document.setDocumentMargin(0);
        auto option = document.defaultTextOption();
        option.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
        document.setDefaultTextOption(option);
        document.setPlainText(text());
        document.setTextWidth(qMax(1, width - 4));
        return int(std::ceil(document.size().height())) + 4;
    }
    QSize sizeHint() const override { return {320, heightForWidth(320)}; }
    QSize minimumSizeHint() const override { return {0, editor_->fontMetrics().lineSpacing() + 4}; }
protected:
    void resizeEvent(QResizeEvent *event) override
    {
        QLabel::resizeEvent(event);
        editor_->setGeometry(contentsRect());
    }
    void paintEvent(QPaintEvent *) override {}
private:
    QTextEdit *editor_ = nullptr;
};
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
QString scalar(const QJsonValue &value, int decimals = 3)
{
    return numeric(value) ? QString::number(value.toDouble(), 'f', decimals) : QStringLiteral("N/A");
}
QString sceneTitle(const QString &name)
{
    static const QHash<QString, QString> titles{{"All", "全部样本"}, {"Small", "小目标"}, {"Occluded", "遮挡"},
        {"Blurred", "模糊"}, {"No Ball", "无目标"}, {"No difficulty", "无困难标记"}};
    return titles.value(name, name);
}
void configureReadOnlyTable(QTableWidget *table)
{
    table->setStyleSheet("QTableWidget { color:#F5F5F5; background:#272727; border:1px solid #353535; border-radius:4px; } QHeaderView { background:#303030; color:#B8B8B8; }");
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table->horizontalHeader()->setStretchLastSection(true);
    table->setAlternatingRowColors(false);
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

// Diagnostic plots consume report data only. They never execute inference or
// invent samples for missing fields in an older report.
class EvaluationPlot final : public QWidget
{
public:
    EvaluationPlot() { setMinimumSize(380, 300); setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); }
    void setReport(const QJsonObject &report) { report_ = report; update(); }
    void setKind(const QString &kind) { kind_ = kind; update(); }
protected:
    void paintEvent(QPaintEvent *) override
    {
        struct Series { QString title; QColor color; QList<QPointF> points; };
        QList<Series> lines;
        const auto summary = report_.value("metrics").toObject();
        const auto rows = report_.value("rows").toArray();
        QString xLabel = "置信度阈值", yLabel = "比例";
        double xMax = 1, yMax = 1, xMin = 0;
        const QList<QColor> colors{QColor("#60CDFF"), QColor("#A6E3B0"), QColor("#D4B6FF"), QColor("#FFB88A")};
        auto append = [&lines, &colors](const QString &title, const QJsonArray &data, const QString &xKey, const QString &yKey) {
            Series series{title, colors.at(lines.size() % colors.size()), {}};
            for (const auto &entry : data) {
                const auto item = entry.toObject();
                if (numeric(item.value(xKey)) && numeric(item.value(yKey))) series.points.append({item.value(xKey).toDouble(), item.value(yKey).toDouble()});
            }
            std::stable_sort(series.points.begin(), series.points.end(), [](const QPointF &a, const QPointF &b) { return a.x() < b.x(); });
            lines.append(series);
        };
        if (kind_ == "pr") {
            xLabel = "Recall"; yLabel = "Precision";
            const auto pr = report_.value("pr_curve").toObject();
            const auto recall = pr.value("recall").toArray(), precision = pr.value("precision_envelope").toArray();
            Series series{"PR · IoU 0.50", colors[0], {}};
            for (int i = 0; i < qMin(recall.size(), precision.size()); ++i)
                if (numeric(recall[i]) && numeric(precision[i])) series.points.append({recall[i].toDouble(), precision[i].toDouble()});
            lines.append(series);
        } else if (kind_ == "ap") {
            xLabel = "匹配 IoU"; yLabel = "目标类 AP"; xMin = .5;
            Series series{"AP–IoU", colors[0], {}};
            const auto values = summary.value("ap_per_iou").toArray();
            for (int i = 0; i < values.size(); ++i) if (numeric(values[i])) series.points.append({.5 + .05 * i, values[i].toDouble()});
            lines.append(series);
        } else if (kind_ == "froc") {
            xLabel = "每图误报框数 FPPI"; yLabel = "Recall";
            append("FROC", rows, "fp_boxes_per_image", "recall");
            xMax = 0; for (const auto &line : lines) for (const auto &point : line.points) xMax = qMax(xMax, point.x());
            if (xMax <= 0) xMax = 1;
        } else if (kind_ == "counts") {
            yLabel = "目标框数量"; yMax = 0;
            for (const QString &key : {QString("TP"), QString("FP"), QString("FN")}) append(key, rows, "confidence", key);
            for (const auto &line : lines) for (const auto &point : line.points) yMax = qMax(yMax, point.y());
            if (yMax <= 0) yMax = 1;
        } else if (kind_ == "negative") {
            yLabel = "无目标图片误报率";
            append("无目标图片误报率", rows, "confidence", "no_ball_false_positive_rate");
        }
        QPainter painter(this); painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#272727")); painter.setFont(QFont(font().family(), 9));
        const QRectF plot(66, 48, qMax(1, width() - 92), qMax(1, height() - 100));
        for (int tick = 0; tick <= 5; ++tick) {
            const double fraction = tick / 5.0, x = plot.left() + plot.width() * fraction, y = plot.bottom() - plot.height() * fraction;
            painter.setPen(QPen(QColor("#414141"), 1, Qt::DotLine));
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom())); painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
            painter.setPen(QColor("#BCBEC3"));
            painter.drawText(QRectF(x - 28, plot.bottom() + 7, 56, 20), Qt::AlignCenter, QString::number(xMin + (xMax - xMin) * fraction, 'g', 3));
            painter.drawText(QRectF(0, y - 10, 55, 20), Qt::AlignRight | Qt::AlignVCenter, QString::number(yMax * fraction, 'g', 3));
        }
        painter.setPen(QColor("#D2D2D2"));
        painter.drawText(QRectF(plot.left(), height() - 24, plot.width(), 20), Qt::AlignCenter, xLabel);
        painter.drawText(QRectF(plot.left(), 4, plot.width(), 20), Qt::AlignRight, yLabel);
        bool any = false; int legend = int(plot.left());
        for (const auto &line : lines) {
            painter.setPen(QPen(line.color, 2.2)); painter.drawLine(legend, 30, legend + 16, 30);
            painter.setPen(QColor("#E4E4E4")); painter.drawText(legend + 22, 34, line.title);
            legend += painter.fontMetrics().horizontalAdvance(line.title) + 42;
            QPainterPath path; bool first = true;
            painter.save(); painter.setClipRect(plot.adjusted(-2, -2, 2, 2));
            for (const auto &point : line.points) {
                any = true;
                const QPointF mapped(plot.left() + (point.x() - xMin) / (xMax - xMin) * plot.width(), plot.bottom() - point.y() / yMax * plot.height());
                if (first) path.moveTo(mapped); else path.lineTo(mapped); first = false;
                if (line.points.size() == 1) { painter.setBrush(line.color); painter.drawEllipse(mapped, 3, 3); }
            }
            painter.setPen(QPen(line.color, 2.2)); painter.drawPath(path); painter.restore();
        }
        if (!any) { painter.setPen(QColor("#BCBEC3")); painter.drawText(plot, Qt::AlignCenter | Qt::TextWordWrap, "此报告没有可用曲线数据\n旧报告可尝试使用预测快照重新分析；缺失指标不会以零值代替。"); }
    }
private:
    QJsonObject report_;
    QString kind_ = "pr";
};

EvaluationPanel::EvaluationPanel(QWidget *parent) : QWidget(parent)
{
    setObjectName("evaluationPanel");
    QPalette colors = palette();
    colors.setColor(QPalette::Window, QColor("#202020")); colors.setColor(QPalette::WindowText, QColor("#F5F5F5"));
    colors.setColor(QPalette::Base, QColor("#303030")); colors.setColor(QPalette::AlternateBase, QColor("#272727"));
    colors.setColor(QPalette::Text, QColor("#F5F5F5")); colors.setColor(QPalette::Button, QColor("#333333"));
    colors.setColor(QPalette::ButtonText, QColor("#F5F5F5")); colors.setColor(QPalette::Highlight, QColor("#354C59"));
    colors.setColor(QPalette::HighlightedText, Qt::white); colors.setColor(QPalette::Midlight, QColor("#353535"));
    colors.setColor(QPalette::Light, QColor("#353535")); setPalette(colors);
    controls_ = new QWidget(this);
    controls_->setObjectName("evaluationControls");
    controls_->setStyleSheet("#evaluationControls { background:#272727; }");
    controls_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto *configuration = new QVBoxLayout(controls_);
    configuration->setContentsMargins(0, 2, 0, 2);
    configuration->setSpacing(2);
    auto *modeRow = new QHBoxLayout;
    modeRow->setSpacing(8);
    modeTitle_ = label("工作模式", "sectionTitle");
    modeTitle_->setObjectName("evaluationModeLabel");
    modeRow->addWidget(modeTitle_);
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

    datasetSettings_ = new DatasetSettingsPopup(controls_);
    datasetSettings_->setObjectName("evaluationDatasetSettings");
    datasetSettings_->setStyleSheet("#evaluationDatasetSettings { background:#272727; border:1px solid #505050; border-radius:8px; } #evaluationDatasetSettingsContent, #evaluationDatasetSettingsScroll, #evaluationDatasetSettingsScroll > QWidget > QWidget { background:#272727; }");
    datasetSettings_->setAccessibleName("数据集设置");
    auto *popupLayout = new QVBoxLayout(datasetSettings_);
    popupLayout->setContentsMargins(14, 12, 14, 12);
    popupLayout->setSpacing(10);
    auto *popupHeader = new QHBoxLayout;
    popupHeader->addWidget(label("数据集设置", "sectionTitle"));
    popupHeader->addStretch();
    auto *closeSettings = new QToolButton;
    closeSettings->setObjectName("evaluationDatasetSettingsCloseButton");
    closeSettings->setText("×");
    closeSettings->setToolTip("关闭数据集设置（Esc）");
    closeSettings->setAccessibleName("关闭数据集设置");
    closeSettings->setStyleSheet("QToolButton { color:#D2D2D2; background:transparent; border:0; border-radius:4px; padding:2px 8px; font-size:18px; } QToolButton:hover { background:#3B3B3B; }");
    popupHeader->addWidget(closeSettings);
    popupLayout->addLayout(popupHeader);
    datasetSettingsScroll_ = new QScrollArea;
    datasetSettingsScroll_->setObjectName("evaluationDatasetSettingsScroll");
    datasetSettingsScroll_->setFrameShape(QFrame::NoFrame);
    datasetSettingsScroll_->setWidgetResizable(true);
    datasetSettingsScroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    datasetSettingsContent_ = new QWidget;
    datasetSettingsContent_->setObjectName("evaluationDatasetSettingsContent");
    auto *settingsLayout = new QVBoxLayout(datasetSettingsContent_);
    settingsLayout->setContentsMargins(0, 0, 4, 0);
    settingsLayout->setSpacing(10);
    datasetSettingsScroll_->setWidget(datasetSettingsContent_);
    popupLayout->addWidget(datasetSettingsScroll_, 1);
    connect(closeSettings, &QToolButton::clicked, this, &EvaluationPanel::hideDatasetSettings);
    auto *dismissSettings = new QShortcut(QKeySequence(Qt::Key_Escape), datasetSettings_);
    dismissSettings->setContext(Qt::WidgetWithChildrenShortcut);
    connect(dismissSettings, &QShortcut::activated, this, &EvaluationPanel::hideDatasetSettings);
    controls_->installEventFilter(this);
    datasetSettings_->installEventFilter(this);
    datasetSettings_->hide();
    connect(datasetSettingsButton_, &QToolButton::toggled, this, [this](bool expanded) {
        if (expanded) showDatasetSettings(); else hideDatasetSettings();
    });
    connect(eligibilityStatus_, &QToolButton::clicked, this, [this] {
        showDatasetSettings();
    });

    validation_ = new WrappedErrorLabel;
    validation_->setObjectName("evaluationValidationHint");
    validation_->setStyleSheet("color:#FFB4AB; font-size:11px;");
    validation_->setMinimumWidth(0);
    settingsLayout->addWidget(validation_);
    datasetControls_ = new QWidget;
    auto *datasetConfiguration = new QVBoxLayout(datasetControls_);
    datasetConfiguration->setContentsMargins(0, 0, 0, 0);
    datasetConfiguration->setSpacing(10);
    settingsLayout->addWidget(datasetControls_);
    auto *mapping = new QHBoxLayout;
    mapping->setSpacing(9);
    modelId_ = new NoWheel<QSpinBox>; modelId_->setObjectName("evaluationModelClassIdSpin"); modelId_->setRange(0, 0);
    datasetId_ = new NoWheel<QSpinBox>; datasetId_->setObjectName("evaluationDatasetClassIdSpin"); datasetId_->setRange(0, 0);
    modelId_->setFixedWidth(82); datasetId_->setFixedWidth(82);
    mapping->addWidget(label("模型 ID", "muted")); mapping->addWidget(modelId_);
    mapping->addWidget(label("→ 数据集 ID", "muted")); mapping->addWidget(datasetId_);
    mappingHint_ = label({}, "evaluationMappingHint");
    mappingHint_->setWordWrap(true);
    mappingHint_->setStyleSheet("color:#B8B8B8; font-size:11px;");
    mappingHint_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    mapping->addStretch();
    auto *advancedToggle = new QToolButton;
    advancedToggle->setObjectName("evaluationAdvancedButton");
    advancedToggle->setText("高级参数"); advancedToggle->setCheckable(true); advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    advancedToggle->setArrowType(Qt::RightArrow);
    advancedToggle->setStyleSheet(toggleStyle);
    mapping->addWidget(advancedToggle);
    datasetConfiguration->addLayout(mapping);
    datasetConfiguration->addWidget(mappingHint_);

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
    auto *parameters = new QGridLayout;
    parameters->setHorizontalSpacing(10);
    parameters->setVerticalSpacing(8);
    matchIou_ = number("evaluationMatchIouSpin", .05, 1, .05, .5);
    scanStart_ = number("evaluationScanStartSpin", .001, .999, .01, .05, 3);
    scanEnd_ = number("evaluationScanEndSpin", .001, .999, .01, .95, 3);
    scanStep_ = number("evaluationScanStepSpin", .001, .5, .01, .01, 3);
    for (auto *spin : {matchIou_, scanStart_, scanEnd_, scanStep_}) spin->setFixedWidth(88);
    parameters->addWidget(label("匹配 IoU", "muted"), 0, 0); parameters->addWidget(matchIou_, 0, 1);
    parameters->addWidget(label("扫描起点", "muted"), 0, 2); parameters->addWidget(scanStart_, 0, 3);
    parameters->addWidget(label("扫描终点", "muted"), 1, 0); parameters->addWidget(scanEnd_, 1, 1);
    parameters->addWidget(label("扫描步长", "muted"), 1, 2); parameters->addWidget(scanStep_, 1, 3);
    parameters->setColumnStretch(0, 1); parameters->setColumnStretch(2, 1);
    auto *inferenceParameters = new QGridLayout;
    inferenceParameters->setHorizontalSpacing(10); inferenceParameters->setVerticalSpacing(8);
    batch_ = new NoWheel<QSpinBox>; batch_->setObjectName("evaluationBatchSizeSpin"); batch_->setRange(1, 32); batch_->setValue(1);
    threads_ = new NoWheel<QSpinBox>; threads_->setObjectName("evaluationCpuThreadsSpin"); threads_->setRange(0, 256); threads_->setSpecialValueText("自动");
    maximumDetections_ = new NoWheel<QSpinBox>; maximumDetections_->setObjectName("evaluationMaxDetectionsSpin"); maximumDetections_->setRange(1, 10000); maximumDetections_->setValue(300);
    scoreFloor_ = number("evaluationScoreFloorSpin", .000001, .999, .001, .001, 6);
    inferenceParameters->addWidget(label("推理 batch", "muted"), 0, 0); inferenceParameters->addWidget(batch_, 0, 1);
    inferenceParameters->addWidget(label("CPU 线程", "muted"), 0, 2); inferenceParameters->addWidget(threads_, 0, 3);
    inferenceParameters->addWidget(label("候选分数下限", "muted"), 1, 0); inferenceParameters->addWidget(scoreFloor_, 1, 1);
    inferenceParameters->addWidget(label("每图最多目标", "muted"), 1, 2); inferenceParameters->addWidget(maximumDetections_, 1, 3);
    for (auto *spin : {batch_, threads_, maximumDetections_}) spin->setMinimumWidth(88);
    threads_->setToolTip("0 保留后端默认值；显式线程数仅支持独立 PT／TorchScript Python 进程。ONNX 请选择自动。");
    batch_->setToolTip("PT 使用真正的批量推理；ONNX／TorchScript 保持逐张执行，并在报告记录实际 batch 与原因。");
    scoreFloor_->setToolTip("采集候选的最低置信度；必须不高于扫描起点。提高此值可能截断 PR／AP，修改后需要重新推理。");
    metadata_->setMinimumWidth(100);
    metadata_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    metadata_->setToolTip("可选审核 metadata.csv；标准 YOLO 标签无需此文件。扫描复用低分预测，NMS 使用检测配置。");
    advancedLayout->addLayout(parameters);
    advancedLayout->addLayout(inferenceParameters);
    advancedLayout->addWidget(label("PT 可批量推理；ONNX／TorchScript 实际 batch=1。CPU 线程 0 保留后端默认值。", "tiny"));
    advancedLayout->addLayout(metadataRow);
    advancedLayout->addWidget(label("扫描复用低分预测，NMS 使用检测配置。标准 YOLO 标签无需审核文件。", "tiny"));
    datasetConfiguration->addWidget(advanced); advanced->hide();
    connect(advancedToggle, &QToolButton::toggled, this, [this, advanced, advancedToggle](bool enabled) {
        advanced->setVisible(enabled); advancedToggle->setArrowType(enabled ? Qt::DownArrow : Qt::RightArrow);
        if (datasetSettings_->isVisible()) positionDatasetSettings();
    });

    auto *outer = new QVBoxLayout(this); outer->setContentsMargins(0, 0, 0, 0);
    auto *content = new QWidget; content->setObjectName("evaluationAnalysisContent");
    content->setStyleSheet("#evaluationAnalysisContent { background:#202020; }");
    auto *layout = new QVBoxLayout(content); layout->setContentsMargins(0, 0, 9, 0); layout->setSpacing(12);
    reportStatus_ = label("尚无评估报告", "evaluationReportStatus");
    reportStatus_->setStyleSheet("font-size:16px; font-weight:600; color:#F5F5F5;");
    layout->addWidget(reportStatus_);
    reportIdentity_ = label("导入带标注的 YOLO 数据集，检查类别映射后点击开始评估。", "evaluationReportIdentity");
    reportIdentity_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(reportIdentity_);
    auto *thresholdHeader = card(); auto *thresholdHeaderLayout = new QHBoxLayout(thresholdHeader);
    thresholdHeaderLayout->addWidget(label("分析阈值", "sectionTitle"));
    threshold_ = new ThresholdSpin;
    threshold_->setObjectName("evaluationAnalysisThresholdSpin");
    threshold_->setRange(0, 1); threshold_->setDecimals(3); threshold_->setSingleStep(.01); threshold_->setValue(.25);
    threshold_->setToolTip("选择报告中已计算的工作点；如需新的网格或匹配 IoU，请使用“报告与重分析”。");
    thresholdHeaderLayout->addWidget(threshold_); thresholdHeaderLayout->addStretch();
    apply_ = button("采用检测配置", "evaluationApplyThresholdButton"); apply_->setProperty("fluentRole", "primary");
    apply_->setToolTip("明确采用报告保存的模型推理设置与当前分析阈值，供工作台检测使用。");
    thresholdHeaderLayout->addWidget(apply_); layout->addWidget(thresholdHeader);
    analysisTabs_ = new QTabWidget; analysisTabs_->setObjectName("evaluationAnalysisTabs");
    analysisTabs_->setDocumentMode(true);
    analysisTabs_->tabBar()->setDrawBase(false);
    analysisTabs_->setStyleSheet(
        "QTabWidget#evaluationAnalysisTabs::pane { background:#202020; border:0; }"
        "QTabBar::tab { background:#272727; color:#B8B8B8; border:1px solid #353535; border-bottom:2px solid transparent; border-top-left-radius:4px; border-top-right-radius:4px; padding:7px 13px; margin-right:4px; }"
        "QTabBar::tab:selected { background:#303030; color:#F5F5F5; border-bottom:2px solid #60CDFF; }"
        "QTabBar::tab:hover { background:#353535; color:#F5F5F5; }");
    auto makePage = [this](const QString &title, const char *name) {
        auto *pageScroll = new QScrollArea; pageScroll->setWidgetResizable(true); pageScroll->setFrameShape(QFrame::NoFrame);
        pageScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto *page = new QWidget; page->setObjectName(name);
        page->setStyleSheet(QString("#%1 { background:#202020; }").arg(name));
        auto *pageLayout = new QVBoxLayout(page); pageLayout->setContentsMargins(3, 12, 9, 6); pageLayout->setSpacing(12);
        pageScroll->setWidget(page); analysisTabs_->addTab(pageScroll, title); return pageLayout;
    };
    auto *overviewLayout = makePage("评估总览", "evaluationOverviewPage");
    auto *thresholdLayout = makePage("阈值明细", "evaluationThresholdPage");
    auto *diagnosticLayout = makePage("诊断图表", "evaluationDiagnosticsPage");
    auto *sceneLayout = makePage("场景分析", "evaluationScenesPage");
    auto *reportLayout = makePage("报告与重分析", "evaluationReportPage");
    layout->addWidget(analysisTabs_, 1);
    auto *metricGrid = new QGridLayout; metricGrid->setSpacing(10);
    const QStringList titles = {"mAP50", "mAP50–95", "Precision", "Recall", "F1", "无目标图片误报率"};
    for (int i = 0; i < titles.size(); ++i) {
        auto *frame = card(); auto *inside = new QVBoxLayout(frame); inside->setContentsMargins(13, 12, 13, 12);
        inside->addWidget(label(titles[i]));
        auto *value = label("N/A", qPrintable(QString("evaluationMetric%1").arg(i)));
        value->setStyleSheet("font-size:24px; font-weight:600; color:#F5F5F5;");
        inside->addWidget(value); metrics_.append(value); metricGrid->addWidget(frame, i / 3, i % 3);
    }
    overviewLayout->addLayout(metricGrid);
    details_ = new QTableWidget(3, 4); details_->setObjectName("evaluationExtendedMetricsTable");
    configureReadOnlyTable(details_); details_->setSelectionMode(QAbstractItemView::NoSelection);
    details_->horizontalHeader()->hide(); details_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    details_->setMinimumHeight(122); details_->setMaximumHeight(140);
    overviewLayout->addWidget(details_);
    counts_ = label("TP / FP / FN 与样本数量将在评估后显示。", "evaluationCounts");
    overviewLayout->addWidget(counts_);
    errorSummary_ = label({}, "evaluationErrorSummaryLabel");
    errorSummary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    errorSummary_->setStyleSheet("color:#F9CC9D; font-weight:600;");
    errorSummary_->hide();
    overviewLayout->addWidget(errorSummary_);
    auto *analysis = card(); auto *analysisLayout = new QVBoxLayout(analysis);
    curve_ = new EvaluationCurve;
    analysisLayout->addWidget(curve_);
    analysisLayout->addWidget(label("曲线来自保存的预测。点击曲线或表格更新分析，不会自动修改工作台配置。AP 来自低置信度候选，与当前工作阈值区分。", "tiny"));
    thresholdLayout->addWidget(analysis);
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
    overviewLayout->addWidget(recommendCard); overviewLayout->addStretch();
    auto *tableCard = card(); auto *tableLayout = new QVBoxLayout(tableCard);
    auto *filterRow = new QHBoxLayout;
    filterRow->addWidget(label("阈值明细与错例", "sectionTitle"), 1);
    filter_ = new NoWheel<QComboBox>; filter_->setObjectName("evaluationImageFilterCombo");
    filter_->addItem("全部图片", "all"); filter_->addItem("包含误检", "fp"); filter_->addItem("包含漏检", "fn"); filter_->addItem("无目标图片误报", "negative_fp");
    filter_->addItem("全部错例（误检或漏检）", "errors");
    filterRow->addWidget(filter_); tableLayout->addLayout(filterRow);
    table_ = new QTableWidget(0, 8); table_->setObjectName("evaluationThresholdTable");
    configureReadOnlyTable(table_);
    table_->setHorizontalHeaderLabels({"置信度", "Precision", "Recall", "F1", "无目标误报率", "TP", "FP", "FN"});
    table_->setSelectionBehavior(QAbstractItemView::SelectRows); table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers); table_->verticalHeader()->hide();
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setStretchLastSection(true); table_->setMinimumHeight(230);
    tableLayout->addWidget(table_); thresholdLayout->addWidget(tableCard); thresholdLayout->addStretch();

    auto *plotHeader = new QHBoxLayout; plotHeader->addWidget(label("分析保存的预测", "sectionTitle"));
    plotKind_ = new NoWheel<QComboBox>; plotKind_->setObjectName("evaluationDiagnosticPlotCombo");
    plotKind_->addItem("PR 曲线 · IoU 0.50", "pr"); plotKind_->addItem("AP–IoU", "ap");
    plotKind_->addItem("FROC · Recall–FPPI", "froc"); plotKind_->addItem("TP / FP / FN–阈值", "counts");
    plotKind_->addItem("无目标误报率–阈值", "negative"); plotHeader->addStretch(); plotHeader->addWidget(plotKind_);
    diagnosticLayout->addLayout(plotHeader);
    detailPlot_ = new EvaluationPlot; detailPlot_->setObjectName("evaluationDiagnosticPlot"); diagnosticLayout->addWidget(detailPlot_, 1);
    diagnosticLayout->addWidget(label("PR 与 AP 取自低分候选；FROC 和计数曲线来自实际扫描工作点。分母为零或旧报告未保存的数据保持 N/A。", "tiny"));
    auto *matrices = new QHBoxLayout;
    auto makeMatrix = [matrices](const QString &title, const char *name, QTableWidget **table) {
        auto *frame = card(); auto *matrixLayout = new QVBoxLayout(frame); matrixLayout->addWidget(label(title, "sectionTitle"));
        *table = new QTableWidget(2, 2); (*table)->setObjectName(name); configureReadOnlyTable(*table);
        (*table)->setHorizontalHeaderLabels({"预测目标", "预测无目标"}); (*table)->setVerticalHeaderLabels({"真实目标", "真实无目标"});
        (*table)->verticalHeader()->show(); (*table)->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        (*table)->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff); (*table)->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        (*table)->setSelectionMode(QAbstractItemView::NoSelection); (*table)->setFixedHeight(108); matrixLayout->addWidget(*table); matrices->addWidget(frame);
    };
    makeMatrix("检测定位 · 按 IoU 匹配目标框", "evaluationLocalizationMatrix", &localizationMatrix_);
    makeMatrix("图像存在性 · 是否出现目标", "evaluationPresenceMatrix", &presenceMatrix_);
    diagnosticLayout->addLayout(matrices);
    diagnosticLayout->addWidget(label("定位矩阵统计目标框，背景 TN 没有定义；存在性矩阵统计图片，只判断是否检出目标，不代表框定位正确。", "tiny"));

    sceneHint_ = label("场景分组需要审核 metadata；无目标组可直接按目标类别 GT 推导。", "evaluationSceneHint");
    sceneLayout->addWidget(sceneHint_);
    scenes_ = new QTableWidget(0, 12); scenes_->setObjectName("evaluationSceneMetricsTable"); configureReadOnlyTable(scenes_);
    scenes_->setHorizontalHeaderLabels({"场景", "审核状态", "图片", "AP50", "AP75", "AP50–95", "Precision", "Recall", "F1", "误报率", "TP / FP / FN", "工作阈值"});
    scenes_->setMinimumHeight(280); sceneLayout->addWidget(scenes_, 1);
    reviewScene_ = button("查看此场景错例", "evaluationReviewSceneButton");
    auto *sceneActions = new QHBoxLayout; sceneActions->addWidget(label("分组可能重叠；未审核不等于无困难。AP 发布条件与本报告的评估范围一致。", "tiny"), 1); sceneActions->addWidget(reviewScene_);
    sceneLayout->addLayout(sceneActions);

    auto *exportCard = card(); auto *exportLayout = new QVBoxLayout(exportCard);
    exportLayout->addWidget(label("新报告图表输出", "sectionTitle"));
    auto *formats = new QHBoxLayout;
    figurePng_ = new QCheckBox("PNG"); figurePng_->setObjectName("evaluationFigurePngCheck"); figurePng_->setChecked(true);
    figureSvg_ = new QCheckBox("SVG"); figureSvg_->setObjectName("evaluationFigureSvgCheck"); figureSvg_->setChecked(true);
    figurePdf_ = new QCheckBox("PDF"); figurePdf_->setObjectName("evaluationFigurePdfCheck");
    figurePdf_->setChecked(true);
    formats->addWidget(figurePng_); formats->addWidget(figureSvg_); formats->addWidget(figurePdf_); formats->addStretch();
    formats->addWidget(label("DPI", "muted")); figureDpi_ = new NoWheel<QSpinBox>; figureDpi_->setObjectName("evaluationFigureDpiSpin");
    figureDpi_->setRange(72, 1200); figureDpi_->setValue(300); formats->addWidget(figureDpi_); exportLayout->addLayout(formats);
    exportLayout->addWidget(label("输出格式用于新评估与重新分析生成的图表。导出原始报告仅复制原文件，不改写历史成果。", "tiny"));
    reportLayout->addWidget(exportCard);
    auto *reanalyzeCard = card(); auto *reanalyzeLayout = new QVBoxLayout(reanalyzeCard);
    reanalyzeLayout->addWidget(label("从预测快照重新分析", "sectionTitle"));
    auto *reanalyzeGrid = new QGridLayout; reanalyzeGrid->setHorizontalSpacing(12); reanalyzeGrid->setVerticalSpacing(10);
    reanalyzeIou_ = number("evaluationReanalysisIouSpin", .05, 1, .05, .5);
    reanalyzeStart_ = number("evaluationReanalysisStartSpin", 0, 1, .01, .05, 3);
    reanalyzeEnd_ = number("evaluationReanalysisEndSpin", 0, 1, .01, .95, 3);
    reanalyzeStep_ = number("evaluationReanalysisStepSpin", .001, .5, .01, .01, 3);
    reanalyzeConfidence_ = number("evaluationReanalysisConfidenceSpin", 0, 1, .01, .25, 3);
    const QList<QPair<QString, QDoubleSpinBox *>> analysisFields{{"匹配 IoU", reanalyzeIou_}, {"报告工作阈值", reanalyzeConfidence_},
        {"扫描起点", reanalyzeStart_}, {"扫描终点", reanalyzeEnd_}, {"扫描步长", reanalyzeStep_}};
    for (int i = 0; i < analysisFields.size(); ++i) {
        reanalyzeGrid->addWidget(label(analysisFields[i].first, "muted"), i / 2, (i % 2) * 2);
        reanalyzeGrid->addWidget(analysisFields[i].second, i / 2, (i % 2) * 2 + 1);
    }
    reanalyzeLayout->addLayout(reanalyzeGrid);
    reanalyzeLayout->addWidget(label("复用报告内的 GT 和预测，不重新运行模型。只调整匹配与分析条件；更改模型、预处理或 NMS 需要新的评估。生成独立报告，原报告保持不变。", "tiny"));
    analysisValidation_ = label({}, "evaluationReanalysisValidationLabel"); analysisValidation_->setStyleSheet("color:#E8C688;"); reanalyzeLayout->addWidget(analysisValidation_);
    reanalyze_ = button("重新分析并生成新报告", "evaluationReanalyzeButton"); reanalyze_->setProperty("fluentRole", "primary");
    auto *reanalyzeActions = new QHBoxLayout; reanalyzeActions->addStretch(); reanalyzeActions->addWidget(reanalyze_); reanalyzeLayout->addLayout(reanalyzeActions);
    reportLayout->addWidget(reanalyzeCard);
    auto *footer = new QHBoxLayout;
    openFolder_ = button("打开保存文件夹", "evaluationOpenDirectoryButton");
    openReport_ = button("打开报告", "evaluationOpenReportButton");
    export_ = button("导出原始评估报告", "evaluationExportReportButton");
    importExternal_ = button("导入脚本评估报告", "evaluationImportExternalReportButton");
    importExternal_->setToolTip("选择 model-evaluation 脚本的报告目录，后台校验 GT／预测快照并生成独立应用报告；不需要重新推理或选用当前模型。");
    footer->addWidget(openFolder_); footer->addWidget(openReport_); footer->addWidget(export_); footer->addWidget(importExternal_); footer->addStretch();
    reportLayout->addLayout(footer); reportLayout->addStretch();
    // Only each page scrolls. An additional outer scroller gives the tab stack
    // its size hint and can push its bottom outside a compact window.
    outer->addWidget(content, 1);

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
    for (auto *control : {matchIou_, scanStart_, scanEnd_, scanStep_, scoreFloor_})
        connect(control, &QDoubleSpinBox::valueChanged, this, [this] { if (!updating_) { updateEligibility(); emit configChanged(); } });
    for (auto *control : {batch_, threads_, maximumDetections_})
        connect(control, &QSpinBox::valueChanged, this, [this] { if (!updating_) { updateEligibility(); emit configChanged(); } });
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
    connect(apply_, &QPushButton::clicked, this, [this] {
        if (!busy_ && !report_.isEmpty() && report_.value("allow_apply_profile").toBool(true)) emit applyRequested(analysisThreshold());
    });
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
    connect(plotKind_, &QComboBox::currentIndexChanged, this, [this] { detailPlot_->setKind(plotKind_->currentData().toString()); });
    for (auto *spin : {reanalyzeIou_, reanalyzeStart_, reanalyzeEnd_, reanalyzeStep_, reanalyzeConfidence_})
        connect(spin, &QDoubleSpinBox::valueChanged, this, &EvaluationPanel::updateAnalysisEligibility);
    for (auto *check : {figurePng_, figureSvg_, figurePdf_}) connect(check, &QCheckBox::toggled, this, [this] {
        updateAnalysisEligibility(); if (!updating_) updateEligibility();
    });
    connect(reanalyze_, &QPushButton::clicked, this, [this] {
        updateAnalysisEligibility(); if (reanalyze_->isEnabled()) emit reanalysisRequested(analysisOptions());
    });
    connect(importExternal_, &QPushButton::clicked, this, [this] {
        if (!importExternal_->isEnabled()) return;
        const QString directory = QFileDialog::getExistingDirectory(this, "导入脚本评估报告目录");
        if (!directory.isEmpty() && !busy_) emit externalReportImportRequested(directory, analysisOptions());
    });
    connect(scenes_, &QTableWidget::itemSelectionChanged, this, &EvaluationPanel::updateAnalysisEligibility);
    connect(reviewScene_, &QPushButton::clicked, this, [this] {
        const int row = scenes_->currentRow(); if (!busy_ && row >= 0 && reviewScene_->isEnabled())
            emit sceneReviewRequested(scenes_->item(row, 0)->data(Qt::UserRole).toString());
    });
    updateMapping(); updateEligibility(); updateRecommendation();
}

EvaluationPanel::~EvaluationPanel()
{
    if (qApp) qApp->removeEventFilter(this);
    if (datasetSettingsHost_) datasetSettingsHost_->removeEventFilter(this);
    if (controls_) controls_->removeEventFilter(this);
    if (datasetSettings_) {
        datasetSettings_->removeEventFilter(this);
        datasetSettings_->hide();
    }
}
bool EvaluationPanel::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == datasetSettings_ && (event->type() == QEvent::Hide || event->type() == QEvent::Destroy)) {
        if (qApp) qApp->removeEventFilter(this);
        if (datasetSettingsButton_) {
            QSignalBlocker blocker(datasetSettingsButton_);
            datasetSettingsButton_->setChecked(false);
            datasetSettingsButton_->setArrowType(Qt::RightArrow);
        }
    } else if (datasetSettings_ && datasetSettings_->isVisible() && event->type() == QEvent::KeyPress &&
               static_cast<QKeyEvent *>(event)->key() == Qt::Key_Escape) {
        hideDatasetSettings(); event->accept(); return true;
    } else if (datasetSettings_ && datasetSettings_->isVisible() && event->type() == QEvent::MouseButtonPress) {
        auto *target = qobject_cast<QWidget *>(watched);
        const auto *mouse = static_cast<QMouseEvent *>(event);
        const QPoint global = mouse->globalPosition().toPoint();
        const bool inPopup = datasetSettings_->rect().contains(datasetSettings_->mapFromGlobal(global));
        // Descendant popup windows (for example a combobox list) are allowed
        // to extend outside the settings card while editing its controls.
        const bool childPopup = target && target != datasetSettings_ && datasetSettings_->isAncestorOf(target);
        if (!inPopup && !childPopup) {
            const bool anchor = datasetSettingsButton_ && datasetSettingsButton_->rect().contains(datasetSettingsButton_->mapFromGlobal(global));
            hideDatasetSettings();
            if (anchor || target == datasetSettings_) { event->accept(); return true; }
        }
    } else if (datasetSettings_ && datasetSettings_->isVisible() &&
               (watched == controls_ || watched == datasetSettingsHost_)) {
        switch (event->type()) {
        case QEvent::Hide:
        case QEvent::HideToParent:
        case QEvent::Move:
        case QEvent::Resize:
        case QEvent::Close:
        case QEvent::ParentChange:
            hideDatasetSettings();
            break;
        default: break;
        }
    }
    return QWidget::eventFilter(watched, event);
}
void EvaluationPanel::hideDatasetSettings()
{
    if (qApp) qApp->removeEventFilter(this);
    if (datasetSettings_) datasetSettings_->hide();
    if (datasetSettingsButton_) {
        QSignalBlocker blocker(datasetSettingsButton_);
        datasetSettingsButton_->setChecked(false);
        datasetSettingsButton_->setArrowType(Qt::RightArrow);
    }
}
void EvaluationPanel::positionDatasetSettings()
{
    auto *screen = datasetSettingsButton_->screen();
    const QRect available = screen ? screen->availableGeometry() : QRect(0, 0, 1280, 800);
    const int width = qMin(620, qMax(1, available.width() - 24));
    const int bodyWidth = qMax(1, width - 48);
    validation_->setMaximumWidth(bodyWidth);
    validation_->setMinimumHeight(validation_->isHidden() ? 0 : validation_->heightForWidth(bodyWidth));
    mappingHint_->setMaximumWidth(bodyWidth);
    auto *bodyLayout = datasetSettingsContent_->layout();
    bodyLayout->invalidate();
    bodyLayout->activate();
    const int bodyHeight = bodyLayout->hasHeightForWidth() ? bodyLayout->heightForWidth(bodyWidth) : bodyLayout->sizeHint().height();
    datasetSettingsContent_->setMinimumHeight(bodyHeight);
    const int height = qMin(qMax(140, bodyHeight + 68), qMax(1, available.height() - 24));
    datasetSettings_->setFixedSize(width, height);
    const QRect anchor(datasetSettingsButton_->mapToGlobal(QPoint()), datasetSettingsButton_->size());
    const int left = std::clamp(anchor.right() - width + 1, available.left() + 8, available.right() - width - 7);
    int top = anchor.bottom() + 8;
    if (top + height > available.bottom() - 8) top = anchor.top() - height - 8;
    top = std::clamp(top, available.top() + 8, available.bottom() - height - 7);
    datasetSettings_->move(left, top);
}
void EvaluationPanel::showDatasetSettings()
{
    if (busy_ || (yaml_.isEmpty() && manifest_.isEmpty()) || !datasetSettingsButton_->isVisible()) {
        hideDatasetSettings(); return;
    }
    QWidget *host = controls_ ? controls_->window() : nullptr;
    if (datasetSettingsHost_ != host) {
        if (datasetSettingsHost_) datasetSettingsHost_->removeEventFilter(this);
        datasetSettingsHost_ = host;
        if (host) host->installEventFilter(this);
    }
    positionDatasetSettings();
    {
        QSignalBlocker blocker(datasetSettingsButton_);
        datasetSettingsButton_->setChecked(true);
        datasetSettingsButton_->setArrowType(Qt::DownArrow);
    }
    datasetSettings_->show();
    if (qApp) qApp->installEventFilter(this);
    datasetSettings_->raise();
    datasetSettings_->setFocus(Qt::PopupFocusReason);
}
void EvaluationPanel::chooseDataset()
{
    if (busy_) return;
    emit datasetSelectionStarted();
    hideDatasetSettings();
    DatasetSourceDialog dialog(controls_ ? controls_->window() : this, yaml_);
    if (dialog.exec() != QDialog::Accepted || dialog.source().isEmpty()) return;
    setDatasetSource(dialog.source(), datasetSplit(), metadataPath());
    emit datasetRequested(yaml_, datasetSplit(), metadataPath());
}
void EvaluationPanel::chooseDatasetFile()
{
    chooseDataset();
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
    busy_ = busy;
    if (busy_) hideDatasetSettings();
    updateEligibility();
}
void EvaluationPanel::setMode(bool evaluation)
{
    if (busy_) return;
    mode_->setCurrentIndex(evaluation && canEvaluate() ? 1 : 0);
    updateEligibility();
}
void EvaluationPanel::setModeControlVisible(bool visible)
{
    mode_->setVisible(visible);
    modeTitle_->setVisible(visible);
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
    if (scoreFloor() > scanStart()) return "候选分数下限不能高于扫描起点，否则无法完整分析这些工作点。";
    if (scoreFloor() > model_.confidence) return "候选分数下限不能高于当前检测置信度，请降低候选下限或调整检测置信度。";
    if (QFileInfo(model_.modelPath).suffix().compare("onnx", Qt::CaseInsensitive) == 0 && cpuThreads() != 0)
        return "ONNX 评估暂不支持指定 CPU 线程数，请将 CPU 线程设为自动。";
    if (!figurePng_->isChecked() && !figureSvg_->isChecked() && !figurePdf_->isChecked()) return "请在报告与重分析页至少选择一种图表输出格式。";
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
int EvaluationPanel::batchSize() const { return batch_->value(); }
int EvaluationPanel::cpuThreads() const { return threads_->value(); }
double EvaluationPanel::scoreFloor() const { return scoreFloor_->value(); }
int EvaluationPanel::maxDetections() const { return maximumDetections_->value(); }
QJsonObject EvaluationPanel::exportOptions() const
{
    QJsonArray formats;
    if (figurePng_->isChecked()) formats.append("png");
    if (figureSvg_->isChecked()) formats.append("svg");
    if (figurePdf_->isChecked()) formats.append("pdf");
    return {{"figure_formats", formats}, {"figure_dpi", figureDpi_->value()}};
}
QJsonObject EvaluationPanel::analysisOptions() const
{
    auto options = exportOptions();
    options.insert("confidence", reanalyzeConfidence_->value()); options.insert("match_iou", reanalyzeIou_->value());
    options.insert("scan_start", reanalyzeStart_->value()); options.insert("scan_end", reanalyzeEnd_->value()); options.insert("scan_step", reanalyzeStep_->value());
    return options;
}
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
    mappingHint_->setText(text);
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
    datasetSettingsButton_->setEnabled(!busy_);
    eligibilityStatus_->setEnabled(!busy_);
    if (!hasSource || busy_) hideDatasetSettings();
    split_->setVisible(!yaml_.isEmpty()); reload_->setVisible(!yaml_.isEmpty());
    for (QWidget *control : QList<QWidget *>{split_, reload_, metadata_, metadataBrowse_, modelId_, datasetId_, matchIou_, scanStart_, scanEnd_, scanStep_, batch_, threads_, maximumDetections_, scoreFloor_, figurePng_, figureSvg_, figurePdf_, figureDpi_})
        control->setEnabled(!busy_);
    reload_->setEnabled(!busy_ && !yaml_.isEmpty());
    const bool warning = !importFailed && !invalidDataset &&
        (model_.task == vision::ModelTask::Classification || model_.modelPath.isEmpty() || model_.modelClassCount <= 0);
    static_cast<WrappedErrorLabel *>(validation_)->setContent(detail, warning ? "#E8C688" : "#FFB4AB");
    validation_->setToolTip(detail);
    validation_->setAccessibleDescription(detail);
    validation_->setVisible(!detail.isEmpty() && hasSource);
    validation_->setStyleSheet(warning ? "color:#E8C688; font-size:11px;" : "color:#FFB4AB; font-size:11px;");
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
    eligibilityStatus_->setToolTip((detail.isEmpty() ? QStringLiteral("数据集与目标类别映射有效，可以开始评估。") : detail) + "\n点击查看数据集设置。");
    eligibilityStatus_->setAccessibleName("评估状态：" + status);
    if (datasetSettings_->isVisible()) positionDatasetSettings();
    const bool rows = !report_.value("rows").toArray().isEmpty();
    const bool mayApply = report_.value("allow_apply_profile").toBool(true);
    apply_->setEnabled(!busy_ && rows && mayApply);
    apply_->setToolTip(mayApply ? QStringLiteral("明确采用报告保存的模型推理设置与当前分析阈值，供工作台检测使用。") :
        QStringLiteral("此导入报告没有完整的应用推理配置，可分析阈值与错例，不能采用到检测工作台。"));
    threshold_->setEnabled(rows && !busy_); filter_->setEnabled(rows && !busy_);
    table_->setEnabled(rows && !busy_); curve_->setEnabled(rows && !busy_);
    const bool files = !directory_.isEmpty() && QFileInfo(directory_).isDir();
    openFolder_->setEnabled(files); export_->setEnabled(files && !busy_ && !exporting_);
    openReport_->setEnabled(files && !markdownPath(directory_).isEmpty());
    updateAnalysisEligibility();
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
    reanalyzeIou_->setValue(request.value("match_iou").toDouble(.5));
    reanalyzeStart_->setValue(request.value("scan_start").toDouble(.05));
    reanalyzeEnd_->setValue(request.value("scan_end").toDouble(.95));
    reanalyzeStep_->setValue(request.value("scan_step").toDouble(.01));
    reanalyzeConfidence_->setValue(request.value("confidence").toDouble(request.value("analysis_threshold").toDouble(.25)));
    // Output preferences belong to the next product, not the immutable source
    // report. Opening a historic report does not silently change them.
    setImageFilter("all", false);
    const double initial = nearestConfidence(rows, request.value("analysis_threshold").toDouble(request.value("confidence").toDouble(analysisThreshold())));
    static_cast<ThresholdSpin *>(threshold_)->setRows(rows);
    threshold_->setValue(initial);
    curve_->setRows(rows); curve_->setThreshold(initial);
    detailPlot_->setReport(report);
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
        reportIdentity_->setText("导入带标注的 YOLO 数据集，检查类别映射后点击开始评估。");
        reportIdentity_->setToolTip({});
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
        const QString color = config.value("color_mode").toString() == "grayscale" ? "灰度" :
            config.value("color_mode").toString() == "color" ? "彩色" : "颜色未记录";
        const QString device = config.value("resolved_device").toString(config.value("device").toString("未记录"));
        reportIdentity_->setText(reportIdentity_->text() + QString("\n%1 · %2 · %3 · NMS IoU %4\nAP 仅针对映射的目标类别；无目标图片指没有此目标类 GT 的图片。").arg(input, color, device).arg(config.value("nms_iou").toDouble(.45), 0, 'f', 2));
        reportIdentity_->setToolTip(reportIdentity_->text());
        reportIdentity_->setText(QString("%1 · %2 · 模型 ID %3 → 数据集 ID %4\n%5 · %6 · %7 · NMS %8 · 匹配 IoU %9")
            .arg(QFileInfo(modelPath).fileName(), split).arg(request.value("model_class_id").toInt()).arg(request.value("dataset_class_id").toInt())
            .arg(input, color, device).arg(config.value("nms_iou").toDouble(.45), 0, 'f', 2).arg(request.value("match_iou").toDouble(.5), 0, 'f', 2));
    }
    errorSummary_->setVisible(!report.isEmpty());
    setErrorSummary(0, 0, 0, 0, 0, manifest_.value("samples").toArray().size(), !report.isEmpty(), false);
    updateScenes(); updateMetrics(); updateEligibility();
}
QString EvaluationPanel::imageFilter() const { return filter_->currentData().toString(); }
void EvaluationPanel::setImageFilter(const QString &filter, bool notify)
{
    const int index = filter_->findData(filter);
    if (index < 0) return;
    if (notify) {
        if (filter_->currentIndex() == index) emit imageFilterChanged(filter);
        else filter_->setCurrentIndex(index);
    } else {
        QSignalBlocker blocker(filter_);
        filter_->setCurrentIndex(index);
    }
}
void EvaluationPanel::setErrorSummary(int evaluated, int errors, int falsePositive, int falseNegative,
                                    int negativeFalsePositive, int collection, bool pending,
                                    bool available, const QString &error)
{
    errorSummary_->setVisible(!report_.isEmpty());
    const QString text = pending ? QStringLiteral("正在整理当前阈值的错例…") :
        !available ? (error.isEmpty() ? QStringLiteral("当前阈值的错例统计尚未就绪。") :
                       QStringLiteral("错例统计不可用 · ") + error.section('\n', 0, 0)) :
        QString("错图 %1 / 已评估 %2 / 集合 %3 · 误检图 %4 · 漏检图 %5 · 无目标误报图 %6")
            .arg(errors).arg(evaluated).arg(collection).arg(falsePositive).arg(falseNegative).arg(negativeFalsePositive);
    errorSummary_->setText(text);
    errorSummary_->setToolTip(error.isEmpty() ? text + "\n错图按图片去重：包含误检或漏检；仅统计本次已完成评估的样本。" : error);
}
void EvaluationPanel::setOperatingMetrics(const QJsonObject &metrics)
{
    operatingOverride_ = metrics;
    updateMetrics();
}
QJsonObject EvaluationPanel::selectedRow() const
{
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
    if (best.isEmpty()) best = report_.value("metrics").toObject();
    if (!operatingOverride_.isEmpty() && std::abs(operatingOverride_.value("confidence").toDouble(-1) - analysisThreshold()) < 1e-7) {
        // Review computes the IoU match for overlays. Older C++ review schemas
        // omit diagnostic metrics already saved for this exact work point.
        for (auto it = operatingOverride_.begin(); it != operatingOverride_.end(); ++it) best.insert(it.key(), it.value());
    }
    return best;
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
    const auto interval = row.value("no_ball_fpr_wilson95").toArray();
    const QString wilson = interval.size() == 2 && numeric(interval[0]) && numeric(interval[1]) ? percentage(interval[0]) + " ～ " + percentage(interval[1]) : QStringLiteral("N/A");
    const QList<QPair<QString, QString>> extended{{"AP75", headlineAP ? metricText(summary, "mAP75") : QStringLiteral("N/A")},
        {"每图误报框数 FPPI", scalar(row.value("fp_boxes_per_image"))},
        {"无目标图误报框／图", scalar(row.value("no_ball_fp_boxes_per_image"))},
        {"正样本整图漏检率", metricText(row, "positive_image_miss_rate")},
        {"误报率 Wilson 95%", wilson},
        {"无目标图误报框数", numeric(row.value("no_ball_false_positive_boxes")) ? QString::number(row.value("no_ball_false_positive_boxes").toInt()) : QStringLiteral("N/A")}};
    for (int i = 0; i < extended.size(); ++i) {
        const int r = i / 2, c = (i % 2) * 2;
        auto *title = new QTableWidgetItem(extended[i].first), *value = new QTableWidgetItem(extended[i].second);
        title->setForeground(QColor("#B8B8B8")); value->setForeground(QColor("#F5F5F5"));
        title->setToolTip(extended[i].first); value->setToolTip(extended[i].second);
        details_->setItem(r, c, title); details_->setItem(r, c + 1, value);
    }
    auto fillMatrix = [](QTableWidget *table, const QJsonObject &data, bool localization) {
        const QStringList keys{"TP", "FN", "FP", "TN"};
        for (int i = 0; i < 4; ++i) {
            const auto value = data.value(keys[i]);
            const QString text = localization && i == 3 ? QStringLiteral("不定义") : numeric(value) ? QString::number(value.toInt()) : QStringLiteral("N/A");
            auto *item = new QTableWidgetItem(keys[i] + "  " + text); item->setTextAlignment(Qt::AlignCenter);
            item->setForeground(i == 0 || i == 3 ? QColor("#A6E3B0") : QColor("#FFB88A")); table->setItem(i / 2, i % 2, item);
        }
    };
    fillMatrix(localizationMatrix_, row, true); fillMatrix(presenceMatrix_, row.value("presence_confusion").toObject(), false);
    const auto sceneReview = report_.value("scene_review").toObject();
    const double sceneConfidence = requestOf(report_).value("confidence").toDouble(requestOf(report_).value("analysis_threshold").toDouble(.25));
    sceneHint_->setText(report_.isEmpty() ? "完成评估后显示场景分组；审核未知不作为无困难。" : sceneReview.isEmpty() ?
        QString("旧报告未记录逐图场景审核状态。分组表保留已保存的数据，缺失项显示 N/A。\n分组表使用报告工作阈值 %1；当前分析阈值 %2。需要其他工作点的场景报告，请重新分析。")
            .arg(sceneConfidence, 0, 'f', 3).arg(analysisThreshold(), 0, 'f', 3) :
        QString("场景工作点 %1 · 已审核 %2 张 / 未审核 %3 张。\n分组表使用报告生成时的阈值；当前分析阈值 %4。需要其他工作点的完整场景报告，请重新分析。")
            .arg(sceneConfidence, 0, 'f', 3).arg(sceneReview.value("reviewed_images").toInt()).arg(sceneReview.value("unreviewed_images").toInt())
            .arg(analysisThreshold(), 0, 'f', 3));
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
void EvaluationPanel::updateScenes()
{
    const auto groups = report_.value("groups").toObject();
    const QStringList names{"All", "Small", "Occluded", "Blurred", "No Ball", "No difficulty"};
    QSignalBlocker blocker(scenes_); scenes_->setRowCount(report_.isEmpty() ? 0 : names.size());
    for (int row = 0; row < scenes_->rowCount(); ++row) {
        const QString name = names[row]; const auto group = groups.value(name).toObject();
        // Old reports stored only operating metrics for three groups. Preserve
        // those facts without treating absent AP or review status as known.
        const bool available = !group.isEmpty() && group.value("available").toBool(true);
        const QString status = group.value("review_status").toString();
        const QString review = status == "reviewed" ? "已审核" : status == "derived_from_gt" ? "按 GT 推导" :
            status == "not_applicable" ? "整体" : status == "unreviewed" ? "未审核" : "旧报告未记录";
        const QStringList text{sceneTitle(name), review,
            available && numeric(group.value("images")) ? QString::number(group.value("images").toInt()) : QStringLiteral("N/A"),
            available ? metricText(group, "mAP50") : QStringLiteral("N/A"), available ? metricText(group, "mAP75") : QStringLiteral("N/A"),
            available ? metricText(group, "mAP50_95") : QStringLiteral("N/A"), available ? metricText(group, "precision") : QStringLiteral("N/A"),
            available ? metricText(group, "recall") : QStringLiteral("N/A"), available ? metricText(group, "F1") : QStringLiteral("N/A"),
            available ? metricText(group, "no_ball_false_positive_rate") : QStringLiteral("N/A"),
            available && numeric(group.value("TP")) ? QString("%1 / %2 / %3").arg(group.value("TP").toInt()).arg(group.value("FP").toInt()).arg(group.value("FN").toInt()) : QStringLiteral("N/A"),
            available ? scalar(group.value("confidence")) : QStringLiteral("N/A")};
        for (int column = 0; column < text.size(); ++column) {
            auto *item = new QTableWidgetItem(text[column]); item->setTextAlignment(column ? Qt::AlignCenter : Qt::AlignLeft | Qt::AlignVCenter);
            item->setToolTip(text[column]); if (!available) item->setForeground(QColor("#909090"));
            if (column == 0) {
                item->setData(Qt::UserRole, name);
                item->setData(Qt::UserRole + 1, group.value("available").toBool() &&
                    group.value("sample_images").isArray() && !group.value("sample_images").toArray().isEmpty() && group.value("images").toInt() > 0);
            }
            scenes_->setItem(row, column, item);
        }
    }
    if (scenes_->rowCount()) scenes_->selectRow(0);
}
void EvaluationPanel::updateAnalysisEligibility()
{
    QString optionProblem;
    if (reanalyzeStart_->value() > reanalyzeEnd_->value()) optionProblem = "扫描起点不能大于终点。";
    else if ((reanalyzeEnd_->value() - reanalyzeStart_->value()) / reanalyzeStep_->value() > 1000) optionProblem = "扫描点过多，请增大步长（最多 1001 个工作点）。";
    else if (!figurePng_->isChecked() && !figureSvg_->isChecked() && !figurePdf_->isChecked()) optionProblem = "请至少选择一种图表输出格式。";
    importExternal_->setEnabled(!busy_ && !exporting_ && optionProblem.isEmpty());
    QString problem = optionProblem;
    if (problem.isEmpty()) {
        if (report_.isEmpty()) problem = "打开或完成一份评估报告后，可以重新分析保存的预测；也可直接导入脚本评估报告。";
        else if (directory_.isEmpty() || !QFileInfo(directory_).isDir()) problem = "报告尚无可读取的保存目录，无法重新分析。";
        else if (report_.value("rows").toArray().isEmpty()) problem = "报告没有已完成样本的工作点，不能重新分析。";
        else if (qMin(reanalyzeStart_->value(), reanalyzeConfidence_->value()) + 1e-12 < requestOf(report_).value("score_floor").toDouble(.001))
            problem = "分析或扫描阈值低于原预测采集下限；需要降低候选下限后重新评估。";
    }
    reanalyze_->setEnabled(!busy_ && !exporting_ && problem.isEmpty());
    analysisValidation_->setText(busy_ ? QStringLiteral("任务运行期间不能修改或重新分析。") : problem);
    for (auto *control : {reanalyzeIou_, reanalyzeStart_, reanalyzeEnd_, reanalyzeStep_, reanalyzeConfidence_}) control->setEnabled(!busy_);
    const int row = scenes_->currentRow();
    reviewScene_->setEnabled(!busy_ && row >= 0 && scenes_->item(row, 0) && scenes_->item(row, 0)->data(Qt::UserRole + 1).toBool() &&
        !directory_.isEmpty() && QFileInfo(directory_).isDir());
    reviewScene_->setToolTip(reviewScene_->isEnabled() ? QStringLiteral("按当前分析阈值查看此场景内的误检或漏检图片。") :
        QStringLiteral("此场景未审核、没有样本或报告未保存场景名单。旧报告可重新分析；未审核场景需提供审核信息后重新评估。"));
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
