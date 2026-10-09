#include "modelconversionpage.h"
#include "icons.h"

#include <QComboBox>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QRunnable>
#include <QScrollArea>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QThreadPool>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
constexpr qsizetype protocolLimit = 1024 * 1024;
QLabel *label(const QString &value, const char *role)
{
    auto *result = new QLabel(value);
    result->setObjectName(role);
    result->setWordWrap(true);
    result->setTextFormat(Qt::PlainText);
    return result;
}
QPushButton *action(const QString &value, const QString &symbol, const char *name, bool primary = false)
{
    auto *result = new QPushButton(value);
    result->setObjectName(name);
    result->setProperty("fluentRole", primary ? "primary" : "secondary");
    result->setMinimumHeight(36);
    result->setCursor(Qt::PointingHandCursor);
    result->setIcon(ui::icon(symbol, QColor(primary ? "#00374D" : "#D2D2D2")));
    return result;
}
QFrame *panel()
{
    auto *result = new QFrame;
    result->setObjectName("card");
    return result;
}
QString helperPath()
{
    const QString override = qEnvironmentVariable("VISION_STUDIO_CONVERSION_WORKER");
    if (!override.isEmpty())
        return QFileInfo(override).absoluteFilePath();
    const QString home = qEnvironmentVariable("VISION_STUDIO_HOME");
    const QString app = QCoreApplication::applicationDirPath();
    const QStringList candidates = {home + "/scripts/model_convert.py", app + "/../../scripts/model_convert.py",
                                    app + "/../scripts/model_convert.py",
                                    QString::fromUtf8(VISION_PROJECT_DIR) + "/scripts/model_convert.py"};
    for (const QString &path : candidates)
        if (QFileInfo(path).isFile())
            return QFileInfo(path).absoluteFilePath();
    return {};
}
QString pythonPath(const QString &script)
{
    const QString override = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    if (!override.isEmpty())
        return QFileInfo(override).isAbsolute() ? override : QStandardPaths::findExecutable(override);
    const QStringList candidates = {
        qEnvironmentVariable("VISION_STUDIO_HOME") + "/runtime/bin/python",
        QFileInfo(script).absolutePath() + "/../runtime/bin/python",
        QString::fromUtf8(VISION_PROJECT_DIR) + "/runtime/bin/python",
        QDir::homePath() + "/VisionStudio/runtime/bin/python"};
    for (const QString &path : candidates)
        if (QFileInfo(path).isFile() && QFileInfo(path).isExecutable())
            return QFileInfo(path).absoluteFilePath();
    return {};
}
bool supportedSource(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == "pt" || suffix == "torchscript";
}
}

ModelConversionPage::ModelConversionPage(const QString &dataRoot, QWidget *parent)
    : QWidget(parent), dataRoot_(dataRoot)
{
    setObjectName("modelConversionPage");
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->viewport()->setObjectName("conversionViewport");
    auto *content = new QWidget;
    content->setObjectName("conversionContent");
    auto *layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 0, 10, 0);
    layout->setSpacing(14);

    auto *source = panel();
    auto *sourceLayout = new QVBoxLayout(source);
    sourceLayout->setContentsMargins(22, 18, 22, 18);
    auto *sourceHeading = new QHBoxLayout;
    sourceHeading->addWidget(label("01   来源模型", "sectionTitle"), 1);
    chooseSource_ = action("在模型库中选择", "model", "conversionChooseModelButton");
    sourceHeading->addWidget(chooseSource_);
    sourceLayout->addLayout(sourceHeading);
    sourceName_ = label("尚未选择模型", "modelName");
    sourceName_->setObjectName("conversionSourceName");
    sourcePath_ = label("模型转换使用模型库中全局选用的模型。", "muted");
    sourcePath_->setObjectName("conversionSourcePath");
    sourcePath_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    sourceLayout->addWidget(sourceName_);
    sourceLayout->addWidget(sourcePath_);
    supportHint_ = label("支持完整的 YOLO PT 检查点与 TorchScript。", "muted");
    supportHint_->setObjectName("conversionSupportHint");
    sourceLayout->addWidget(supportHint_);
    layout->addWidget(source);

    auto *options = panel();
    auto *optionsLayout = new QVBoxLayout(options);
    optionsLayout->setContentsMargins(22, 18, 22, 18);
    optionsLayout->setSpacing(14);
    optionsLayout->addWidget(label("02   导出设置", "sectionTitle"));
    auto *form = new QFormLayout;
    form->setHorizontalSpacing(18);
    form->setVerticalSpacing(12);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    format_ = new QComboBox;
    format_->setObjectName("conversionFormat");
    format_->addItem("ONNX · 通用推理模型", "onnx");
    format_->addItem("TorchScript · PyTorch 推理模型", "torchscript");
    form->addRow("目标格式", format_);
    imageSize_ = new QSpinBox;
    imageSize_->setObjectName("conversionImageSize");
    imageSize_->setRange(32, 4096);
    imageSize_->setSingleStep(32);
    imageSize_->setSuffix(" px");
    imageSize_->setValue(640);
    form->addRow("输入尺寸", imageSize_);
    opset_ = new QComboBox;
    opset_->setObjectName("conversionOpset");
    opset_->addItem("12 · 优先兼容 OpenCV", 12);
    opset_->addItem("17 · 较新的 ONNX 算子集", 17);
    form->addRow("ONNX Opset", opset_);
    auto *directoryRow = new QHBoxLayout;
    directory_ = new QLineEdit(dataRoot_ + "/converted-models");
    directory_->setObjectName("conversionOutputDirectory");
    chooseDirectory_ = action("浏览…", "folder", "conversionChooseDirectoryButton");
    directoryRow->addWidget(directory_, 1);
    directoryRow->addWidget(chooseDirectory_);
    form->addRow("保存文件夹", directoryRow);
    filename_ = new QLineEdit;
    filename_->setObjectName("conversionOutputFilename");
    form->addRow("文件名称", filename_);
    optionsLayout->addLayout(form);
    auto *note = label("FP32 · batch=1 · 固定正方形输入 · 不含 NMS。输入通道按模型读取，支持 1 / 3 通道。\n"
                       "保留原模型，不覆盖已有文件。ONNX 无法还原为原始可训练的 PT 检查点。", "muted");
    optionsLayout->addWidget(note);
    layout->addWidget(options);

    auto *execution = panel();
    auto *executionLayout = new QVBoxLayout(execution);
    executionLayout->setContentsMargins(22, 18, 22, 18);
    executionLayout->setSpacing(12);
    auto *buttons = new QHBoxLayout;
    start_ = action("开始转换", "convert", "conversionStartButton", true);
    cancel_ = action("取消转换", "cross", "conversionCancelButton");
    buttons->addWidget(start_);
    buttons->addWidget(cancel_);
    buttons->addStretch();
    openOutput_ = action("打开输出文件夹", "folder", "conversionOpenOutputButton");
    addModel_ = action("加入模型库", "plus", "conversionAddModelButton");
    buttons->addWidget(openOutput_);
    buttons->addWidget(addModel_);
    executionLayout->addLayout(buttons);
    status_ = label("准备就绪", "sectionTitle");
    status_->setObjectName("conversionStatus");
    executionLayout->addWidget(status_);
    progress_ = new QProgressBar;
    progress_->setObjectName("conversionProgress");
    progress_->setRange(0, 100);
    progress_->setValue(0);
    progress_->setTextVisible(false);
    progress_->setFixedHeight(5);
    executionLayout->addWidget(progress_);
    summary_ = label("转换在后台执行，可在完成后加入模型库。", "muted");
    summary_->setObjectName("conversionSummary");
    summary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    executionLayout->addWidget(summary_);
    log_ = new QPlainTextEdit;
    log_->setObjectName("conversionLog");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(400);
    log_->setMinimumHeight(120);
    executionLayout->addWidget(log_);
    layout->addWidget(execution);
    layout->addStretch();
    scroll->setWidget(content);
    outer->addWidget(scroll);
    connect(chooseSource_, &QPushButton::clicked, this, &ModelConversionPage::chooseLibraryRequested);
    connect(chooseDirectory_, &QPushButton::clicked, this, [this] {
        const QString selected = QFileDialog::getExistingDirectory(this, "选择模型输出文件夹", directory_->text());
        if (!selected.isEmpty()) directory_->setText(selected);
    });
    connect(format_, &QComboBox::currentIndexChanged, this, [this] { updateDestination(true); updateControls(); });
    connect(imageSize_, &QSpinBox::valueChanged, this, [this] { updateDestination(); });
    connect(start_, &QPushButton::clicked, this, &ModelConversionPage::start);
    connect(cancel_, &QPushButton::clicked, this, &ModelConversionPage::cancel);
    connect(openOutput_, &QPushButton::clicked, this, [this] {
        openFolder(completedOutput_.isEmpty() ? directory_->text() : QFileInfo(completedOutput_).absolutePath());
    });
    connect(addModel_, &QPushButton::clicked, this, [this] {
        if (QFileInfo(completedOutput_).isFile()) emit convertedModelReady(completedOutput_);
    });
    updateSource();
    updateDestination(true);
    updateControls();
}

ModelConversionPage::~ModelConversionPage()
{
    if (process_) {
        process_->disconnect(this);
        process_->kill();
        process_->waitForFinished(500);
    }
    cleanupJob();
}

void ModelConversionPage::setSourceModel(const QString &path, const QString &displayName)
{
    source_ = path;
    displayName_ = displayName;
    if (!isBusy()) { updateSource(); updateDestination(true); updateControls(); }
}

bool ModelConversionPage::isBusy() const { return process_ != nullptr; }

void ModelConversionPage::updateSource()
{
    sourceName_->setText(source_.isEmpty() ? "尚未选择模型" :
                        (displayName_.isEmpty() ? QFileInfo(source_).fileName() : displayName_));
    sourcePath_->setText(source_.isEmpty() ? "在模型库中选择要转换的模型。" : source_);
    supportHint_->setText(source_.isEmpty() ? "转换使用模型库中全局选用的模型。" :
        (!QFileInfo(source_).isFile() ? "模型文件已移动或不存在，请重新选择。" :
         (!supportedSource(source_) ? "当前选择为 ONNX 或其他格式。首版提供 PT → ONNX / TorchScript，请在模型库选择 PT 模型。" :
          "支持 YOLOv5、YOLOv8 / YOLO11 完整检查点；裸权重需要原始网络定义，无法直接导出。")));
}

void ModelConversionPage::updateDestination(bool force)
{
    if (isBusy()) return;
    if (!force && filename_->isModified()) return;
    QString name = QFileInfo(source_).completeBaseName();
    if (name.isEmpty()) name = "model";
    name.replace(QRegularExpression("[^\\p{L}\\p{N}_.-]"), "_");
    const QString suffix = format_->currentData().toString() == "onnx" ? "onnx" : "torchscript";
    filename_->setText(QString("%1-%2.%3").arg(name.left(80)).arg(imageSize_->value()).arg(suffix));
    filename_->setModified(false);
}

void ModelConversionPage::updateControls()
{
    const bool active = isBusy();
    for (QWidget *widget : QList<QWidget *>{format_, imageSize_, directory_, filename_, chooseDirectory_, chooseSource_})
        widget->setEnabled(!active);
    opset_->setEnabled(!active && format_->currentData().toString() == "onnx");
    start_->setEnabled(!active && QFileInfo(source_).isFile() && supportedSource(source_));
    cancel_->setEnabled(active && !cancelling_);
    openOutput_->setEnabled(!active);
    addModel_->setEnabled(!active && QFileInfo(completedOutput_).isFile());
}

void ModelConversionPage::appendLog(const QString &message) { log_->appendPlainText(message.left(4000)); }

void ModelConversionPage::start()
{
    if (isBusy()) return;
    const QString name = filename_->text().trimmed();
    const QString selectedFormat = format_->currentData().toString();
    const QString extension = selectedFormat == "onnx" ? ".onnx" : ".torchscript";
    const auto fail = [this](const QString &message) { status_->setText("无法开始转换"); summary_->setText(message); };
    if (name.isEmpty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') ||
        name.contains(QChar::Null) || !name.endsWith(extension, Qt::CaseInsensitive)) {
        fail("文件名称应为单个文件名，并以 " + extension + " 结尾。"); return;
    }
    if (imageSize_->value() % 32 != 0) { fail("输入尺寸必须是 32 的倍数。"); return; }
    const QString directory = directory_->text().trimmed();
    if (directory.isEmpty() || !QDir().mkpath(directory)) { fail("无法创建输出文件夹，请检查路径和写入权限。"); return; }
    output_ = QDir(QFileInfo(directory).canonicalFilePath()).absoluteFilePath(name);
    const QFileInfo destination(output_);
    if (destination.exists() || destination.isSymLink()) { fail("输出文件已存在，请修改文件名。原文件不会被覆盖。"); return; }
    if (!QFileInfo(source_).isFile() || !supportedSource(source_)) { fail("请在模型库选择有效的 PT 或 TorchScript 模型。"); return; }
    const QString script = helperPath(), python = pythonPath(script);
    if (!QFileInfo(script).isFile() || !QFileInfo(python).isExecutable()) {
        fail("未找到模型转换运行环境。请使用完整安装包，或配置 VISION_STUDIO_PYTHON。"); return;
    }
    runningSource_ = source_;
    completedOutput_.clear();
    completedMetadata_ = {};
    pendingOutput_.clear(); stderrOutput_.clear(); jobTemp_.clear();
    receivedResult_ = cancelling_ = protocolError_ = false;
    QTemporaryDir work(QFileInfo(output_).absolutePath() + "/.vision-studio-convert-XXXXXX");
    struct stat workInfo{};
    if (!work.isValid() || ::lstat(QFile::encodeName(work.path()).constData(), &workInfo) != 0) {
        fail("无法创建本次转换的临时目录。"); return;
    }
    work.setAutoRemove(false);
    jobTemp_ = work.path(); jobTempInode_ = workInfo.st_ino; jobTempDevice_ = workInfo.st_dev;
    log_->clear();
    status_->setText("正在转换…");
    summary_->setText("转换模型：" + QFileInfo(runningSource_).fileName() + "\n输出：" + output_);
    progress_->setRange(0, 0);
    process_ = new QProcess(this);
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    environment.insert("PYTHONNOUSERSITE", "1");
    environment.insert("YOLO_AUTOINSTALL", "false");
    environment.insert("YOLOv5_AUTOINSTALL", "false");
    environment.insert("OMP_NUM_THREADS", "2");
    environment.insert("MKL_NUM_THREADS", "2");
    const QString settings = dataRoot_ + "/conversion-runtime";
    QDir().mkpath(settings);
    environment.insert("YOLO_CONFIG_DIR", settings);
    environment.insert("YOLOV5_CONFIG_DIR", settings);
    environment.insert("MPLCONFIGDIR", settings);
    process_->setProcessEnvironment(environment);
    connect(process_, &QProcess::readyReadStandardOutput, this, &ModelConversionPage::readOutput);
    connect(process_, &QProcess::readyReadStandardError, this, [this] {
        const QByteArray bytes = process_->readAllStandardError();
        stderrOutput_.append(bytes);
        if (stderrOutput_.size() > 16384) stderrOutput_ = stderrOutput_.right(16384);
        const QString text = QString::fromUtf8(bytes).trimmed();
        if (!text.isEmpty()) appendLog(text);
    });
    connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) {
            summary_->setText("转换进程无法启动：" + process_->errorString());
            complete(-1, false);
        }
    });
    connect(process_, &QProcess::finished, this, [this](int code, QProcess::ExitStatus status) {
        complete(code, status == QProcess::NormalExit);
    });
    const QStringList arguments = {script, "--source", runningSource_, "--format", selectedFormat,
        "--output", output_, "--image-size", QString::number(imageSize_->value()),
        "--opset", QString::number(opset_->currentData().toInt()), "--work-dir", jobTemp_};
    process_->start(python, arguments);
    updateControls();
    emit busyChanged(true);
}

void ModelConversionPage::readOutput()
{
    if (!process_) return;
    pendingOutput_.append(process_->readAllStandardOutput());
    if (pendingOutput_.size() > protocolLimit) {
        protocolError_ = true; summary_->setText("转换进程返回的消息过大。"); cancel(); return;
    }
    int end = -1;
    while ((end = pendingOutput_.indexOf('\n')) >= 0) {
        const QByteArray line = pendingOutput_.left(end).trimmed();
        pendingOutput_.remove(0, end + 1);
        if (line.isEmpty()) continue;
        QJsonParseError error;
        const auto document = QJsonDocument::fromJson(line, &error);
        if (error.error != QJsonParseError::NoError || !document.isObject()) {
            protocolError_ = true; summary_->setText("转换进程返回无效消息。"); cancel(); return;
        }
        const auto event = document.object();
        const QString type = event.value("type").toString(), message = event.value("message").toString();
        if (event.value("protocol").toInt() != 1 ||
            (type != "status" && type != "progress" && type != "error" && type != "result")) {
            protocolError_ = true; summary_->setText("转换进程的协议版本或事件类型不兼容。"); cancel(); return;
        }
        if (!message.isEmpty()) appendLog(message);
        if (type == "status" && event.contains("job_temp")) {
            const QString path = event.value("job_temp").toString();
            if (QFileInfo(path).absoluteFilePath() != QFileInfo(jobTemp_).absoluteFilePath()) {
                protocolError_ = true; summary_->setText("转换进程的临时目录与本次任务不一致。"); cancel(); return;
            }
        }
        if (type == "progress") {
            progress_->setRange(0, 100);
            progress_->setValue(qBound(0, event.value("percent").toInt(), 100));
            status_->setText(message.isEmpty() ? "正在转换…" : message);
        } else if (type == "error") {
            if (event.value("code").toString() != "cancelled" || !cancelling_) {
                summary_->setText(message); protocolError_ = true;
            }
        } else if (type == "result") {
            const QString path = event.value("output").toString();
            if (QFileInfo(path).absoluteFilePath() != QFileInfo(output_).absoluteFilePath()) {
                protocolError_ = true; summary_->setText("转换进程输出路径与请求不一致。"); cancel(); return;
            }
            receivedResult_ = true;
            completedMetadata_ = event;
            summary_->setText(QString("%1 · FP32 · %2 通道 · %3 × %3\n%4")
                .arg(event.value("format").toString().toUpper())
                .arg(event.value("input_channels").toInt()).arg(imageSize_->value()).arg(path));
            updateControls();
        }
    }
}

void ModelConversionPage::cancel()
{
    if (!process_ || cancelling_) return;
    cancelling_ = true;
    status_->setText("正在取消…");
    process_->terminate();
    QPointer<QProcess> process(process_);
    QTimer::singleShot(1000, this, [process] {
        if (process && process->state() != QProcess::NotRunning) process->kill();
    });
    updateControls();
}

void ModelConversionPage::cleanupJob()
{
    if (jobTemp_.isEmpty()) return;
    const QString path = jobTemp_;
    const quint64 inode = jobTempInode_, device = jobTempDevice_;
    jobTemp_.clear();
    QThreadPool::globalInstance()->start(QRunnable::create([path, inode, device] {
        struct stat info{};
        if (::lstat(QFile::encodeName(path).constData(), &info) == 0 && S_ISDIR(info.st_mode) &&
            info.st_uid == ::geteuid() && quint64(info.st_ino) == inode && quint64(info.st_dev) == device)
            QDir(path).removeRecursively();
    }));
}

void ModelConversionPage::complete(int exitCode, bool normalExit)
{
    if (!process_) return;
    readOutput();
    const QByteArray finalErrors = process_->readAllStandardError();
    stderrOutput_.append(finalErrors);
    stderrOutput_ = stderrOutput_.right(16384);
    if (!finalErrors.trimmed().isEmpty()) appendLog(QString::fromUtf8(finalErrors));
    if (!pendingOutput_.trimmed().isEmpty()) {
        protocolError_ = true; summary_->setText("转换进程返回了未完成的消息。");
    }
    const bool success = normalExit && exitCode == 0 && receivedResult_ && !protocolError_ && QFileInfo(output_).isFile();
    const bool cancelled = cancelling_;
    process_->disconnect(this);
    process_->deleteLater(); process_ = nullptr;
    cleanupJob();
    progress_->setRange(0, 100); progress_->setValue(success ? 100 : 0);
    if (success) { completedOutput_ = output_; status_->setText("转换完成"); }
    else if (cancelled && !protocolError_) {
        status_->setText("已取消转换");
        if (receivedResult_ && QFileInfo(output_).isFile()) completedOutput_ = output_;
        summary_->setText(QFileInfo(output_).isFile()
            ? "任务已取消；输出文件已在取消前写出，保留于：" + output_
            : "转换已停止，原始模型保留。可修改设置后重新开始。");
    } else {
        status_->setText("转换失败");
        if (!protocolError_ && summary_->text().startsWith("转换模型："))
            summary_->setText("转换进程未成功完成，请查看下方日志。\n" + QString::fromUtf8(stderrOutput_).right(1500));
    }
    cancelling_ = false;
    updateSource();
    if (source_ != runningSource_) updateDestination(true);
    updateControls();
    emit busyChanged(false);
}

bool ModelConversionPage::openFolder(const QString &path)
{
    if (path.isEmpty() || !QDir().mkpath(path) || !QDesktopServices::openUrl(QUrl::fromLocalFile(QDir(path).absolutePath()))) {
        status_->setText("无法打开输出文件夹"); summary_->setText("请检查文件夹路径或系统文件管理器。"); return false;
    }
    return true;
}
