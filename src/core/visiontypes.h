#pragma once
#include <QImage>
#include <QMetaType>
#include <QRectF>
#include <QRegularExpression>
#include <QSet>
#include <QStringList>
#include <QVector>
#include <algorithm>

namespace vision
{
enum class ModelTask
{
    YoloV5,
    YoloV8,
    Classification
};
enum class SourceKind
{
    Images,
    Video,
    Camera
};
enum class InputColorMode
{
    Color,
    Grayscale
};
enum class StereoView
{
    Full,
    Left,
    Right
};
enum class ComputeDevice
{
    Auto,
    CPU,
    CUDA
};
struct ModelConfig
{
    QString modelPath;
    QStringList labels;
    // Labels only map original IDs to names; they never define the model's class range.
    QVector<int> classFilter;
    int modelClassCount = 0;
    ModelTask task = ModelTask::YoloV8;
    int inputSize = 640;
    float confidence = 0.25f;
    float iou = 0.45f;
    bool swapRB = true;
    InputColorMode colorMode = InputColorMode::Color;
    // Resolved from the model when loaded, never from the image's pixel format.
    int inputChannels = 0;
    double scale = 1.0 / 255.0;
    double meanR = 0, meanG = 0, meanB = 0;
    ComputeDevice device = ComputeDevice::CPU;
    int deviceIndex = 0;
    // Filled by the loaded backend; the preference alone never implies CUDA is active.
    ComputeDevice resolvedDevice = ComputeDevice::CPU;
    QString deviceName;
    QString deviceNotice;
};
struct Prediction
{
    int classId = -1;
    QString label;
    float confidence = 0;
    QRectF box;
};
struct InferenceResult
{
    // Display/export/recording pixels follow the input color mode.
    QImage image;
    // Preserve the original selected source ROI for switching preview modes.
    QImage originalImage;
    QVector<Prediction> predictions;
    QString source;
    QString modelName;
    QString backend = "OpenCV DNN";
    ModelTask task = ModelTask::YoloV8;
    QVector<int> classFilter;
    int modelClassCount = 0;
    double inferenceMs = 0;
    double totalMs = 0;
    qint64 frameNumber = 0;
    StereoView stereoView = StereoView::Full;
    QSize sourceFrameSize;
    bool demonstration = false;
    ComputeDevice requestedDevice = ComputeDevice::CPU;
    ComputeDevice device = ComputeDevice::CPU;
    int deviceIndex = -1;
    QString deviceName;
    QString deviceNotice;
};
struct JobRequest
{
    ModelConfig config;
    SourceKind sourceKind = SourceKind::Images;
    QStringList files;
    int cameraIndex = 0;
    StereoView stereoView = StereoView::Full;
};
QStringList cocoLabels();
QImage inputPreviewImage(const QImage &image, InputColorMode mode);

inline bool parseClassFilter(const QString &text, QVector<int> *ids, QString *error = nullptr)
{
    if (ids)
        ids->clear();
    if (error)
        error->clear();
    QVector<int> parsed;
    QSet<int> seen;
    const auto tokens = text.split(QRegularExpression(QStringLiteral("[\\s,，]+")), Qt::SkipEmptyParts);
    for (const QString &token : tokens)
    {
        bool ok = false;
        const int value = token.toInt(&ok);
        if (!ok || value < 0 || !QRegularExpression(QStringLiteral("^[0-9]+$")).match(token).hasMatch())
        {
            if (error)
                *error = QStringLiteral("类别 ID 必须是从 0 开始的非负整数，用逗号或空格分隔。无效值：%1").arg(token);
            return false;
        }
        if (!seen.contains(value))
        {
            parsed.append(value);
            seen.insert(value);
        }
    }
    std::sort(parsed.begin(), parsed.end());
    if (ids)
        *ids = parsed;
    return true;
}

inline bool validateClassFilter(const QVector<int> &ids, int modelClassCount, ModelTask task,
                                QString *error = nullptr)
{
    if (error)
        error->clear();
    if (ids.isEmpty())
        return true;
    QString message;
    if (task == ModelTask::Classification)
        message = QStringLiteral("类别筛选仅用于目标检测；分类任务不支持此筛选。");
    else if (modelClassCount <= 0)
        message = QStringLiteral("尚未取得模型实际类别数量，暂时不能使用类别筛选。请等待模型信息加载完成。");
    else
        for (int id : ids)
            if (id < 0 || id >= modelClassCount)
            {
                message = QStringLiteral("类别 ID %1 超出当前模型范围：0～%2（共 %3 类）。")
                              .arg(id).arg(modelClassCount - 1).arg(modelClassCount);
                break;
            }
    if (error)
        *error = message;
    return message.isEmpty();
}
} // namespace vision
Q_DECLARE_METATYPE(vision::InferenceResult)
Q_DECLARE_METATYPE(vision::JobRequest)
Q_DECLARE_METATYPE(vision::ModelConfig)
Q_DECLARE_METATYPE(vision::ComputeDevice)
