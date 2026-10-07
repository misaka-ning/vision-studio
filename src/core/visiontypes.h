#pragma once
#include <QImage>
#include <QMetaType>
#include <QRectF>
#include <QStringList>
#include <QVector>

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
struct ModelConfig
{
    QString modelPath;
    QStringList labels;
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
    QImage image;
    QVector<Prediction> predictions;
    QString source;
    QString modelName;
    QString backend = "OpenCV DNN";
    ModelTask task = ModelTask::YoloV8;
    double inferenceMs = 0;
    double totalMs = 0;
    qint64 frameNumber = 0;
    StereoView stereoView = StereoView::Full;
    QSize sourceFrameSize;
    bool demonstration = false;
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
} // namespace vision
Q_DECLARE_METATYPE(vision::InferenceResult)
Q_DECLARE_METATYPE(vision::JobRequest)
Q_DECLARE_METATYPE(vision::ModelConfig)
