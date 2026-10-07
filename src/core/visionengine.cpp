#include "visionengine.h"
#include "ptbackend.h"

#include <QElapsedTimer>
#include <QByteArrayView>
#include <QFile>
#include <QFileInfo>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace vision
{
namespace
{

[[noreturn]] void fail(const QString &message)
{
    throw std::runtime_error(message.toUtf8().constData());
}

// Read only the ONNX input declaration; OpenCV remains the network importer.
// Schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// Views skip tensor weights without copying or adding a Python dependency.
struct ProtoField
{
    quint32 number = 0;
    int wire = 0;
    quint64 value = 0;
    QByteArrayView bytes;
};

class ProtoReader
{
  public:
    explicit ProtoReader(QByteArrayView bytes) : bytes_(bytes) {}
    bool next(ProtoField &field)
    {
        if (position_ == bytes_.size())
            return false;
        field = {};
        const quint64 tag = varint();
        if ((tag >> 3) == 0 || (tag >> 3) > 0x1fffffff)
            invalid();
        field.number = quint32(tag >> 3);
        field.wire = int(tag & 7);
        if (field.wire == 0)
            field.value = varint();
        else if (field.wire == 2)
        {
            const quint64 size = varint();
            if (size > quint64(bytes_.size() - position_))
                invalid();
            field.bytes = bytes_.sliced(position_, qsizetype(size));
            position_ += qsizetype(size);
        }
        else if (field.wire == 1 || field.wire == 5)
        {
            const int size = field.wire == 1 ? 8 : 4;
            if (bytes_.size() - position_ < size)
                invalid();
            position_ += size;
        }
        else
            invalid();
        return true;
    }

  private:
    [[noreturn]] static void invalid()
    {
        fail(QStringLiteral("ONNX 输入声明损坏或不完整，无法识别模型通道数。"));
    }
    quint64 varint()
    {
        quint64 value = 0;
        for (int index = 0; index < 10; ++index)
        {
            if (position_ == bytes_.size())
                invalid();
            const quint8 byte = quint8(bytes_[position_++]);
            if (index == 9 && (byte & 0xfe))
                invalid();
            value |= quint64(byte & 0x7f) << (7 * index);
            if (!(byte & 0x80))
                return value;
        }
        invalid();
    }
    QByteArrayView bytes_;
    qsizetype position_ = 0;
};

QByteArrayView protoMessage(QByteArrayView bytes, quint32 number)
{
    ProtoReader reader(bytes);
    ProtoField field;
    QByteArrayView found;
    while (reader.next(field))
        if (field.number == number && field.wire == 2)
            found = field.bytes;
    return found;
}

int onnxInputChannels(const QByteArray &model)
{
    const QByteArrayView graph = protoMessage(model, 7); // ModelProto.graph
    if (graph.isEmpty())
        fail(QStringLiteral("ONNX 模型没有有效的计算图。"));
    QVector<QByteArrayView> inputs;
    QSet<QByteArray> initializers;
    ProtoReader graphReader(graph);
    ProtoField field;
    while (graphReader.next(field))
    {
        if (field.wire != 2)
            continue;
        if (field.number == 11) // GraphProto.input
            inputs.append(field.bytes);
        else if (field.number == 5) // TensorProto.name
            initializers.insert(protoMessage(field.bytes, 8).toByteArray());
        else if (field.number == 15) // SparseTensorProto.values
            initializers.insert(protoMessage(protoMessage(field.bytes, 1), 8).toByteArray());
    }
    QByteArrayView imageInput;
    int inputCount = 0;
    for (QByteArrayView input : inputs)
    {
        const QByteArray name = protoMessage(input, 1).toByteArray();
        if (name.isEmpty())
            fail(QStringLiteral("ONNX 模型的输入名称无效。"));
        if (!initializers.contains(name))
        {
            imageInput = input;
            ++inputCount;
        }
    }
    if (inputCount != 1)
        fail(QStringLiteral("当前支持一个图像输入的 ONNX 模型，实际有 %1 个输入。").arg(inputCount));
    const QByteArrayView tensor = protoMessage(protoMessage(imageInput, 2), 1);
    const QByteArrayView shape = protoMessage(tensor, 2);
    QVector<quint64> dimensions;
    ProtoReader shapeReader(shape);
    while (shapeReader.next(field))
    {
        if (field.number != 1 || field.wire != 2)
            continue;
        quint64 dimension = 0;
        ProtoReader dimensionReader(field.bytes);
        ProtoField item;
        while (dimensionReader.next(item))
            if (item.number == 1 && item.wire == 0)
                dimension = item.value;
        dimensions.append(dimension);
    }
    if (dimensions.size() != 4 || dimensions[0] > 1)
        fail(QStringLiteral("ONNX 图像输入应为 NCHW [1, C, H, W]，当前输入形状不受支持。"));
    if (dimensions[1] != 1 && dimensions[1] != 3)
        fail(QStringLiteral("无法确定兼容的模型输入通道：当前支持固定 1 或 3 通道。"
                            "请导出通道数固定为 1 或 3 的 NCHW ONNX。"));
    return int(dimensions[1]);
}

QString shapeOf(const cv::Mat &tensor)
{
    QStringList sizes;
    for (int i = 0; i < tensor.dims; ++i)
        sizes << QString::number(tensor.size[i]);
    return "[" + sizes.join(", ") + "]";
}

QString classLabel(const ModelConfig &config, int id, int classCount)
{
    if (!config.labels.isEmpty())
        return config.labels.at(id);
    if (classCount == 80 && config.task != ModelTask::Classification)
        return cocoLabels().at(id);
    return QStringLiteral("类别 %1").arg(id);
}

bool validProbability(float value)
{
    return std::isfinite(value) && value >= 0.f && value <= 1.f;
}

struct Letterbox
{
    cv::Mat image;
    double scaleX = 1;
    double scaleY = 1;
    int left = 0;
    int top = 0;
};

Letterbox makeLetterbox(const cv::Mat &bgr, int side)
{
    Letterbox out;
    const double scale = std::min(double(side) / bgr.cols, double(side) / bgr.rows);
    const int width = std::max(1, int(std::round(bgr.cols * scale)));
    const int height = std::max(1, int(std::round(bgr.rows * scale)));
    out.left = (side - width) / 2;
    out.top = (side - height) / 2;
    out.scaleX = double(width) / bgr.cols;
    out.scaleY = double(height) / bgr.rows;
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, out.image, out.top, side - height - out.top, out.left,
                       side - width - out.left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

double intersectionOverUnion(const QRectF &a, const QRectF &b)
{
    const QRectF intersection = a.intersected(b);
    const double intersectArea = intersection.width() * intersection.height();
    const double unionArea = a.width() * a.height() + b.width() * b.height() - intersectArea;
    return unionArea > 0 ? intersectArea / unionArea : 0;
}

QVector<Prediction> decodeClassification(const cv::Mat &output, const ModelConfig &config)
{
    const bool vector2d = output.dims == 2 && output.size[0] == 1;
    const bool vector4d =
        output.dims == 4 && output.size[0] == 1 && output.size[2] == 1 && output.size[3] == 1;
    if (!vector2d && !vector4d)
        fail(QStringLiteral("分类模型输出应为 [1, C] 或 [1, C, 1, 1]，实际为 %1。").arg(shapeOf(output)));
    const int classes = output.size[1];
    if (classes <= 0 || classes > 100000)
        fail(QStringLiteral("分类模型类别数量无效。"));
    if (!config.labels.isEmpty() && config.labels.size() != classes)
        fail(QStringLiteral("标签文件有 %1 个类别，模型输出有 %2 个类别；请使用匹配的标签文件。")
                 .arg(config.labels.size())
                 .arg(classes));

    std::vector<float> probabilities(classes);
    const float *values = output.ptr<float>();
    double sum = 0;
    bool normalized = true;
    for (int i = 0; i < classes; ++i)
    {
        if (!std::isfinite(values[i]))
            fail(QStringLiteral("分类输出包含无效数值，请检查模型或预处理参数。"));
        normalized = normalized && validProbability(values[i]);
        sum += values[i];
    }
    normalized = normalized && std::abs(sum - 1.0) <= 0.001;
    if (normalized)
    {
        std::copy(values, values + classes, probabilities.begin());
    }
    else
    {
        const float maximum = *std::max_element(values, values + classes);
        sum = 0;
        for (int i = 0; i < classes; ++i)
        {
            probabilities[i] = std::exp(values[i] - maximum);
            sum += probabilities[i];
        }
        for (float &probability : probabilities)
            probability = float(probability / sum);
    }
    std::vector<int> order(classes);
    std::iota(order.begin(), order.end(), 0);
    const int count = std::min(5, classes);
    std::partial_sort(order.begin(), order.begin() + count, order.end(),
                      [&probabilities](int a, int b) { return probabilities[a] > probabilities[b]; });
    QVector<Prediction> predictions;
    for (int i = 0; i < count; ++i)
    {
        const int id = order[i];
        predictions.append({id, classLabel(config, id, classes), probabilities[id], {}});
    }
    return predictions;
}

QVector<Prediction> decodeDetection(const cv::Mat &output, const ModelConfig &config,
                                    const Letterbox &letterbox, const QSize &originalSize)
{
    if (output.dims != 3 || output.size[0] != 1)
        fail(QStringLiteral(
                 "检测模型需要单张原始检测输出；当前形状 %1 不受支持。请导出 batch=1、nms=False 的 ONNX。")
                 .arg(shapeOf(output)));
    const bool v5 = config.task == ModelTask::YoloV5;
    const int channels = output.size[v5 ? 2 : 1];
    const int proposals = output.size[v5 ? 1 : 2];
    const int classes = channels - (v5 ? 5 : 4);
    const int expectedClasses = config.labels.isEmpty() ? 80 : int(config.labels.size());
    if (classes != expectedClasses || proposals <= 0 || proposals > 2000000)
    {
        fail(QStringLiteral(
                 "模型输出 %1 与所选 %2 格式或 %3 个标签不匹配。"
                 "YOLOv5 需要 [1, N, 5+C]，YOLOv8/11 需要 [1, 4+C, N]；分割、姿态和含 NMS 的输出不受支持。")
                 .arg(shapeOf(output), v5 ? "YOLOv5" : "YOLOv8/11")
                 .arg(expectedClasses));
    }
    const float *data = output.ptr<float>();
    const auto value = [data, channels, proposals, v5](int n, int c)
    { return data[v5 ? n * channels + c : c * proposals + n]; };
    const QRectF imageBounds(0, 0, originalSize.width(), originalSize.height());
    QVector<Prediction> candidates;
    // Limit adversarial candidate counts before quadratic NMS, while preserving the strongest detections.
    constexpr int maxCandidates = 30000;
    for (int n = 0; n < proposals; ++n)
    {
        const float objectness = v5 ? value(n, 4) : 1.f;
        if (!validProbability(objectness))
            fail(QStringLiteral("检测输出的目标置信度不是概率，请确认选择了正确的模型格式。"));
        if (objectness < config.confidence)
            continue;
        float score = 0.f;
        int id = -1;
        for (int c = 0; c < classes; ++c)
        {
            const float probability = value(n, c + (v5 ? 5 : 4));
            if (!validProbability(probability))
                fail(QStringLiteral("检测输出的类别置信度不是概率，请确认模型导出与预处理参数。"));
            if (probability > score)
            {
                score = probability;
                id = c;
            }
        }
        score *= objectness;
        if (id < 0 || score < config.confidence)
            continue;
        const float x = value(n, 0), y = value(n, 1);
        const float width = value(n, 2), height = value(n, 3);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height))
            fail(QStringLiteral("检测框包含无效数值，请检查模型导出与预处理参数。"));
        if (width <= 0 || height <= 0)
            continue;
        QRectF box((x - width / 2.0 - letterbox.left) / letterbox.scaleX,
                   (y - height / 2.0 - letterbox.top) / letterbox.scaleY, width / letterbox.scaleX,
                   height / letterbox.scaleY);
        box = box.intersected(imageBounds);
        if (box.width() <= 0 || box.height() <= 0)
            continue;
        candidates.append({id, classLabel(config, id, classes), score, box});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Prediction &a, const Prediction &b) { return a.confidence > b.confidence; });
    if (candidates.size() > maxCandidates)
        candidates.resize(maxCandidates);
    QVector<Prediction> kept;
    kept.reserve(std::min(300, int(candidates.size())));
    for (const Prediction &candidate : candidates)
    {
        bool suppressed = false;
        for (const Prediction &selected : kept)
        {
            if (selected.classId == candidate.classId &&
                intersectionOverUnion(selected.box, candidate.box) > config.iou)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
            kept.append(candidate);
        if (kept.size() >= 300)
            break;
    }
    return kept;
}

} // namespace

VisionEngine::VisionEngine() = default;
VisionEngine::~VisionEngine() = default;

bool VisionEngine::loaded() const noexcept
{
    return m_pt ? m_pt->loaded() : !m_net.empty();
}

QString VisionEngine::backendName() const
{
    return m_pt ? m_pt->backendName() : QStringLiteral("OpenCV DNN / CPU");
}

void VisionEngine::setCancellationCheck(std::function<bool()> check)
{
    m_cancellationCheck = std::move(check);
    if (m_pt)
        m_pt->setCancellationCheck(m_cancellationCheck);
}

void VisionEngine::unload()
{
    m_pt.reset();
    m_net = {};
    m_config = {};
}

void VisionEngine::load(const ModelConfig &requestedConfig)
{
    ModelConfig config = requestedConfig;
    unload();
    if (config.inputSize < 16 || config.inputSize > 4096)
        fail(QStringLiteral("输入尺寸应在 16 到 4096 像素之间。"));
    if (!std::isfinite(config.confidence) || config.confidence < 0 || config.confidence > 1 ||
        !std::isfinite(config.iou) || config.iou < 0 || config.iou > 1)
        fail(QStringLiteral("置信度和 NMS 阈值必须在 0 到 1 之间。"));
    if (!std::isfinite(config.scale) || config.scale <= 0 || !std::isfinite(config.meanR) ||
        !std::isfinite(config.meanG) || !std::isfinite(config.meanB))
        fail(QStringLiteral("预处理缩放系数必须为正数，均值必须是有限数值。"));
    if (!QFileInfo(config.modelPath).isFile())
        fail(QStringLiteral("找不到模型：%1").arg(config.modelPath));
    const QString suffix = QFileInfo(config.modelPath).suffix().toLower();
    if (suffix == "pt")
    {
        auto backend = std::make_unique<PtBackend>();
        backend->setCancellationCheck(m_cancellationCheck);
        backend->load(config);
        m_config = backend->config();
        m_pt = std::move(backend);
        return;
    }
    if (suffix != "onnx")
        fail(QStringLiteral("请选择 .onnx 或 .pt 模型文件。"));
    QFile modelFile(config.modelPath);
    if (!modelFile.open(QIODevice::ReadOnly))
        fail(QStringLiteral("无法读取模型：%1").arg(modelFile.errorString()));
    const QByteArray modelBytes = modelFile.readAll();
    if (modelBytes.isEmpty())
        fail(QStringLiteral("模型文件为空。"));
    config.inputChannels = onnxInputChannels(modelBytes);
    if (config.colorMode == InputColorMode::Color && config.inputChannels != 3)
        fail(QStringLiteral("此模型需要 1 通道输入，请将输入通道模式切换为「灰度」。"));
    if (config.colorMode == InputColorMode::Grayscale)
        config.meanG = config.meanB = config.meanR;
    try
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelBytes.constData(), size_t(modelBytes.size()));
        if (net.empty())
            fail(QStringLiteral("ONNX 文件没有可运行的网络。"));
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        m_config = config;
        m_net = std::move(net);
    }
    catch (const cv::Exception &error)
    {
        fail(QStringLiteral("OpenCV 无法加载此 ONNX 模型。请使用静态 FP32、batch=1 的兼容导出。\n%1")
                 .arg(QString::fromUtf8(error.what())));
    }
}

InferenceResult VisionEngine::infer(const QImage &image, const QString &source)
{
    if (!loaded())
        fail(QStringLiteral("请先加载 ONNX 或 PyTorch 模型。"));
    if (image.isNull())
        fail(QStringLiteral("输入图像为空或无法解码。"));
    if (m_pt)
        return m_pt->infer(image, source);
    QElapsedTimer totalTimer;
    totalTimer.start();
    try
    {
        const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
        const cv::Mat rgbMat(rgb.height(), rgb.width(), CV_8UC3, const_cast<uchar *>(rgb.constBits()),
                             size_t(rgb.bytesPerLine()));
        cv::Mat bgr;
        cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);
        if (m_config.colorMode == InputColorMode::Grayscale)
        {
            cv::Mat gray;
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            if (m_config.inputChannels == 1)
                bgr = gray;
            else
                cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        }
        Letterbox letterbox;
        if (m_config.task == ModelTask::Classification)
        {
            cv::resize(bgr, letterbox.image, cv::Size(m_config.inputSize, m_config.inputSize));
        }
        else
        {
            letterbox = makeLetterbox(bgr, m_config.inputSize);
        }
        const bool grayInput = m_config.colorMode == InputColorMode::Grayscale;
        const cv::Scalar mean = grayInput
                                    ? cv::Scalar::all(m_config.meanR)
                                    : m_config.swapRB
                                          ? cv::Scalar(m_config.meanR, m_config.meanG, m_config.meanB)
                                          : cv::Scalar(m_config.meanB, m_config.meanG, m_config.meanR);
        const cv::Mat blob = cv::dnn::blobFromImage(letterbox.image, m_config.scale,
                                                    cv::Size(m_config.inputSize, m_config.inputSize), mean,
                                                    !grayInput && m_config.swapRB, false, CV_32F);
        m_net.setInput(blob);
        QElapsedTimer inferenceTimer;
        inferenceTimer.start();
        std::vector<cv::Mat> outputs;
        m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
        const double inferenceMs = inferenceTimer.nsecsElapsed() / 1e6;
        if (outputs.empty())
            fail(QStringLiteral("模型没有输出张量。"));
        // Older YOLOv5 exports may expose three auxiliary raw feature heads.
        // Accept these only alongside exactly one decoded detection tensor.
        const cv::Mat *primary = nullptr;
        for (const cv::Mat &tensor : outputs)
        {
            const bool auxiliaryV5 =
                m_config.task == ModelTask::YoloV5 && tensor.dims == 5 && tensor.size[0] == 1 &&
                tensor.size[1] == 3 &&
                tensor.size[4] == (m_config.labels.isEmpty() ? 85 : m_config.labels.size() + 5);
            if (auxiliaryV5)
                continue;
            if (primary)
                fail(QStringLiteral("模型含多个主要输出。当前支持原始目标检测和单标签分类；分割、姿态与端到端"
                                    " NMS 模型不受支持。"));
            primary = &tensor;
        }
        if (!primary)
            fail(QStringLiteral("模型没有解码后的检测输出，请导出标准推理 ONNX。"));
        cv::Mat output;
        if (primary->depth() != CV_32F)
            primary->convertTo(output, CV_32F);
        else
            output = primary->isContinuous() ? *primary : primary->clone();
        InferenceResult result;
        result.image = image;
        result.source = source;
        result.modelName = QFileInfo(m_config.modelPath).fileName();
        result.task = m_config.task;
        result.inferenceMs = inferenceMs;
        result.predictions = m_config.task == ModelTask::Classification
                                 ? decodeClassification(output, m_config)
                                 : decodeDetection(output, m_config, letterbox, image.size());
        result.totalMs = totalTimer.nsecsElapsed() / 1e6;
        return result;
    }
    catch (const cv::Exception &error)
    {
        fail(QStringLiteral("模型推理失败。请检查输入尺寸和 ONNX 算子是否兼容本机 OpenCV。\n%1")
                 .arg(QString::fromUtf8(error.what())));
    }
}

QStringList cocoLabels()
{
    static const QStringList labels = {"person",        "bicycle",      "car",
                                       "motorcycle",    "airplane",     "bus",
                                       "train",         "truck",        "boat",
                                       "traffic light", "fire hydrant", "stop sign",
                                       "parking meter", "bench",        "bird",
                                       "cat",           "dog",          "horse",
                                       "sheep",         "cow",          "elephant",
                                       "bear",          "zebra",        "giraffe",
                                       "backpack",      "umbrella",     "handbag",
                                       "tie",           "suitcase",     "frisbee",
                                       "skis",          "snowboard",    "sports ball",
                                       "kite",          "baseball bat", "baseball glove",
                                       "skateboard",    "surfboard",    "tennis racket",
                                       "bottle",        "wine glass",   "cup",
                                       "fork",          "knife",        "spoon",
                                       "bowl",          "banana",       "apple",
                                       "sandwich",      "orange",       "broccoli",
                                       "carrot",        "hot dog",      "pizza",
                                       "donut",         "cake",         "chair",
                                       "couch",         "potted plant", "bed",
                                       "dining table",  "toilet",       "tv",
                                       "laptop",        "mouse",        "remote",
                                       "keyboard",      "cell phone",   "microwave",
                                       "oven",          "toaster",      "sink",
                                       "refrigerator",  "book",         "clock",
                                       "vase",          "scissors",     "teddy bear",
                                       "hair drier",    "toothbrush"};
    return labels;
}

} // namespace vision
