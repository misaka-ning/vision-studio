#include "core/inferenceworker.h"
#include "core/visionengine.h"
#include <QFile>
#include <QTemporaryDir>
#include <QtTest>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>

// Build small, real ONNX networks without a Python/onnx dependency. The network
// runs Conv -> GlobalAveragePool -> Reshape; its bias supplies known predictions.
// This exercises the actual OpenCV forward and decoder, not a mocked decoder.
namespace fixture
{
using Bytes = QByteArray;
Bytes varint(quint64 value)
{
    Bytes out;
    do
    {
        quint8 byte = value & 0x7f;
        value >>= 7;
        if (value)
            byte |= 0x80;
        out.append(char(byte));
    } while (value);
    return out;
}
Bytes integer(int field, quint64 value)
{
    return varint(quint64(field) << 3) + varint(value);
}
Bytes bytes(int field, const Bytes &value)
{
    return varint((quint64(field) << 3) | 2) + varint(value.size()) + value;
}
Bytes string(int field, const char *value)
{
    return bytes(field, Bytes(value));
}
Bytes shape(const QVector<qint64> &dimensions)
{
    Bytes result;
    for (qint64 dimension : dimensions)
        result += bytes(1, dimension < 0 ? string(2, "dynamic") : integer(1, dimension));
    return result;
}
Bytes info(const char *name, const QVector<qint64> &dimensions, int dataType = 1)
{
    Bytes tensorType = integer(1, dataType) + bytes(2, shape(dimensions));
    return string(1, name) + bytes(2, bytes(1, tensorType));
}
template <class T> Bytes raw(const QVector<T> &values)
{
    Bytes result;
    for (const T &value : values)
    {
        // ONNX tensor raw_data uses little endian, independent of host order.
        quint8 encoded[sizeof(T)];
        std::memcpy(encoded, &value, sizeof(T));
#if Q_BYTE_ORDER == Q_BIG_ENDIAN
        std::reverse(encoded, encoded + sizeof(T));
#endif
        result.append(reinterpret_cast<const char *>(encoded), sizeof(T));
    }
    return result;
}
template <class T>
Bytes tensor(const char *name, const QVector<qint64> &dimensions, int dataType, const QVector<T> &values)
{
    Bytes result;
    for (qint64 dimension : dimensions)
        result += integer(1, dimension);
    result += integer(2, dataType) + string(8, name) + bytes(9, raw(values));
    return result;
}
Bytes intsAttribute(const char *name, const QVector<qint64> &values)
{
    Bytes result = string(1, name);
    for (qint64 value : values)
        result += integer(8, value);
    return result + integer(20, 7); // AttributeProto.INTS
}
Bytes node(const char *op, const QVector<const char *> &inputs, const char *output,
           const QVector<Bytes> &attributes = {})
{
    Bytes result;
    for (const char *input : inputs)
        result += string(1, input);
    result += string(2, output) + string(4, op);
    for (const Bytes &attribute : attributes)
        result += bytes(5, attribute);
    return result;
}
QString model(const QString &directory, const QString &name, const QVector<qint64> &outputShape,
              const QVector<float> &values, QVector<float> weights = {}, int inputChannels = 3,
              bool dynamicChannels = false, bool initializerInputs = false)
{
    const int channels = values.size();
    if (weights.isEmpty())
        weights.fill(0.0f, channels * inputChannels);
    Bytes graph;
    graph += bytes(
        1, node("Conv", {"images", "weights", "bias"}, "conv", {intsAttribute("kernel_shape", {1, 1})}));
    graph += bytes(1, node("GlobalAveragePool", {"conv"}, "average"));
    graph += bytes(1, node("Reshape", {"average", "shape"}, "output"));
    graph += string(2, "vision_studio_test_fixture");
    graph += bytes(5, tensor("weights", {channels, inputChannels, 1, 1}, 1, weights));
    graph += bytes(5, tensor("bias", {channels}, 1, values));
    graph += bytes(5, tensor("shape", {outputShape.size()}, 7, outputShape));
    graph += bytes(11, info("images", {1, dynamicChannels ? -1 : inputChannels, 64, 64}));
    if (initializerInputs)
    {
        // Some older exports also list initializers as graph inputs. These must
        // not be counted as extra runtime images or mistaken for the channel axis.
        graph += bytes(11, info("weights", {channels, inputChannels, 1, 1}));
        graph += bytes(11, info("bias", {channels}));
        graph += bytes(11, info("shape", {outputShape.size()}, 7));
    }
    graph += bytes(12, info("output", outputShape));
    Bytes model =
        integer(1, 7) + string(2, "Vision Studio tests") + bytes(7, graph) + bytes(8, integer(2, 11));
    const QString path = directory + "/" + name + ".onnx";
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(model) != model.size())
        throw std::runtime_error("Unable to save ONNX test fixture");
    return path;
}
QVector<float> v8Predictions()
{
    // [1, 6, 12]: four coordinates followed by two class scores.
    QVector<float> out(6 * 12, 0.0f);
    const QVector<QVector<float>> candidates = {{32, 32, 32, 24, .95f, .02f},
                                                {33, 32, 32, 24, .90f, .02f},
                                                {32, 32, 32, 24, .02f, .80f},
                                                {0, 16, 32, 32, .70f, .02f},
                                                {-40, -40, 16, 16, .99f, .02f}};
    for (int row = 0; row < candidates.size(); ++row)
        for (int channel = 0; channel < 6; ++channel)
            out[channel * 12 + row] = candidates[row][channel];
    return out;
}
QVector<float> v5Predictions()
{
    QVector<float> out(12 * 7, 0.0f);
    const QVector<QVector<float>> candidates = {{32, 32, 32, 24, .80f, .95f, .02f},
                                                {33, 32, 32, 24, .70f, .95f, .02f},
                                                {32, 32, 32, 24, .90f, .02f, .80f},
                                                {0, 16, 32, 32, .90f, .70f, .02f},
                                                {-40, -40, 16, 16, .99f, .99f, .02f}};
    for (int row = 0; row < candidates.size(); ++row)
        for (int col = 0; col < 7; ++col)
            out[row * 7 + col] = candidates[row][col];
    return out;
}
} // namespace fixture

class CoreTests : public QObject
{
    Q_OBJECT
  private:
    QTemporaryDir directory;
    vision::ModelConfig config(const QString &path, vision::ModelTask task) const
    {
        vision::ModelConfig cfg;
        cfg.modelPath = path;
        cfg.task = task;
        cfg.inputSize = 64;
        cfg.confidence = .5f;
        cfg.iou = .45f;
        cfg.labels = {"alpha", "beta"};
        if (task == vision::ModelTask::Classification)
            cfg.labels.append("gamma");
        return cfg;
    }
    QImage image(int width = 128, int height = 64) const
    {
        QImage value(width, height, QImage::Format_RGB888);
        value.fill(QColor(17, 84, 136));
        return value;
    }
    float classConfidence(const vision::InferenceResult &result, int classId) const
    {
        for (const auto &prediction : result.predictions)
            if (prediction.classId == classId)
                return prediction.confidence;
        return std::numeric_limits<float>::quiet_NaN();
    }
    QString stereoSensitiveModel(const QString &name) const
    {
        QVector<float> values(6 * 12, 0.0f);
        values[0] = values[12] = 32;
        values[24] = 32;
        values[36] = 24;
        QVector<float> weights(values.size() * 3, 0.0f);
        // Red means alpha; blue means beta. Scores depend on the actual selected
        // pixels, while the known box tests coordinate mapping into that view.
        weights[48 * 3] = 1;
        weights[60 * 3 + 2] = 1;
        return fixture::model(directory.path(), name, {1, 6, 12}, values, weights);
    }
    QImage stereoImage(int width = 128, int height = 64) const
    {
        QImage result(width, height, QImage::Format_RGB888);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                result.setPixelColor(x, y, x < width / 2 ? QColor(255, 0, 0) : QColor(0, 0, 255));
        return result;
    }
    void verifyDetection(const vision::InferenceResult &result, float topConfidence)
    {
        QCOMPARE(result.predictions.size(), 3);
        const auto &first = result.predictions[0];
        QCOMPARE(first.classId, 0);
        QCOMPARE(first.label, QString("alpha"));
        QVERIFY(std::abs(first.confidence - topConfidence) < .002f);
        QVERIFY(std::abs(first.box.x() - 32.0) < .1);
        QVERIFY(std::abs(first.box.y() - 8.0) < .1);
        QVERIFY(std::abs(first.box.width() - 64.0) < .1);
        QVERIFY(std::abs(first.box.height() - 48.0) < .1);
        int sameClass = 0, otherClass = 0, clipped = 0;
        for (const auto &prediction : result.predictions)
        {
            if (prediction.classId == 0)
                ++sameClass;
            if (prediction.classId == 1)
                ++otherClass;
            if (prediction.box.x() < .1)
            {
                ++clipped;
                QVERIFY(std::abs(prediction.box.y()) < .1);
                QVERIFY(std::abs(prediction.box.width() - 32.0) < .1);
                QVERIFY(std::abs(prediction.box.height() - 32.0) < .1);
            }
            QVERIFY(prediction.box.left() >= 0);
            QVERIFY(prediction.box.top() >= 0);
            QVERIFY(prediction.box.right() <= 128.1);
            QVERIFY(prediction.box.bottom() <= 64.1);
        }
        QCOMPARE(sameClass, 2);
        QCOMPARE(otherClass, 1);
        QCOMPARE(clipped, 1);
        QVERIFY(result.totalMs >= 0);
        QVERIFY(result.inferenceMs >= 0);
    }
  private slots:
    void initTestCase()
    {
        QVERIFY(directory.isValid());
        qRegisterMetaType<vision::InferenceResult>();
        qRegisterMetaType<vision::JobRequest>();
    }
    void yoloV8LetterboxClippingAndNms()
    {
        const QString path = fixture::model(directory.path(), "v8", {1, 6, 12}, fixture::v8Predictions());
        vision::VisionEngine engine;
        engine.load(config(path, vision::ModelTask::YoloV8));
        const auto result = engine.infer(image(), "rectangular-image");
        verifyDetection(result, .95f);
        QCOMPARE(result.source, QString("rectangular-image"));
        QCOMPARE(result.image.size(), QSize(128, 64));
        QCOMPARE(result.image, image());
        QCOMPARE(result.originalImage, image());
        QVERIFY(!result.demonstration);
    }
    void yoloV5UsesObjectnessAndClassScores()
    {
        const QString path = fixture::model(directory.path(), "v5", {1, 12, 7}, fixture::v5Predictions());
        vision::VisionEngine engine;
        engine.load(config(path, vision::ModelTask::YoloV5));
        verifyDetection(engine.infer(image(), "v5"), .76f);
    }
    void classificationLogitsBecomeProbabilities()
    {
        const QString path = fixture::model(directory.path(), "logits", {1, 3}, {-1.0f, 0.0f, 2.0f});
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.confidence = 0;
        vision::VisionEngine engine;
        engine.load(cfg);
        const auto result = engine.infer(image(), "classification");
        QCOMPARE(result.predictions.size(), 3);
        QCOMPARE(result.predictions[0].classId, 2);
        QCOMPARE(result.predictions[0].label, QString("gamma"));
        const double expected = std::exp(2.0) / (std::exp(-1.0) + 1.0 + std::exp(2.0));
        QVERIFY(std::abs(result.predictions[0].confidence - expected) < .002);
        double sum = 0;
        for (const auto &prediction : result.predictions)
            sum += prediction.confidence;
        QVERIFY(std::abs(sum - 1.0) < .002);
    }
    void classificationProbabilitiesRemainUnchanged()
    {
        const QString path = fixture::model(directory.path(), "probabilities", {1, 3}, {.2f, .3f, .5f});
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.confidence = 0;
        vision::VisionEngine engine;
        engine.load(cfg);
        const auto result = engine.infer(image(), "classification");
        QCOMPARE(result.predictions.size(), 3);
        QCOMPARE(result.predictions[0].classId, 2);
        QVERIFY(std::abs(result.predictions[0].confidence - .5f) < .002);
        QVERIFY(std::abs(result.predictions[1].confidence - .3f) < .002);
        QVERIFY(std::abs(result.predictions[2].confidence - .2f) < .002);
    }
    void preprocessingRespectsColorChannels()
    {
        const QString path =
            fixture::model(directory.path(), "color", {1, 3}, {0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.confidence = 0;
        cfg.scale = 1;
        QImage input(64, 64, QImage::Format_RGB888);
        input.fill(QColor(0, 0, 1));
        vision::VisionEngine engine;
        engine.load(cfg);
        auto result = engine.infer(input, "rgb");
        QCOMPARE(result.predictions[0].classId, 2);
        QVERIFY(std::abs(result.predictions[0].confidence - 1.0f) < .002);
        cfg.swapRB = false;
        engine.load(cfg);
        result = engine.infer(input, "bgr");
        QCOMPARE(result.predictions[0].classId, 0);
        QVERIFY(std::abs(result.predictions[0].confidence - 1.0f) < .002);
    }
    void preprocessingSubtractsMeanBeforeScaling()
    {
        const QString path =
            fixture::model(directory.path(), "normalization", {1, 3}, {0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.confidence = 0;
        cfg.scale = .1;
        cfg.meanR = cfg.meanG = cfg.meanB = 1;
        QImage input(64, 64, QImage::Format_RGB888);
        input.fill(QColor(11, 21, 31));
        vision::VisionEngine engine;
        engine.load(cfg);
        const auto result = engine.infer(input, "normalized");
        QCOMPARE(result.predictions[0].classId, 2);
        const double expected = std::exp(3.0) / (std::exp(1.0) + std::exp(2.0) + std::exp(3.0));
        QVERIFY(std::abs(result.predictions[0].confidence - expected) < .002);
    }
    void inputPreviewUsesModelLuminanceAndPreservesColor_data()
    {
        QTest::addColumn<int>("format");
        QTest::newRow("rgb888-padded-scanlines") << int(QImage::Format_RGB888);
        QTest::newRow("rgba8888") << int(QImage::Format_RGBA8888);
        QTest::newRow("rgb32") << int(QImage::Format_RGB32);
    }
    void inputPreviewUsesModelLuminanceAndPreservesColor()
    {
        QFETCH(int, format);
        QImage input(3, 2, QImage::Format(format));
        for (int y = 0; y < input.height(); ++y)
        {
            input.setPixelColor(0, y, QColor(255, 0, 0));
            input.setPixelColor(1, y, QColor(0, 255, 0));
            input.setPixelColor(2, y, QColor(0, 0, 255));
        }
        input.setDevicePixelRatio(2);
        QCOMPARE(vision::inputPreviewImage(input, vision::InputColorMode::Color), input);
        const auto gray = vision::inputPreviewImage(input, vision::InputColorMode::Grayscale);
        QCOMPARE(gray.size(), input.size());
        QCOMPARE(gray.devicePixelRatio(), input.devicePixelRatio());
        QCOMPARE(gray.format(), QImage::Format_Grayscale8);
        for (int y = 0; y < gray.height(); ++y)
        {
            QCOMPARE(gray.pixelColor(0, y), QColor(76, 76, 76));
            QCOMPARE(gray.pixelColor(1, y), QColor(150, 150, 150));
            QCOMPARE(gray.pixelColor(2, y), QColor(29, 29, 29));
        }
        // The preview owns its pixels after either the source or conversion
        // temporaries are gone; rendering must never reference an OpenCV buffer.
        input.fill(Qt::black);
        QCOMPARE(gray.pixelColor(0, 1), QColor(76, 76, 76));
        QVERIFY(vision::inputPreviewImage({}, vision::InputColorMode::Grayscale).isNull());
    }
    void grayscaleSingleChannelUsesLuminance_data()
    {
        QTest::addColumn<QColor>("color");
        QTest::addColumn<int>("luminance");
        // These three pixels distinguish the luminance coefficients and RGB/BGR
        // ordering. Expected values are the rounded 8-bit BT.601 intensities.
        QTest::newRow("red") << QColor(255, 0, 0) << 76;
        QTest::newRow("green") << QColor(0, 255, 0) << 150;
        QTest::newRow("blue") << QColor(0, 0, 255) << 29;
    }
    void grayscaleSingleChannelUsesLuminance()
    {
        QFETCH(QColor, color);
        QFETCH(int, luminance);
        const QString path = fixture::model(directory.path(), "gray-c1", {1, 3}, {0, 0, -2},
                                            {1, 0, 0}, 1);
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.colorMode = vision::InputColorMode::Grayscale;
        cfg.meanR = 19;
        cfg.meanG = 88;
        cfg.meanB = -13;
        QImage input(64, 64, QImage::Format_RGB888);
        input.fill(color);
        vision::VisionEngine engine;
        const double value = (luminance - 19.0) / 255.0;
        const double expected = std::exp(value) / (std::exp(value) + 1.0 + std::exp(-2.0));
        for (bool swap : {true, false})
        {
            cfg.swapRB = swap;
            engine.load(cfg);
            QCOMPARE(engine.config().inputChannels, 1);
            const auto result = engine.infer(input, "gray-c1");
            QCOMPARE(result.originalImage, input);
            QCOMPARE(result.image.size(), input.size());
            QCOMPARE(result.image.format(), QImage::Format_Grayscale8);
            QCOMPARE(result.image.pixelColor(0, 0), QColor(luminance, luminance, luminance));
            QCOMPARE(result.image.pixelColor(63, 63), QColor(luminance, luminance, luminance));
            QCOMPARE(result.predictions.size(), 3);
            QVERIFY(std::abs(classConfidence(result, 0) - expected) < .0002);
        }
    }
    void grayscaleThreeChannelsReplicateOneIntensity()
    {
        const QString path = fixture::model(directory.path(), "gray-c3", {1, 4}, {0, 0, 0, -2},
                                            {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0});
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.labels = {"channel-0", "channel-1", "channel-2", "baseline"};
        cfg.colorMode = vision::InputColorMode::Grayscale;
        cfg.scale = .02;
        cfg.meanR = 17;
        cfg.meanG = 93;
        cfg.meanB = -42;
        QImage input(64, 64, QImage::Format_RGB888);
        input.fill(QColor(255, 0, 0));
        vision::VisionEngine engine;
        const double value = (76.0 - 17.0) * .02;
        const double expected = std::exp(value) / (3.0 * std::exp(value) + std::exp(-2.0));
        for (bool swap : {true, false})
        {
            cfg.swapRB = swap;
            engine.load(cfg);
            QCOMPARE(engine.config().inputChannels, 3);
            const auto result = engine.infer(input, "gray-c3");
            QCOMPARE(result.originalImage, input);
            QCOMPARE(result.image.size(), input.size());
            QCOMPARE(result.image.pixelColor(0, 0), QColor(76, 76, 76));
            QCOMPARE(result.predictions.size(), 4);
            for (int channel = 0; channel < 3; ++channel)
                QVERIFY(std::abs(classConfidence(result, channel) - expected) < .0002);
        }
    }
    void grayscaleDetectionPaddingUses114_data()
    {
        QTest::addColumn<int>("inputChannels");
        QTest::newRow("single-channel") << 1;
        QTest::newRow("replicated-three-channel") << 3;
    }
    void grayscaleDetectionPaddingUses114()
    {
        QFETCH(int, inputChannels);
        QVector<float> values(6 * 12, 0.0f);
        values[0] = values[12] = 32;
        values[24] = 32;
        values[36] = 24;
        QVector<float> weights(values.size() * inputChannels, 0.0f);
        // Candidate zero's class score is the actual mean input intensity.
        weights[48 * inputChannels] = 1;
        const QString path = fixture::model(directory.path(), "gray-padding-" + QString::number(inputChannels),
                                            {1, 6, 12}, values, weights, inputChannels);
        auto cfg = config(path, vision::ModelTask::YoloV8);
        cfg.colorMode = vision::InputColorMode::Grayscale;
        cfg.confidence = .01f;
        QImage input(128, 64, QImage::Format_RGB888);
        input.fill(QColor(255, 0, 0));
        vision::VisionEngine engine;
        engine.load(cfg);
        const auto result = engine.infer(input, "gray-letterbox");
        QCOMPARE(result.originalImage, input);
        QCOMPARE(result.image.size(), input.size());
        QCOMPARE(result.image.pixelColor(0, 0), QColor(76, 76, 76));
        QCOMPARE(result.image.pixelColor(127, 63), QColor(76, 76, 76));
        QCOMPARE(result.predictions.size(), 1);
        // The 128x64 image occupies half of the 64x64 model input; the remaining
        // half is gray padding 114. This also detects padding applied after scale.
        const double expected = (76.0 + 114.0) / (2.0 * 255.0);
        QVERIFY(std::abs(result.predictions[0].confidence - expected) < .0002);
        QVERIFY(std::abs(result.predictions[0].box.width() - 64.0) < .1);
    }
    void colorInputRejectsSingleChannelModel()
    {
        const QString path = fixture::model(directory.path(), "color-c1-rejected", {1, 3}, {0, 0, -2},
                                            {1, 0, 0}, 1);
        auto cfg = config(path, vision::ModelTask::Classification);
        QCOMPARE(cfg.colorMode, vision::InputColorMode::Color);
        vision::VisionEngine engine;
        bool rejected = false;
        try
        {
            engine.load(cfg);
        }
        catch (const std::exception &error)
        {
            const QString message = QString::fromUtf8(error.what());
            QVERIFY2(message.contains(QStringLiteral("灰度")) && message.contains(QStringLiteral("1")),
                     qPrintable(message));
            rejected = true;
        }
        QVERIFY(rejected);
        QVERIFY(!engine.loaded());
    }
    void unsupportedONNXInputChannels_data()
    {
        QTest::addColumn<int>("inputChannels");
        QTest::addColumn<bool>("dynamicChannels");
        QTest::newRow("four-channels") << 4 << false;
        QTest::newRow("symbolic-channel-axis") << 3 << true;
    }
    void unsupportedONNXInputChannels()
    {
        QFETCH(int, inputChannels);
        QFETCH(bool, dynamicChannels);
        const QString path = fixture::model(directory.path(), "unsupported-input", {1, 3}, {0, 0, -2},
                                            {}, inputChannels, dynamicChannels);
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.colorMode = vision::InputColorMode::Grayscale;
        vision::VisionEngine engine;
        bool rejected = false;
        try
        {
            engine.load(cfg);
        }
        catch (const std::exception &error)
        {
            const QString message = QString::fromUtf8(error.what());
            QVERIFY2(message.contains(QStringLiteral("通道")), qPrintable(message));
            rejected = true;
        }
        QVERIFY(rejected);
        QVERIFY(!engine.loaded());
    }
    void initializerGraphInputsRemainCompatible()
    {
        const QString path = fixture::model(directory.path(), "initializer-inputs", {1, 3}, {0, 0, -2},
                                            {1, 0, 0}, 1, false, true);
        auto cfg = config(path, vision::ModelTask::Classification);
        cfg.colorMode = vision::InputColorMode::Grayscale;
        QImage input(64, 64, QImage::Format_RGB888);
        input.fill(QColor(0, 255, 0));
        vision::VisionEngine engine;
        engine.load(cfg);
        QCOMPARE(engine.config().inputChannels, 1);
        const auto result = engine.infer(input, "legacy-initializer-inputs");
        QCOMPARE(result.originalImage, input);
        QCOMPARE(result.image.pixelColor(0, 0), QColor(150, 150, 150));
        const double value = 150.0 / 255.0;
        const double expected = std::exp(value) / (std::exp(value) + 1.0 + std::exp(-2.0));
        QVERIFY(std::abs(classConfidence(result, 0) - expected) < .0002);
    }
    void missingModelReportsFailure()
    {
        vision::VisionEngine engine;
        QVERIFY_EXCEPTION_THROWN(
            engine.load(config(directory.path() + "/missing.onnx", vision::ModelTask::YoloV8)),
            std::exception);
    }
    void unsupportedDetectionOutputReportsFailure()
    {
        const QString path = fixture::model(directory.path(), "bad-shape", {1, 3}, {.2f, .3f, .5f});
        vision::VisionEngine engine;
        engine.load(config(path, vision::ModelTask::YoloV8));
        QVERIFY_EXCEPTION_THROWN(engine.infer(image(), "bad-shape"), std::exception);
    }
    void emptyImageReportsFailure()
    {
        const QString path = fixture::model(directory.path(), "valid", {1, 6, 12}, fixture::v8Predictions());
        vision::VisionEngine engine;
        engine.load(config(path, vision::ModelTask::YoloV8));
        QVERIFY_EXCEPTION_THROWN(engine.infer(QImage(), "empty"), std::exception);
    }
    void workerProcessesEveryImageInBatch()
    {
        const QString model = fixture::model(directory.path(), "batch-model", {1, 3}, {.2f, .3f, .5f});
        const QString first = directory.path() + "/batch-first.png";
        const QString second = directory.path() + "/batch-second.png";
        QVERIFY(image().save(first));
        QVERIFY(image(64, 128).save(second));
        vision::JobRequest request;
        request.config = config(model, vision::ModelTask::Classification);
        request.files = {first, second};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        QSignalSpy progress(&worker, &vision::InferenceWorker::progress);
        worker.prepare();
        worker.run(request);
        QCOMPARE(failures.count(), 0);
        QCOMPARE(results.count(), 2);
        QCOMPARE(finished.count(), 1);
        QCOMPARE(finished[0][0].toBool(), false);
        const auto firstResult = qvariant_cast<vision::InferenceResult>(results[0][0]);
        const auto secondResult = qvariant_cast<vision::InferenceResult>(results[1][0]);
        QCOMPARE(firstResult.source, first);
        QCOMPARE(secondResult.source, second);
        QCOMPARE(firstResult.frameNumber, qint64(1));
        QCOMPARE(secondResult.frameNumber, qint64(2));
        QCOMPARE(firstResult.image.size(), QSize(128, 64));
        QCOMPARE(secondResult.image.size(), QSize(64, 128));
        QCOMPARE(progress.last()[0].toInt(), 2);
        QCOMPARE(progress.last()[1].toInt(), 2);
    }
    void workerStopsBatchAndCanRunAgain()
    {
        const QString model = fixture::model(directory.path(), "stop-model", {1, 3}, {.2f, .3f, .5f});
        const QString input = directory.path() + "/stop-image.png";
        QVERIFY(image().save(input));
        vision::JobRequest request;
        request.config = config(model, vision::ModelTask::Classification);
        request.files = {input, input, input};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        const auto stopConnection = connect(&worker, &vision::InferenceWorker::resultReady, &worker,
                                            [&worker] { worker.requestStop(); });
        worker.prepare();
        worker.run(request);
        QCOMPARE(failures.count(), 0);
        QCOMPARE(results.count(), 1);
        QCOMPARE(finished.count(), 1);
        QCOMPARE(finished[0][0].toBool(), true);
        disconnect(stopConnection);
        results.clear();
        finished.clear();
        request.files = {input};
        worker.prepare();
        worker.run(request);
        QCOMPARE(results.count(), 1);
        QCOMPARE(finished.count(), 1);
        QCOMPARE(finished[0][0].toBool(), false);
    }
    void workerDeliversVideoLastFrameAtEndOfFile()
    {
        const QString model = fixture::model(directory.path(), "video-model", {1, 3}, {.2f, .3f, .5f});
        const QString path = directory.path() + "/three-frames.avi";
        cv::VideoWriter writer(path.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                               cv::Size(64, 64));
        QVERIFY2(writer.isOpened(), "The local OpenCV video backend must support MJPG for this test.");
        for (int i = 0; i < 3; ++i)
            writer.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(20 + i * 30, 80, 120)));
        writer.release();
        vision::JobRequest request;
        request.config = config(model, vision::ModelTask::Classification);
        request.sourceKind = vision::SourceKind::Video;
        request.files = {path};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        QSignalSpy progress(&worker, &vision::InferenceWorker::progress);
        worker.prepare();
        worker.run(request);
        QCOMPARE(failures.count(), 0);
        QVERIFY(results.count() >= 2);
        QCOMPARE(finished.count(), 1);
        QCOMPARE(finished[0][0].toBool(), false);
        const auto last = qvariant_cast<vision::InferenceResult>(results.last()[0]);
        QCOMPARE(last.source, path);
        QCOMPARE(last.frameNumber, qint64(3));
        QCOMPARE(last.image.size(), QSize(64, 64));
        QCOMPARE(last.predictions[0].classId, 2);
        QCOMPARE(progress.last()[0].toInt(), 3);
    }
    void workerReportsUnreadableInputAndFinishes()
    {
        const QString model = fixture::model(directory.path(), "source-error-model", {1, 3}, {.2f, .3f, .5f});
        vision::JobRequest request;
        request.config = config(model, vision::ModelTask::Classification);
        request.files = {directory.path() + "/missing-image.png"};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.prepare();
        worker.run(request);
        QCOMPARE(results.count(), 0);
        QCOMPARE(failures.count(), 1);
        QVERIFY(!failures[0][0].toString().isEmpty());
        QCOMPARE(finished.count(), 1);
        QCOMPARE(finished[0][0].toBool(), false);
    }
    void stereoOddWidthSelectionAndInference()
    {
        QImage input = stereoImage(129, 64);
        input.setPixelColor(128, 0, QColor(255, 255, 0));
        const auto full = vision::selectStereoView(input, vision::StereoView::Full);
        const auto left = vision::selectStereoView(input, vision::StereoView::Left);
        const auto right = vision::selectStereoView(input, vision::StereoView::Right);
        QCOMPARE(full, input);
        QCOMPARE(left.size(), QSize(64, 64));
        QCOMPARE(right.size(), QSize(65, 64));
        QCOMPARE(left.pixelColor(63, 0), QColor(255, 0, 0));
        QCOMPARE(right.pixelColor(0, 0), QColor(0, 0, 255));
        QCOMPARE(right.pixelColor(64, 0), QColor(255, 255, 0));
        auto cfg = config(stereoSensitiveModel("odd-stereo"), vision::ModelTask::YoloV8);
        cfg.confidence = .1f;
        vision::VisionEngine engine;
        engine.load(cfg);
        const auto leftResult = engine.infer(left, "left-eye");
        const auto rightResult = engine.infer(right, "right-eye");
        QCOMPARE(leftResult.predictions.size(), 1);
        QCOMPARE(rightResult.predictions.size(), 1);
        QCOMPARE(leftResult.predictions[0].classId, 0);
        QCOMPARE(rightResult.predictions[0].classId, 1);
        QVERIFY(leftResult.predictions[0].confidence > .95f);
        QVERIFY(rightResult.predictions[0].confidence > .95f);
        QVERIFY(std::abs(leftResult.predictions[0].box.x() - 16.0) < .1);
        QVERIFY(std::abs(leftResult.predictions[0].box.width() - 32.0) < .1);
        QVERIFY(std::abs(rightResult.predictions[0].box.x() - 16.25) < .1);
        QVERIFY(std::abs(rightResult.predictions[0].box.width() - 32.5) < .1);
        QVERIFY(rightResult.predictions[0].box.right() <= 65);
    }
    void stereoSelectionRejectsEmptyAndTooNarrowFrames()
    {
        for (auto view : {vision::StereoView::Full, vision::StereoView::Left, vision::StereoView::Right})
            QVERIFY_EXCEPTION_THROWN(vision::selectStereoView(QImage(), view), std::exception);
        QImage onePixel(1, 8, QImage::Format_RGB888);
        QVERIFY_EXCEPTION_THROWN(vision::selectStereoView(onePixel, vision::StereoView::Left), std::exception);
        QVERIFY_EXCEPTION_THROWN(vision::selectStereoView(onePixel, vision::StereoView::Right), std::exception);
        const QImage minimum = stereoImage(2, 8);
        QCOMPARE(vision::selectStereoView(minimum, vision::StereoView::Left).size(), QSize(1, 8));
        QCOMPARE(vision::selectStereoView(minimum, vision::StereoView::Right).size(), QSize(1, 8));
        QCOMPARE(vision::selectStereoView(minimum, vision::StereoView::Left).pixelColor(0, 0), QColor(255, 0, 0));
        QCOMPARE(vision::selectStereoView(minimum, vision::StereoView::Right).pixelColor(0, 0), QColor(0, 0, 255));
    }
    void workerSelectsSideBySideVideoView_data()
    {
        QTest::addColumn<int>("viewValue");
        QTest::addColumn<int>("expectedClass");
        QTest::addColumn<int>("expectedWidth");
        QTest::newRow("full-frame") << int(vision::StereoView::Full) << -1 << 128;
        QTest::newRow("left-eye") << int(vision::StereoView::Left) << 0 << 64;
        QTest::newRow("right-eye") << int(vision::StereoView::Right) << 1 << 64;
    }
    void workerSelectsSideBySideVideoView()
    {
        QFETCH(int, viewValue);
        QFETCH(int, expectedClass);
        QFETCH(int, expectedWidth);
        const QString path = directory.path() + "/stereo-" + QString::number(viewValue) + ".avi";
        cv::VideoWriter writer(path.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                               cv::Size(128, 64));
        QVERIFY2(writer.isOpened(), "The local OpenCV video backend must support MJPG for this test.");
        cv::Mat frame(64, 128, CV_8UC3, cv::Scalar(255, 0, 0));
        frame(cv::Rect(0, 0, 64, 64)).setTo(cv::Scalar(0, 0, 255));
        for (int i = 0; i < 3; ++i)
            writer.write(frame);
        writer.release();
        vision::JobRequest request;
        request.config = config(stereoSensitiveModel("video-stereo-" + QString::number(viewValue)),
                                vision::ModelTask::YoloV8);
        request.config.confidence = .1f;
        request.sourceKind = vision::SourceKind::Video;
        request.stereoView = vision::StereoView(viewValue);
        request.files = {path};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.prepare();
        worker.run(request);
        QCOMPARE(failures.count(), 0);
        QVERIFY(results.count() >= 2);
        QCOMPARE(finished.count(), 1);
        const auto result = qvariant_cast<vision::InferenceResult>(results.last()[0]);
        QCOMPARE(result.frameNumber, qint64(3));
        QCOMPARE(result.source, path);
        QCOMPARE(result.sourceFrameSize, QSize(128, 64));
        QCOMPARE(result.stereoView, vision::StereoView(viewValue));
        QCOMPARE(result.image.size(), QSize(expectedWidth, 64));
        QCOMPARE(result.predictions.size(), 1);
        const auto &prediction = result.predictions[0];
        if (expectedClass >= 0)
        {
            QCOMPARE(prediction.classId, expectedClass);
            QVERIFY(prediction.confidence > .95f);
            QVERIFY(std::abs(prediction.box.x() - 16.0) < .1);
            QVERIFY(std::abs(prediction.box.y() - 20.0) < .1);
            QVERIFY(std::abs(prediction.box.width() - 32.0) < .1);
            QVERIFY(std::abs(prediction.box.height() - 24.0) < .1);
        }
        else
        {
            QVERIFY(std::abs(prediction.box.x() - 32.0) < .1);
            QVERIFY(std::abs(prediction.box.y() - 8.0) < .1);
            QVERIFY(std::abs(prediction.box.width() - 64.0) < .1);
            QVERIFY(std::abs(prediction.box.height() - 48.0) < .1);
        }
        QVERIFY(prediction.box.right() <= result.image.width());
    }
    void workerKeepsImagesWholeWhenStereoViewSelected()
    {
        const QString path = directory.path() + "/whole-still.png";
        QVERIFY(stereoImage().save(path));
        vision::JobRequest request;
        request.config = config(stereoSensitiveModel("whole-still"), vision::ModelTask::YoloV8);
        request.config.confidence = .1f;
        request.sourceKind = vision::SourceKind::Images;
        request.stereoView = vision::StereoView::Right;
        request.files = {path};
        vision::InferenceWorker worker;
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy failures(&worker, &vision::InferenceWorker::failed);
        worker.prepare();
        worker.run(request);
        QCOMPARE(failures.count(), 0);
        QCOMPARE(results.count(), 1);
        const auto result = qvariant_cast<vision::InferenceResult>(results[0][0]);
        QCOMPARE(result.stereoView, vision::StereoView::Full);
        QCOMPARE(result.sourceFrameSize, QSize(128, 64));
        QCOMPARE(result.image.size(), QSize(128, 64));
        QCOMPARE(result.predictions.size(), 1);
        QVERIFY(std::abs(result.predictions[0].box.width() - 64.0) < .1);
    }
};

QTEST_GUILESS_MAIN(CoreTests)
#include "core_tests.moc"
