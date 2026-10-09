#include "core/visionengine.h"
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>

int main(int argc, char **argv)
{
    QCoreApplication application(argc, argv);
    try
    {
        if (argc != 5) throw std::runtime_error("model metadata-smoke-json bus-image cpu|cuda required");
        QFile metadataFile(QString::fromLocal8Bit(argv[2]));
        if (!metadataFile.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot read conversion metadata");
        const QJsonObject metadata = QJsonDocument::fromJson(metadataFile.readAll()).object().value("conversion").toObject();
        const auto shape = metadata.value("shape").toArray();
        const auto names = metadata.value("names").toObject();
        if (metadata.value("task").toString() != "detect" || (metadata.value("layout").toString() != "v8" && metadata.value("layout").toString() != "v5") ||
            shape.size() != 4 || shape[0].toInt() != 1 || shape[1].toInt() != 3 || shape[2].toInt() != 640 || shape[3].toInt() != 640 || names.isEmpty())
            throw std::runtime_error("unexpected metadata; expected full v5/v8 detect C3 input640");
        vision::ModelConfig config;
        config.modelPath = QString::fromLocal8Bit(argv[1]);
        config.task = metadata.value("layout").toString() == "v5" ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8;
        config.inputSize = 640;
        config.inputChannels = 3;
        config.device = QString::fromLocal8Bit(argv[4]) == "cuda" ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU;
        config.deviceIndex = 0;
        for (int index = 0; index < names.size(); ++index)
        {
            const QString label = names.value(QString::number(index)).toString();
            if (label.isEmpty()) throw std::runtime_error("non-contiguous converted names");
            config.labels.append(label);
        }
        const QImage image(QString::fromLocal8Bit(argv[3]));
        if (image.isNull()) throw std::runtime_error("cannot decode actual input image");
        vision::VisionEngine engine;
        engine.load(config);
        const auto result = engine.infer(image, QString::fromLocal8Bit(argv[3]));
        QJsonArray predictions;
        for (const auto &p : result.predictions)
            predictions.append(QJsonObject{{"class_id", p.classId}, {"label", p.label}, {"confidence", p.confidence},
                                           {"x", p.box.x()}, {"y", p.box.y()}, {"width", p.box.width()}, {"height", p.box.height()}});
        const bool resolved = result.device == config.device && result.requestedDevice == config.device;
        const QJsonObject report{{"success", resolved && !predictions.isEmpty()}, {"model", config.modelPath},
                                 {"requested_device", config.device == vision::ComputeDevice::CUDA ? "cuda" : "cpu"},
                                 {"actual_device", result.device == vision::ComputeDevice::CUDA ? "cuda" : "cpu"},
                                 {"device_index", result.deviceIndex}, {"device_name", result.deviceName},
                                 {"device_notice", result.deviceNotice}, {"backend", result.backend},
                                 {"inference_ms", result.inferenceMs}, {"predictions", predictions},
                                 {"input_channels", engine.config().inputChannels}, {"input_size", engine.config().inputSize},
                                 {"label_count", engine.config().labels.size()}, {"qt", qVersion()}};
        std::cout << QJsonDocument(report).toJson(QJsonDocument::Compact).constData() << '\n';
        return report.value("success").toBool() ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        const QJsonObject report{{"success", false}, {"error", QString::fromUtf8(error.what())}};
        std::cout << QJsonDocument(report).toJson(QJsonDocument::Compact).constData() << '\n';
        return 1;
    }
}
