// SPDX-License-Identifier: AGPL-3.0-only
// Independent numerical comparison using the exact OpenCV linked by the app.
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <opencv2/dnn.hpp>
#include <cstdio>
#include <cstring>
#include <stdexcept>

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    try {
        const auto a = app.arguments();
        if (a.size() != 6) return 2;
        const int channels = a[4].toInt(), size = a[5].toInt();
        if ((channels != 1 && channels != 3) || size < 32 || size > 4096) return 2;
        QFile model(a[1]), input(a[2]), output(a[3]);
        if (!model.open(QIODevice::ReadOnly) || !input.open(QIODevice::ReadOnly)) return 3;
        const auto modelBytes = model.readAll(), image = input.readAll();
        int shape[] = {1, channels, size, size};
        cv::Mat blob(4, shape, CV_32F);
        if (image.size() != qint64(blob.total()*blob.elemSize())) return 4;
        std::memcpy(blob.data, image.constData(), size_t(image.size()));
        auto network = cv::dnn::readNetFromONNX(modelBytes.constData(), size_t(modelBytes.size()));
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); network.setInput(blob);
        const auto result = network.forward();
        if (!result.isContinuous() || result.depth() != CV_32F || !output.open(QIODevice::WriteOnly | QIODevice::NewOnly)) return 5;
        const auto bytes = qint64(result.total()*result.elemSize());
        if (output.write(reinterpret_cast<const char *>(result.data), bytes) != bytes) return 6;
        output.close();
        QJsonArray dimensions; for (int i=0;i<result.dims;++i) dimensions.append(result.size[i]);
        const auto json = QJsonDocument(QJsonObject{{"opencv_version",CV_VERSION},{"shape",dimensions}}).toJson(QJsonDocument::Compact);
        std::fwrite(json.constData(),1,size_t(json.size()),stdout); std::fflush(stdout); return 0;
    } catch (const std::exception &error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
