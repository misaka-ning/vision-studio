#include "inferenceworker.h"
#include "visionengine.h"

#include <QElapsedTimer>
#include <QFile>
#include <QImageReader>
#include <QThread>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <stdexcept>

namespace vision
{
namespace
{

QImage frameToImage(const cv::Mat &frame)
{
    cv::Mat rgb;
    if (frame.channels() == 4)
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, qsizetype(rgb.step), QImage::Format_RGB888).copy();
}

} // namespace

void InferenceWorker::run(vision::JobRequest request)
{
    try
    {
        if (stopping())
        {
            emit finished(true);
            return;
        }
        emit status(QStringLiteral("正在加载视觉模型…"));
        VisionEngine engine;
        engine.setCancellationCheck([this] { return stopping(); });
        engine.load(request.config);
        if (stopping())
        {
            emit finished(true);
            return;
        }
        emit modelReady(engine.backendName(), engine.config());
        emit status(QStringLiteral("模型已加载 · %1 / CPU").arg(engine.backendName()));
        if (request.sourceKind == SourceKind::Images)
        {
            if (request.files.isEmpty())
                throw std::runtime_error("请添加至少一张图像。");
            const int total = int(request.files.size());
            emit progress(0, total);
            for (int i = 0; i < total && !stopping(); ++i)
            {
                QImageReader reader(request.files.at(i));
                reader.setAutoTransform(true);
                const QImage image = reader.read();
                if (image.isNull())
                    throw std::runtime_error(QStringLiteral("无法读取图像 %1：%2")
                                                 .arg(request.files.at(i), reader.errorString())
                                                 .toUtf8()
                                                 .constData());
                auto result = engine.infer(image, request.files.at(i));
                result.frameNumber = i + 1;
                if (stopping())
                    break;
                emit resultReady(std::move(result));
                emit progress(i + 1, total);
            }
        }
        else
        {
            const bool camera = request.sourceKind == SourceKind::Camera;
            if (!camera && request.files.isEmpty())
                throw std::runtime_error("请选择视频文件。");
            cv::VideoCapture capture;
            const QString source =
                camera ? QStringLiteral("摄像头 %1").arg(request.cameraIndex) : request.files.first();
            const bool opened = camera ? capture.open(request.cameraIndex, cv::CAP_V4L2)
                                       : capture.open(QFile::encodeName(source).constData());
            if (!opened)
                throw std::runtime_error(QStringLiteral("无法打开%1。请检查文件格式、设备连接与访问权限。")
                                             .arg(source)
                                             .toUtf8()
                                             .constData());
            if (camera)
            {
                capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
                capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            }
            const double count = capture.get(cv::CAP_PROP_FRAME_COUNT);
            const int total = !camera && std::isfinite(count) && count > 0
                                  ? int(std::min(count, double(std::numeric_limits<int>::max())))
                                  : 0;
            const double rawFps = capture.get(cv::CAP_PROP_FPS);
            const double fps = std::isfinite(rawFps) && rawFps > 0.1 && rawFps < 1000 ? rawFps : 25;
            emit progress(0, total);
            emit status(camera ? QStringLiteral("摄像头实时推理中…") : QStringLiteral("视频逐帧推理中…"));
            qint64 frameNumber = 0;
            qint64 lastDeliveredNumber = 0;
            std::optional<InferenceResult> lastResult;
            QElapsedTimer displayTimer;
            displayTimer.start();
            bool receivedFrame = false;
            while (!stopping())
            {
                cv::Mat frame;
                if (!capture.read(frame) || frame.empty())
                {
                    if (camera && !stopping())
                        throw std::runtime_error("摄像头画面中断，请检查设备连接后重新运行。");
                    break;
                }
                if (stopping())
                    break;
                receivedFrame = true;
                ++frameNumber;
                QElapsedTimer frameTimer;
                frameTimer.start();
                auto result = engine.infer(frameToImage(frame), source);
                result.frameNumber = frameNumber;
                lastResult = std::move(result);
                if (stopping())
                    break;
                // Always process every video frame. Only UI delivery is throttled.
                if (frameNumber == 1 || displayTimer.elapsed() >= 67 || (total > 0 && frameNumber >= total))
                {
                    emit resultReady(*lastResult);
                    lastDeliveredNumber = frameNumber;
                    emit progress(int(std::min(frameNumber, qint64(std::numeric_limits<int>::max()))), total);
                    displayTimer.restart();
                }
                if (!camera)
                {
                    const int remaining = int(1000.0 / fps - frameTimer.elapsed());
                    for (int waited = 0; waited < remaining && !stopping(); waited += 10)
                        QThread::msleep(unsigned(std::min(10, remaining - waited)));
                }
            }
            if (!receivedFrame && !stopping())
                throw std::runtime_error("视频没有可解码的图像帧。");
            // CAP_PROP_FRAME_COUNT can be absent or inaccurate. Deliver the actual
            // last processed frame at EOF (and stop), even if UI pacing skipped it.
            if (lastResult && lastDeliveredNumber != lastResult->frameNumber)
                emit resultReady(*lastResult);
            emit progress(int(std::min(frameNumber, qint64(std::numeric_limits<int>::max()))), total);
            capture.release();
        }
        emit status(stopping() ? QStringLiteral("推理已停止") : QStringLiteral("推理完成"));
        emit finished(stopping());
    }
    catch (const cv::Exception &error)
    {
        if (!stopping())
            emit failed(QStringLiteral("视频或摄像头处理失败：%1").arg(QString::fromUtf8(error.what())));
        else
            emit status(QStringLiteral("推理已停止"));
        emit finished(stopping());
    }
    catch (const std::exception &error)
    {
        if (!stopping())
            emit failed(QString::fromUtf8(error.what()));
        else
            emit status(QStringLiteral("推理已停止"));
        emit finished(stopping());
    }
}

} // namespace vision
