#include "inferenceworker.h"
#include "videorecorder.h"
#include "visionengine.h"

#include <QElapsedTimer>
#include <QFile>
#include <QImageReader>
#include <QMutexLocker>
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

QImage selectStereoView(const QImage &image, StereoView view)
{
    if (image.isNull())
        throw std::runtime_error("双目画面为空，无法选择左目或右目。");
    if (view == StereoView::Full)
        return image;
    if (image.width() < 2)
        throw std::runtime_error("双目拼接画面宽度至少需要 2 像素，请检查摄像头输出。");
    const int leftWidth = image.width() / 2;
    return view == StereoView::Left ? image.copy(0, 0, leftWidth, image.height())
                                    : image.copy(leftWidth, 0, image.width() - leftWidth, image.height());
}

void InferenceWorker::prepareRecording(const QString &directory)
{
    const QMutexLocker locker(&m_recordMutex);
    m_recordDirectory = directory;
}

void InferenceWorker::run(vision::JobRequest request)
{
    VideoRecorder recorder;
    auto finishRecording = [this, &recorder]
    {
        if (!recorder.active())
            return;
        RecordingSummary summary;
        QString error;
        if (recorder.finish(&summary, &error))
            emit recordingFinished(summary.path, summary.frames, summary.fps);
        else
            emit recordingFailed(error);
    };
    auto recordFrame = [this, &recorder, &finishRecording](const InferenceResult &result, double fps,
                                                           bool camera, const ModelConfig &config)
    {
        if (!m_recordRequested.load(std::memory_order_acquire))
        {
            finishRecording();
            return;
        }
        QString error;
        if (!recorder.active())
        {
            // A stop may arrive while infer() is producing this valid frame.
            // Retain it in an existing clip, but never open a new clip afterwards.
            if (stopping())
                return;
            QString directory;
            {
                const QMutexLocker locker(&m_recordMutex);
                directory = m_recordDirectory;
            }
            if (!recorder.start(directory, result, camera ? 30 : fps, camera, config, &error))
            {
                m_recordRequested.store(false, std::memory_order_release);
                emit recordingFailed(error);
                return;
            }
            emit recordingStarted(recorder.path());
        }
        if (!recorder.append(result, &error))
        {
            m_recordRequested.store(false, std::memory_order_release);
            finishRecording();
            emit recordingFailed(error);
        }
    };
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
            if (m_recordRequested.exchange(false, std::memory_order_acq_rel))
                emit recordingFailed(QStringLiteral("视频录制仅适用于视频或摄像头输入。"));
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
                result.sourceFrameSize = image.size();
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
                // Preserve the device's native layout. A forced 16:9 mode can
                // change a stereo camera's side-by-side output or aspect ratio.
            }
            const double count = capture.get(cv::CAP_PROP_FRAME_COUNT);
            const int total = !camera && std::isfinite(count) && count > 0
                                  ? int(std::min(count, double(std::numeric_limits<int>::max())))
                                  : 0;
            const double rawFps = capture.get(cv::CAP_PROP_FPS);
            const double fps = std::isfinite(rawFps) && rawFps > 0.1 && rawFps < 1000 ? rawFps : 25;
            emit progress(0, total);
            const QString eye = request.stereoView == StereoView::Left    ? QStringLiteral(" · 左目")
                                : request.stereoView == StereoView::Right ? QStringLiteral(" · 右目")
                                                                          : QString();
            emit status((camera ? QStringLiteral("摄像头实时推理中") : QStringLiteral("视频逐帧推理中")) +
                        eye + QStringLiteral("…"));
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
                const QImage sourceImage = frameToImage(frame);
                auto result = engine.infer(selectStereoView(sourceImage, request.stereoView), source);
                result.frameNumber = frameNumber;
                result.stereoView = request.stereoView;
                result.sourceFrameSize = sourceImage.size();
                lastResult = std::move(result);
                // Record every inferred frame before UI delivery is throttled.
                recordFrame(*lastResult, fps, camera, engine.config());
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
            finishRecording();
        }
        m_recordRequested.store(false, std::memory_order_release);
        emit status(stopping() ? QStringLiteral("推理已停止") : QStringLiteral("推理完成"));
        emit finished(stopping());
    }
    catch (const cv::Exception &error)
    {
        finishRecording();
        m_recordRequested.store(false, std::memory_order_release);
        if (!stopping())
            emit failed(QStringLiteral("视频或摄像头处理失败：%1").arg(QString::fromUtf8(error.what())));
        else
            emit status(QStringLiteral("推理已停止"));
        emit finished(stopping());
    }
    catch (const std::exception &error)
    {
        finishRecording();
        m_recordRequested.store(false, std::memory_order_release);
        if (!stopping())
            emit failed(QString::fromUtf8(error.what()));
        else
            emit status(QStringLiteral("推理已停止"));
        emit finished(stopping());
    }
}

} // namespace vision
