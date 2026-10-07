#include "core/inferenceworker.h"
#include "core/videorecorder.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest>
#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

namespace
{
QJsonObject metadata(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}

vision::InferenceResult sample(QSize size = QSize(160, 96))
{
    vision::InferenceResult result;
    result.image = QImage(size, QImage::Format_RGB888);
    result.image.fill(QColor(70, 115, 160));
    result.source = QStringLiteral("sample-video.avi");
    result.modelName = QStringLiteral("test-model.onnx");
    result.frameNumber = 7;
    result.sourceFrameSize = size;
    result.task = vision::ModelTask::YoloV8;
    result.predictions.append({0, QStringLiteral("person"), .876f, QRectF(20, 30, 45, 40)});
    return result;
}

int decodedFrames(const QString &path, QVector<cv::Mat> *frames = nullptr)
{
    cv::VideoCapture capture(QFile::encodeName(path).constData(), cv::CAP_FFMPEG);
    int count = 0;
    cv::Mat frame;
    while (capture.read(frame))
    {
        if (frames)
            frames->append(frame.clone());
        ++count;
    }
    return count;
}

QString streamFixture(const QString &directory, int frames = 10, double fps = 50)
{
    const QString path = directory + QStringLiteral("/source.avi");
    cv::VideoWriter writer(QFile::encodeName(path).constData(), cv::CAP_FFMPEG,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(160, 96));
    if (!writer.isOpened())
        return {};
    for (int i = 0; i < frames; ++i)
    {
        cv::Mat frame(96, 160, CV_8UC3, cv::Scalar(40 + i, 95, 140));
        writer.write(frame);
    }
    writer.release();
    return path;
}

vision::JobRequest videoRequest(const QString &path)
{
    vision::JobRequest request;
    request.config.modelPath = QStringLiteral(VISION_PROJECT_DIR) + QStringLiteral("/models/yolov5n.onnx");
    request.config.task = vision::ModelTask::YoloV5;
    request.config.inputSize = 640;
    request.sourceKind = vision::SourceKind::Video;
    request.files = {path};
    return request;
}
} // namespace

class RecordingTests final : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        qRegisterMetaType<vision::InferenceResult>();
        qRegisterMetaType<vision::ModelConfig>();
        qRegisterMetaType<vision::JobRequest>();
    }

    void videoIsAnnotatedPlayableAndPublishedOnlyAtFinish()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        auto frame = sample();
        const QImage original = frame.image;
        vision::VideoRecorder recorder;
        QString error;
        QVERIFY2(recorder.start(directory.path(), frame, 12.5, false, {}, &error), qPrintable(error));
        const QString finalPath = recorder.path();
        QVERIFY(!QFileInfo::exists(finalPath));
        QCOMPARE(QDir(directory.path()).entryList({"*.avi"}, QDir::Files).size(), 0);
        for (int i = 0; i < 5; ++i)
        {
            frame.frameNumber = 7 + i;
            frame.image.fill(QColor(70 + i * 15, 115, 160));
            QVERIFY2(recorder.append(frame, &error), qPrintable(error));
        }
        vision::RecordingSummary summary;
        QVERIFY2(recorder.finish(&summary, &error), qPrintable(error));
        QCOMPARE(summary.frames, qint64(5));
        QCOMPARE(summary.fps, 12.5);
        QCOMPARE(summary.contentSize, QSize(160, 96));
        QCOMPARE(summary.encodedSize, QSize(160, 96));
        QCOMPARE(summary.durationSeconds, .4);
        QVERIFY(QFileInfo::exists(finalPath));
        QCOMPARE(QDir(directory.path()).entryList({"*.avi"}, QDir::Files).size(), 1);
        QCOMPARE(QDir(directory.path()).entryList({".*.partial.avi"}, QDir::Files | QDir::Hidden).size(), 0);
        cv::VideoCapture reader(QFile::encodeName(finalPath).constData(), cv::CAP_FFMPEG);
        QVERIFY(reader.isOpened());
        QVERIFY(std::abs(reader.get(cv::CAP_PROP_FPS) - 12.5) < .001);
        QCOMPARE(int(reader.get(cv::CAP_PROP_FRAME_COUNT)), 5);
        reader.release();
        QVector<cv::Mat> decoded;
        QCOMPARE(decodedFrames(finalPath, &decoded), 5);
        QVERIFY(decoded.last().at<cv::Vec3b>(85, 140)[2] > decoded.first().at<cv::Vec3b>(85, 140)[2] + 40);
        // A box edge and confidence label are actually burned into the encoded pixels.
        QVERIFY(decoded.first().at<cv::Vec3b>(50, 20)[1] > 145);
        const QImage annotated = vision::VideoRecorder::annotatedFrame(sample());
        QVERIFY(annotated != original);
        QCOMPARE(sample().image, original);
        int changed = 0;
        for (int y = 8; y < 30; ++y)
            for (int x = 20; x < 100; ++x)
                changed += annotated.pixelColor(x, y) != original.pixelColor(x, y);
        QVERIFY(changed > 150);
        const auto json = metadata(summary.metadataPath);
        QCOMPARE(json.value("schema_version").toInt(), 1);
        QCOMPARE(json.value("requested_device").toString(), QStringLiteral("cpu"));
        QCOMPARE(json.value("actual_device").toString(), QStringLiteral("cpu"));
        QCOMPARE(json.value("device_index").toInt(), -1);
        QCOMPARE(json.value("frames").toInt(), 5);
        QCOMPARE(json.value("first_source_frame").toInt(), 7);
        QCOMPARE(json.value("last_source_frame").toInt(), 11);
        QCOMPARE(json.value("timing_mode").toString(), QStringLiteral("source_fps"));
        QCOMPARE(json.value("frame_times_ms").toArray().size(), 5);
    }

    void grayscaleOddEyePreservesContentAndAddsOnlyEdgePadding()
    {
        QTemporaryDir directory;
        auto frame = sample(QSize(65, 47));
        frame.predictions.clear();
        frame.stereoView = vision::StereoView::Right;
        frame.sourceFrameSize = QSize(129, 47);
        frame.originalImage = frame.image;
        frame.image.fill(QColor(100, 100, 100));
        for (int y = 0; y < frame.image.height(); ++y)
            frame.image.setPixelColor(64, y, QColor(235, 235, 235));
        vision::ModelConfig config;
        config.colorMode = vision::InputColorMode::Grayscale;
        vision::VideoRecorder recorder;
        QString error;
        QVERIFY2(recorder.start(directory.path(), frame, 20, false, config, &error), qPrintable(error));
        QVERIFY2(recorder.append(frame, &error), qPrintable(error));
        vision::RecordingSummary summary;
        QVERIFY2(recorder.finish(&summary, &error), qPrintable(error));
        QCOMPARE(summary.contentSize, QSize(65, 47));
        QCOMPARE(summary.encodedSize, QSize(66, 48));
        QVector<cv::Mat> decoded;
        QCOMPARE(decodedFrames(summary.path, &decoded), 1);
        QCOMPARE(decoded.first().cols, 66);
        QCOMPARE(decoded.first().rows, 48);
        const auto lastContentColumn = decoded.first().at<cv::Vec3b>(35, 64);
        const auto addedColumn = decoded.first().at<cv::Vec3b>(35, 65);
        QVERIFY(lastContentColumn[0] > 210);
        QVERIFY(std::abs(int(lastContentColumn[0]) - int(addedColumn[0])) < 8);
        const auto pixel = decoded.first().at<cv::Vec3b>(35, 40);
        QVERIFY(std::abs(int(pixel[0]) - int(pixel[1])) < 3);
        QVERIFY(std::abs(int(pixel[1]) - int(pixel[2])) < 3);
        const auto json = metadata(summary.metadataPath);
        QCOMPARE(json.value("color_mode").toString(), QStringLiteral("grayscale"));
        QCOMPARE(json.value("stereo_view").toString(), QStringLiteral("right"));
        QCOMPARE(json.value("content_width").toInt(), 65);
        QCOMPARE(json.value("content_height").toInt(), 47);
        QCOMPARE(json.value("encoded_width").toInt(), 66);
        QCOMPARE(json.value("source_frame_size").toObject().value("width").toInt(), 129);
    }

    void cameraTimingUsesMeasuredDurationAndDecodesAllFrames()
    {
        QTemporaryDir directory;
        auto frame = sample();
        vision::VideoRecorder recorder;
        QString error;
        QVERIFY2(recorder.start(directory.path(), frame, 30, true, {}, &error), qPrintable(error));
        for (int i = 0; i < 4; ++i)
        {
            frame.frameNumber = i + 1;
            QVERIFY2(recorder.append(frame, &error), qPrintable(error));
            QTest::qWait(60); // Deliberately lower than the nominal 30 FPS camera rate.
        }
        vision::RecordingSummary summary;
        QVERIFY2(recorder.finish(&summary, &error), qPrintable(error));
        QVERIFY(summary.fps > 5 && summary.fps < 22);
        QVERIFY(summary.durationSeconds >= .24);
        cv::VideoCapture reader(QFile::encodeName(summary.path).constData(), cv::CAP_FFMPEG);
        QVERIFY(reader.isOpened());
        QVERIFY(std::abs(reader.get(cv::CAP_PROP_FPS) - summary.fps) < .001);
        QCOMPARE(int(reader.get(cv::CAP_PROP_FRAME_COUNT)), 4);
        reader.release();
        QCOMPARE(decodedFrames(summary.path), 4);
        const auto json = metadata(summary.metadataPath);
        QCOMPARE(json.value("timing_mode").toString(), QStringLiteral("measured_realtime"));
        QVERIFY(std::abs(json.value("duration_seconds").toDouble() - summary.durationSeconds) < .001);
        const auto times = json.value("frame_times_ms").toArray();
        QCOMPARE(times.size(), 4);
        QVERIFY(times.last().toDouble() - times.first().toDouble() >= 180);
    }

    void emptyDiscardAndInvalidDirectoryNeverPublishBrokenClips()
    {
        QTemporaryDir directory;
        const auto frame = sample();
        vision::VideoRecorder recorder;
        QString error;
        QVERIFY(recorder.start(directory.path(), frame, 25, false, {}, &error));
        const QString emptyPath = recorder.path();
        QVERIFY(!recorder.finish(nullptr, &error));
        QVERIFY(!error.isEmpty());
        QVERIFY(!QFileInfo::exists(emptyPath));
        {
            vision::VideoRecorder abandoned;
            QVERIFY(abandoned.start(directory.path(), frame, 25, false, {}, &error));
            QVERIFY(abandoned.append(frame, &error));
        }
        QCOMPARE(QDir(directory.path()).entryList(QDir::Files | QDir::Hidden).size(), 0);
        const QString blocker = directory.path() + QStringLiteral("/not-a-directory");
        QFile file(blocker);
        QVERIFY(file.open(QIODevice::WriteOnly));
        QCOMPARE(file.write("keep"), qint64(4));
        file.close();
        QVERIFY(!recorder.start(blocker, frame, 25, false, {}, &error));
        QVERIFY(file.open(QIODevice::ReadOnly));
        QCOMPARE(file.readAll(), QByteArray("keep"));
    }

    void changedResolutionEndsClipWithoutLosingItsPreviousFrames()
    {
        QTemporaryDir directory;
        auto frame = sample();
        vision::VideoRecorder recorder;
        QString error;
        QVERIFY(recorder.start(directory.path(), frame, 25, false, {}, &error));
        QVERIFY(recorder.append(frame, &error));
        QVERIFY(recorder.append(frame, &error));
        QVERIFY(!recorder.append(sample(QSize(162, 96)), &error));
        QVERIFY(error.contains(QStringLiteral("尺寸")));
        vision::RecordingSummary summary;
        QVERIFY(recorder.finish(&summary, &error));
        QCOMPARE(summary.frames, qint64(2));
        QCOMPARE(decodedFrames(summary.path), 2);
    }

    void workerRecordsEveryFrameAndFinishesAtEof()
    {
        QTemporaryDir directory;
        const QString input = streamFixture(directory.path(), 8);
        QVERIFY(!input.isEmpty());
        vision::InferenceWorker worker;
        worker.prepareRecording(directory.path() + QStringLiteral("/recordings"));
        worker.prepare();
        connect(&worker, &vision::InferenceWorker::modelReady, &worker,
                [&](const QString &, const vision::ModelConfig &) { worker.requestStartRecording(); });
        QSignalSpy started(&worker, &vision::InferenceWorker::recordingStarted);
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        QCOMPARE(errors.size(), 0);
        QCOMPARE(recordingErrors.size(), 0);
        QCOMPARE(started.size(), 1);
        QCOMPARE(recordings.size(), 1);
        QCOMPARE(finished.size(), 1);
        QCOMPARE(finished.first().at(0).toBool(), false);
        const QString path = recordings.first().at(0).toString();
        QCOMPARE(recordings.first().at(1).toLongLong(), qint64(8));
        QVERIFY(std::abs(recordings.first().at(2).toDouble() - 50) < .001);
        QCOMPARE(decodedFrames(path), 8);
    }

    void gpuRightEyeGrayscaleRecordingPreservesActualDevice()
    {
        if (qEnvironmentVariable("VISION_STUDIO_GPU_TESTS") != "1")
            QSKIP("Real GPU recording requires VISION_STUDIO_GPU_TESTS=1.");
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString input = directory.path() + QStringLiteral("/stereo.avi");
        cv::VideoWriter writer(QFile::encodeName(input).constData(), cv::CAP_FFMPEG,
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(320, 96));
        QVERIFY(writer.isOpened());
        for (int i = 0; i < 8; ++i)
        {
            cv::Mat frame(96, 320, CV_8UC3, cv::Scalar::all(12));
            frame(cv::Rect(160, 0, 160, 96)).setTo(cv::Scalar(70 + 4 * i, 130, 190));
            writer.write(frame);
        }
        writer.release();
        auto request = videoRequest(input);
        request.config.device = vision::ComputeDevice::CUDA;
        request.config.colorMode = vision::InputColorMode::Grayscale;
        request.stereoView = vision::StereoView::Right;
        vision::InferenceWorker worker;
        worker.prepareRecording(directory.path() + QStringLiteral("/recordings"));
        worker.prepare();
        connect(&worker, &vision::InferenceWorker::modelReady, &worker,
                [&](const QString &, const vision::ModelConfig &) { worker.requestStartRecording(); });
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        worker.run(request);
        QVERIFY2(errors.isEmpty(), errors.isEmpty() ? "" : qPrintable(errors.first().first().toString()));
        QCOMPARE(recordingErrors.size(), 0);
        // UI previews are throttled independently of the recording stream.
        // Fast CUDA inference must still record every source frame.
        QVERIFY(results.size() >= 2 && results.size() <= 8);
        QCOMPARE(qvariant_cast<vision::InferenceResult>(results.first().first()).frameNumber, qint64(1));
        QCOMPARE(qvariant_cast<vision::InferenceResult>(results.last().first()).frameNumber, qint64(8));
        QCOMPARE(recordings.size(), 1);
        QCOMPARE(recordings.first().at(1).toLongLong(), qint64(8));
        for (const auto &arguments : results)
        {
            const auto result = qvariant_cast<vision::InferenceResult>(arguments.first());
            QCOMPARE(result.requestedDevice, vision::ComputeDevice::CUDA);
            QCOMPARE(result.device, vision::ComputeDevice::CUDA);
            QCOMPARE(result.deviceIndex, 0);
            QVERIFY(!result.deviceName.isEmpty());
            QCOMPARE(result.stereoView, vision::StereoView::Right);
            QCOMPARE(result.sourceFrameSize, QSize(320, 96));
            QCOMPARE(result.image.size(), QSize(160, 96));
            QCOMPARE(result.originalImage.size(), QSize(160, 96));
            QCOMPARE(result.image.format(), QImage::Format_Grayscale8);
            const QColor pixel = result.image.pixelColor(80, 48);
            QVERIFY(pixel.red() > 100);
            QCOMPARE(pixel.red(), pixel.green());
            QCOMPARE(pixel.green(), pixel.blue());
        }
        const QString output = recordings.first().first().toString();
        QVector<cv::Mat> decoded;
        QCOMPARE(decodedFrames(output, &decoded), 8);
        for (const auto &frame : decoded)
        {
            QCOMPARE(frame.cols, 160);
            QCOMPARE(frame.rows, 96);
            const auto pixel = frame.at<cv::Vec3b>(48, 80);
            QVERIFY(pixel[0] > 100);
            QVERIFY(std::abs(int(pixel[0]) - int(pixel[1])) <= 2);
            QVERIFY(std::abs(int(pixel[1]) - int(pixel[2])) <= 2);
        }
        const auto json = metadata(output.left(output.size() - 4) + QStringLiteral(".json"));
        QCOMPARE(json.value("frames").toInt(), 8);
        QCOMPARE(json.value("requested_device").toString(), QStringLiteral("cuda"));
        QCOMPARE(json.value("actual_device").toString(), QStringLiteral("cuda"));
        QCOMPARE(json.value("device_index").toInt(), 0);
        QVERIFY(!json.value("device_name").toString().isEmpty());
        QCOMPARE(json.value("color_mode").toString(), QStringLiteral("grayscale"));
        QCOMPARE(json.value("stereo_view").toString(), QStringLiteral("right"));
        QCOMPARE(json.value("width").toInt(), 160);
        QCOMPARE(json.value("height").toInt(), 96);
    }

    void workerStartAfterFirstPreviewAndCancelClosesOnlyRecordedFrames()
    {
        QTemporaryDir directory;
        const QString input = streamFixture(directory.path(), 16);
        QVERIFY(!input.isEmpty());
        vision::InferenceWorker worker;
        worker.prepareRecording(directory.path() + QStringLiteral("/recordings"));
        worker.prepare();
        qint64 stoppedFrame = 0;
        connect(&worker, &vision::InferenceWorker::resultReady, &worker,
                [&](const vision::InferenceResult &result)
                {
                    if (result.frameNumber == 1)
                        worker.requestStartRecording();
                    else if (result.frameNumber >= 4)
                    {
                        stoppedFrame = result.frameNumber;
                        worker.requestStop();
                    }
                });
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        QCOMPARE(errors.size(), 0);
        QCOMPARE(recordingErrors.size(), 0);
        QCOMPARE(finished.first().at(0).toBool(), true);
        QCOMPARE(recordings.size(), 1);
        QVERIFY(stoppedFrame >= 4 && stoppedFrame < 16);
        const QString path = recordings.first().at(0).toString();
        QCOMPARE(recordings.first().at(1).toLongLong(), stoppedFrame - 1);
        QCOMPARE(decodedFrames(path), int(stoppedFrame - 1));
        const auto json = metadata(path.left(path.size() - 4) + ".json");
        QCOMPARE(json.value("first_source_frame").toInt(), 2);
        QCOMPARE(json.value("last_source_frame").toInt(), int(stoppedFrame));
    }

    void recordingFailureDoesNotInterruptInference()
    {
        QTemporaryDir directory;
        const QString input = streamFixture(directory.path(), 4);
        const QString blocker = directory.path() + QStringLiteral("/blocker");
        QFile file(blocker);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.close();
        vision::InferenceWorker worker;
        worker.prepareRecording(blocker);
        worker.prepare();
        connect(&worker, &vision::InferenceWorker::modelReady, &worker,
                [&](const QString &, const vision::ModelConfig &) { worker.requestStartRecording(); });
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        QCOMPARE(recordingErrors.size(), 1);
        QCOMPARE(errors.size(), 0);
        QCOMPARE(finished.first().at(0).toBool(), false);
        QVERIFY(!results.isEmpty());
        QCOMPARE(qvariant_cast<vision::InferenceResult>(results.last().at(0)).frameNumber, qint64(4));
    }

    void manualStopKeepsInferenceRunningAndRestartCreatesANewClip()
    {
        QTemporaryDir directory;
        const QString input = streamFixture(directory.path(), 16);
        QVERIFY(!input.isEmpty());
        vision::InferenceWorker worker;
        worker.prepareRecording(directory.path() + QStringLiteral("/recordings"));
        worker.prepare();
        connect(&worker, &vision::InferenceWorker::modelReady, &worker,
                [&](const QString &, const vision::ModelConfig &) { worker.requestStartRecording(); });
        qint64 restartedAfterFrame = 0;
        connect(&worker, &vision::InferenceWorker::resultReady, &worker,
                [&](const vision::InferenceResult &result)
                {
                    if (result.frameNumber == 1)
                        worker.requestStopRecording();
                    else if (!restartedAfterFrame && result.frameNumber >= 3 && result.frameNumber < 16)
                    {
                        restartedAfterFrame = result.frameNumber;
                        worker.requestStartRecording();
                    }
                });
        QSignalSpy started(&worker, &vision::InferenceWorker::recordingStarted);
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        QCOMPARE(errors.size(), 0);
        QCOMPARE(recordingErrors.size(), 0);
        QCOMPARE(finished.first().at(0).toBool(), false);
        QCOMPARE(started.size(), 2);
        QCOMPARE(recordings.size(), 2);
        QVERIFY(restartedAfterFrame >= 3 && restartedAfterFrame < 16);
        const QString first = recordings.first().at(0).toString();
        const QString second = recordings.last().at(0).toString();
        QVERIFY(first != second);
        QCOMPARE(recordings.first().at(1).toLongLong(), qint64(1));
        QCOMPARE(recordings.last().at(1).toLongLong(), 16 - restartedAfterFrame);
        QCOMPARE(decodedFrames(first), 1);
        QCOMPARE(decodedFrames(second), int(16 - restartedAfterFrame));
        const auto json = metadata(second.left(second.size() - 4) + ".json");
        QCOMPARE(json.value("first_source_frame").toInt(), int(restartedAfterFrame + 1));
        QCOMPARE(json.value("last_source_frame").toInt(), 16);
    }

    void stoppingDuringOnnxForwardRecordsTheLastCompletedFrame()
    {
        QTemporaryDir directory;
        // High source FPS removes pacing between frames, so the stop arrives in
        // the next real ONNX forward rather than during a source-rate wait.
        const QString input = streamFixture(directory.path(), 12, 500);
        QVERIFY(!input.isEmpty());
        vision::InferenceWorker worker;
        worker.prepareRecording(directory.path() + QStringLiteral("/recordings"));
        worker.prepare();
        connect(&worker, &vision::InferenceWorker::modelReady, &worker,
                [&](const QString &, const vision::ModelConfig &) { worker.requestStartRecording(); });
        std::thread stopThread;
        connect(&worker, &vision::InferenceWorker::resultReady, &worker,
                [&](const vision::InferenceResult &result)
                {
                    if (result.frameNumber == 1)
                        stopThread = std::thread(
                            [&worker]
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                                worker.requestStop();
                            });
                });
        QSignalSpy results(&worker, &vision::InferenceWorker::resultReady);
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy recordingErrors(&worker, &vision::InferenceWorker::recordingFailed);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        if (stopThread.joinable())
            stopThread.join();
        QCOMPARE(errors.size(), 0);
        QCOMPARE(recordingErrors.size(), 0);
        QCOMPARE(finished.first().at(0).toBool(), true);
        QCOMPARE(recordings.size(), 1);
        QVERIFY(!results.isEmpty());
        const auto last = qvariant_cast<vision::InferenceResult>(results.last().at(0));
        QVERIFY2(last.frameNumber >= 2, "The stop must arrive after the next ONNX frame has begun.");
        const QString path = recordings.first().at(0).toString();
        QCOMPARE(recordings.first().at(1).toLongLong(), last.frameNumber);
        QCOMPARE(decodedFrames(path), int(last.frameNumber));
        const auto json = metadata(path.left(path.size() - 4) + ".json");
        QCOMPARE(json.value("last_source_frame").toInt(), int(last.frameNumber));
    }

    void stoppingWithPendingRecordingNeverOpensANewClip()
    {
        QTemporaryDir directory;
        const QString input = streamFixture(directory.path(), 12, 500);
        QVERIFY(!input.isEmpty());
        vision::InferenceWorker worker;
        const QString output = directory.path() + QStringLiteral("/recordings");
        worker.prepareRecording(output);
        worker.prepare();
        std::thread stopThread;
        connect(&worker, &vision::InferenceWorker::status, &worker,
                [&](const QString &message)
                {
                    if (message.startsWith(QStringLiteral("视频逐帧推理中")))
                    {
                        worker.requestStartRecording();
                        stopThread = std::thread(
                            [&worker]
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                                worker.requestStop();
                            });
                    }
                });
        QSignalSpy started(&worker, &vision::InferenceWorker::recordingStarted);
        QSignalSpy recordings(&worker, &vision::InferenceWorker::recordingFinished);
        QSignalSpy errors(&worker, &vision::InferenceWorker::failed);
        QSignalSpy finished(&worker, &vision::InferenceWorker::finished);
        worker.run(videoRequest(input));
        if (stopThread.joinable())
            stopThread.join();
        QCOMPARE(errors.size(), 0);
        QCOMPARE(finished.first().at(0).toBool(), true);
        QCOMPARE(started.size(), 0);
        QCOMPARE(recordings.size(), 0);
        QCOMPARE(QDir(output).entryList({"*.avi"}, QDir::Files).size(), 0);
    }
};

QTEST_MAIN(RecordingTests)
#include "recording_tests.moc"
