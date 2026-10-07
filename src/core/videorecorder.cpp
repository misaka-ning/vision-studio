#include "videorecorder.h"
#include "gpuruntime.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QFontMetrics>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QSaveFile>
#include <QUuid>
#include <QtEndian>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace vision
{
namespace
{
bool failure(QString *error, const QString &message)
{
    if (error)
        *error = message;
    return false;
}

QString stereoName(StereoView view)
{
    return view == StereoView::Left ? "left" : (view == StereoView::Right ? "right" : "full");
}

QColor classColor(int id)
{
    return QColor::fromHsv((std::max(0, id) * 67 + 158) % 360, 172, 242);
}

bool writeUint32(QFile &file, qint64 position, quint32 value)
{
    const quint32 little = qToLittleEndian(value);
    return file.seek(position) && file.write(reinterpret_cast<const char *>(&little), 4) == 4;
}

struct AviTimingOffsets
{
    qint64 main = -1, scale = -1, rate = -1;
};

// Parse chunk boundaries, rather than searching arbitrary compressed JPEG bytes for FOURCCs.
bool timingOffsets(QFile &file, qint64 begin, qint64 end, AviTimingOffsets &offsets, int depth = 0)
{
    if (depth > 8 || begin < 0 || end > file.size())
        return false;
    for (qint64 position = begin; position + 8 <= end;)
    {
        if (!file.seek(position))
            return false;
        const QByteArray header = file.read(8);
        if (header.size() != 8)
            return false;
        const QByteArray id = header.left(4);
        const quint32 length =
            qFromLittleEndian<quint32>(reinterpret_cast<const uchar *>(header.constData() + 4));
        const qint64 payload = position + 8, next = payload + qint64(length) + (length & 1u);
        if (next > end || next <= position)
            return false;
        if (id == "avih" && length >= 56)
            offsets.main = payload;
        else if (id == "strh" && length >= 56)
        {
            if (!file.seek(payload))
                return false;
            if (file.read(4) == "vids")
            {
                offsets.scale = payload + 20;
                offsets.rate = payload + 24;
            }
        }
        else if (id == "LIST" && length >= 4)
        {
            if (!file.seek(payload))
                return false;
            const QByteArray listType = file.read(4);
            if ((listType == "hdrl" || listType == "strl") &&
                !timingOffsets(file, payload + 4, payload + length, offsets, depth + 1))
                return false;
        }
        // All timing headers live before the potentially multi-gigabyte movie data.
        if (offsets.main >= 0 && offsets.scale >= 0 && offsets.rate >= 0)
            return true;
        position = next;
    }
    return true;
}

bool setAviFrameRate(const QString &path, double fps, QString *error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadWrite))
        return failure(error, "无法完成录制时间信息：" + file.errorString());
    const QByteArray header = file.read(12);
    if (header.size() != 12 || header.left(4) != "RIFF" || header.mid(8, 4) != "AVI ")
        return failure(error, "录制文件不是可识别的 AVI，未发布最终文件。");
    const quint32 riffLength =
        qFromLittleEndian<quint32>(reinterpret_cast<const uchar *>(header.constData() + 4));
    AviTimingOffsets offsets;
    const qint64 end = std::min(file.size(), qint64(riffLength) + 8);
    if (!timingOffsets(file, 12, end, offsets) || offsets.main < 0 || offsets.scale < 0 || offsets.rate < 0)
        return failure(error, "录制文件缺少 AVI 视频时间基准，未发布最终文件。");
    // AVI specifies dwRate / dwScale as frames per second, and avih's first DWORD
    // as microseconds per frame. See Microsoft's AVISTREAMHEADER / AVIMAINHEADER.
    // https://learn.microsoft.com/en-us/previous-versions/ms779638(v=vs.85)
    // https://learn.microsoft.com/en-us/previous-versions/ms779632(v=vs.85)
    constexpr quint32 scale = 1000000;
    const quint32 rate = quint32(std::llround(fps * scale));
    const quint32 period = quint32(std::llround(1000000.0 / fps));
    if (!rate || !period || !writeUint32(file, offsets.main, period) ||
        !writeUint32(file, offsets.scale, scale) || !writeUint32(file, offsets.rate, rate) || !file.flush())
        return failure(error, "无法写入录制时间信息：" + file.errorString());
    return true;
}

} // namespace

VideoRecorder::~VideoRecorder()
{
    discard();
}

bool VideoRecorder::start(const QString &directory, const InferenceResult &firstFrame, double sourceFps,
                          bool realtime, const ModelConfig &config, QString *error)
{
    if (m_active)
        return failure(error, "当前录制尚未结束。");
    if (firstFrame.image.isNull() || firstFrame.image.width() < 2 || firstFrame.image.height() < 2)
        return failure(error, "录制需要至少 2 × 2 像素的有效画面。");
    if (directory.trimmed().isEmpty() || !QDir().mkpath(directory) || !QFileInfo(directory).isDir())
        return failure(error, "无法创建录制目录：" + directory);
    m_created = QDateTime::currentDateTime();
    const QString name = "vision-" + m_created.toString("yyyyMMdd-HHmmss-zzz") + "-" +
                         QUuid::createUuid().toString(QUuid::Id128).left(12);
    const QDir output(QFileInfo(directory).absoluteFilePath());
    m_path = output.filePath(name + ".avi");
    m_temporaryPath = output.filePath("." + name + ".partial.avi");
    m_contentSize = firstFrame.image.size();
    m_encodedSize = QSize((m_contentSize.width() + 1) & ~1, (m_contentSize.height() + 1) & ~1);
    m_firstFrame = firstFrame;
    m_firstFrame.image = {}; // Keep metadata, without retaining an extra full-sized frame.
    m_firstFrame.originalImage = {};
    m_config = config;
    m_sourceFps = std::isfinite(sourceFps) && sourceFps >= 0.1 && sourceFps <= 1000 ? sourceFps : 30;
    m_realtime = realtime;
    m_frames = 0;
    m_lastSourceFrame = 0;
    m_frameTimes.clear();
    QFile reserved(m_temporaryPath);
    if (!reserved.open(QIODevice::WriteOnly | QIODevice::NewOnly))
    {
        m_temporaryPath.clear(); // A colliding or inaccessible file does not belong to this recorder.
        return failure(error, "无法创建录制文件：" + reserved.errorString());
    }
    reserved.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
    reserved.close();
    try
    {
        const bool opened = m_writer.open(QFile::encodeName(m_temporaryPath).constData(), cv::CAP_FFMPEG,
                                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_sourceFps,
                                          cv::Size(m_encodedSize.width(), m_encodedSize.height()), true);
        if (!opened)
        {
            discard();
            return failure(error, "本机无法创建 MJPEG / AVI 录制，请检查输出目录与视频编码环境。");
        }
        m_active = true;
        m_timer.start();
        return true;
    }
    catch (const cv::Exception &exception)
    {
        discard();
        return failure(error, "无法开始视频录制：" + QString::fromUtf8(exception.what()));
    }
}

QImage VideoRecorder::annotatedFrame(const InferenceResult &frame)
{
    if (frame.image.isNull())
        return {};
    QImage output = frame.image.convertToFormat(QImage::Format_RGB888);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont font;
    font.setPixelSize(std::clamp(output.width() / 65, 12, 28));
    painter.setFont(font);
    const QFontMetrics metrics(font);
    const qreal lineWidth = std::clamp(output.width() / 480.0, 1.5, 5.0);
    const QRectF bounds(0, 0, output.width(), output.height());
    QVector<QRectF> labels;
    int classificationRow = 0;
    for (const Prediction &prediction : frame.predictions)
    {
        if (!std::isfinite(prediction.confidence))
            continue;
        const QColor color = classColor(prediction.classId);
        const QString name =
            prediction.label.isEmpty() ? QString::number(prediction.classId) : prediction.label;
        const QString caption = name + "  " + QString::number(prediction.confidence * 100, 'f', 1) + "%";
        const qreal width = std::min(qreal(output.width()), qreal(metrics.horizontalAdvance(caption) + 14));
        const qreal height = std::min(qreal(output.height()), qreal(metrics.height() + 8));
        QRectF box;
        if (frame.task != ModelTask::Classification)
        {
            if (!std::isfinite(prediction.box.x()) || !std::isfinite(prediction.box.y()) ||
                !std::isfinite(prediction.box.width()) || !std::isfinite(prediction.box.height()))
                continue;
            box = prediction.box.intersected(
                bounds.adjusted(lineWidth / 2, lineWidth / 2, -lineWidth / 2, -lineWidth / 2));
            if (box.isEmpty())
                continue;
            painter.setPen(QPen(color, lineWidth));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(box);
        }
        const qreal x = frame.task == ModelTask::Classification ? 8 : box.left();
        const qreal y = frame.task == ModelTask::Classification ? 8 + classificationRow++ * (height + 5)
                                                                : box.top() - height;
        QRectF label(std::clamp(x, 0.0, output.width() - width), std::clamp(y, 0.0, output.height() - height),
                     width, height);
        // A compact collision pass keeps neighbouring confidence labels readable.
        for (int attempt = 0; attempt < labels.size() + 1; ++attempt)
        {
            bool collision = false;
            for (const QRectF &used : labels)
                if (label.intersects(used))
                {
                    collision = true;
                    label.moveTop(std::min(qreal(output.height()) - height, used.bottom() + 2));
                    break;
                }
            if (!collision)
                break;
        }
        labels.append(label);
        painter.setPen(Qt::NoPen);
        QColor background = color.darker(190);
        background.setAlpha(235);
        painter.setBrush(background);
        painter.drawRoundedRect(label, 3, 3);
        painter.setPen(Qt::white);
        painter.drawText(label.adjusted(7, 0, -7, 0), Qt::AlignVCenter | Qt::AlignLeft,
                         metrics.elidedText(caption, Qt::ElideRight, std::max(1, int(width) - 14)));
    }
    return output;
}

bool VideoRecorder::append(const InferenceResult &frame, QString *error)
{
    if (!m_active)
        return failure(error, "视频录制尚未开始。");
    if (frame.image.isNull() || frame.image.size() != m_contentSize ||
        frame.stereoView != m_firstFrame.stereoView)
        return failure(error, "录制期间画面尺寸或所选眼发生变化，当前片段已结束。");
    try
    {
        const QImage annotated = annotatedFrame(frame);
        cv::Mat rgb(annotated.height(), annotated.width(), CV_8UC3,
                    const_cast<uchar *>(annotated.constBits()), size_t(annotated.bytesPerLine()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        if (m_encodedSize != m_contentSize)
            cv::copyMakeBorder(bgr, bgr, 0, m_encodedSize.height() - m_contentSize.height(), 0,
                               m_encodedSize.width() - m_contentSize.width(), cv::BORDER_REPLICATE);
        m_writer.write(bgr);
        m_frameTimes.append(m_realtime ? m_timer.elapsed()
                                       : qint64(std::llround(m_frames * 1000.0 / m_sourceFps)));
        ++m_frames;
        m_lastSourceFrame = frame.frameNumber;
        return true;
    }
    catch (const cv::Exception &exception)
    {
        return failure(error, "视频帧录制失败：" + QString::fromUtf8(exception.what()));
    }
}

bool VideoRecorder::finish(RecordingSummary *summary, QString *error)
{
    if (!m_active)
        return failure(error, "视频录制尚未开始。");
    const qint64 elapsed = m_timer.elapsed();
    m_active = false;
    try
    {
        m_writer.release();
        if (m_frames == 0)
        {
            discard();
            return failure(error, "录制没有收到有效图像帧。");
        }
        const double fps =
            m_realtime && elapsed > 0 ? std::clamp(m_frames * 1000.0 / elapsed, 0.001, 1000.0) : m_sourceFps;
        if (m_realtime && !setAviFrameRate(m_temporaryPath, fps, error))
        {
            discard();
            return false;
        }
        cv::VideoCapture validation(QFile::encodeName(m_temporaryPath).constData(), cv::CAP_FFMPEG);
        if (!validation.isOpened() ||
            qint64(std::llround(validation.get(cv::CAP_PROP_FRAME_COUNT))) != m_frames ||
            int(validation.get(cv::CAP_PROP_FRAME_WIDTH)) != m_encodedSize.width() ||
            int(validation.get(cv::CAP_PROP_FRAME_HEIGHT)) != m_encodedSize.height())
        {
            validation.release();
            discard();
            return failure(error, "录制文件未完整写入，未发布最终文件；请检查可用磁盘空间。");
        }
        cv::Mat first;
        if (!validation.read(first) || first.empty())
        {
            validation.release();
            discard();
            return failure(error, "录制视频无法解码，未发布最终文件。");
        }
        if (m_frames > 1)
        {
            cv::Mat last;
            if (!validation.set(cv::CAP_PROP_POS_FRAMES, double(m_frames - 1)) || !validation.read(last) ||
                last.empty())
            {
                validation.release();
                discard();
                return failure(error, "录制视频的末帧不完整，未发布最终文件；请检查可用磁盘空间。");
            }
        }
        validation.release();
        const QString metadataPath = m_path.left(m_path.size() - 4) + ".json";
        QJsonArray times;
        for (qint64 time : m_frameTimes)
            times.append(double(time));
        const QJsonObject metadata{
            {"schema_version", 1},
            {"application", "Vision Studio"},
            {"codec", "MJPG"},
            {"container", "AVI"},
            {"created_at", m_created.toString(Qt::ISODateWithMs)},
            {"frames", double(m_frames)},
            {"fps", fps},
            {"duration_seconds", m_frames / fps},
            {"width", m_contentSize.width()},
            {"height", m_contentSize.height()},
            {"content_width", m_contentSize.width()},
            {"content_height", m_contentSize.height()},
            {"encoded_width", m_encodedSize.width()},
            {"encoded_height", m_encodedSize.height()},
            {"color_mode", m_config.colorMode == InputColorMode::Grayscale
                               ? "grayscale"
                               : (m_config.swapRB ? "rgb" : "bgr")},
            {"stereo_view", stereoName(m_firstFrame.stereoView)},
            {"source", m_firstFrame.source},
            {"model", m_firstFrame.modelName},
            {"model_file", m_config.modelPath},
            {"backend", m_firstFrame.backend},
            {"requested_device", computeDeviceKey(m_firstFrame.requestedDevice)},
            {"actual_device", computeDeviceKey(m_firstFrame.device)},
            {"device_index", m_firstFrame.deviceIndex},
            {"device_name", m_firstFrame.deviceName},
            {"device_notice", m_firstFrame.deviceNotice},
            {"source_frame_size", QJsonObject{{"width", m_firstFrame.sourceFrameSize.width()},
                                              {"height", m_firstFrame.sourceFrameSize.height()}}},
            {"first_source_frame", double(m_firstFrame.frameNumber)},
            {"last_source_frame", double(m_lastSourceFrame)},
            {"timing_mode", m_realtime ? "measured_realtime" : "source_fps"},
            {"frame_times_ms", times}};
        QSaveFile json(metadataPath);
        const QByteArray bytes = QJsonDocument(metadata).toJson();
        if (!json.open(QIODevice::WriteOnly) || json.write(bytes) != bytes.size() || !json.commit())
        {
            discard();
            return failure(error, "录制信息写入失败：" + json.errorString());
        }
        if (!QFile::rename(m_temporaryPath, m_path))
        {
            QFile::remove(metadataPath);
            discard();
            return failure(error, "无法发布最终录制文件，请检查目录访问权限。");
        }
        if (summary)
            *summary = {m_path, metadataPath, m_frames, fps, m_frames / fps, m_contentSize, m_encodedSize};
        m_temporaryPath.clear();
        m_frameTimes.clear();
        return true;
    }
    catch (const cv::Exception &exception)
    {
        discard();
        return failure(error, "视频录制封口失败：" + QString::fromUtf8(exception.what()));
    }
}

void VideoRecorder::discard() noexcept
{
    try
    {
        m_writer.release();
    }
    catch (...)
    {
    }
    m_active = false;
    if (!m_temporaryPath.isEmpty())
        QFile::remove(m_temporaryPath);
    m_temporaryPath.clear();
    m_frameTimes.clear();
}

} // namespace vision
