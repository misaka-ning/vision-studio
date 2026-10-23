#pragma once

#include "visiontypes.h"
#include <QByteArray>
#include <QJsonObject>
#include <QString>
#include <functional>
#include <memory>

class QProcess;

namespace vision
{

// One persistent Python process per loaded .pt model, owned by the inference thread.
class PtBackend final
{
  public:
    PtBackend();
    ~PtBackend();
    PtBackend(const PtBackend &) = delete;
    PtBackend &operator=(const PtBackend &) = delete;
    void load(const ModelConfig &config);
    InferenceResult infer(const QImage &image, const QString &source);
    QVector<InferenceResult> inferBatch(const QVector<QImage> &images, const QStringList &sources);
    int maximumBatchSize() const noexcept { return m_maximumBatchSize; }
    bool loaded() const noexcept;
    const ModelConfig &config() const noexcept
    {
        return m_config;
    }
    QString backendName() const
    {
        return m_backendName;
    }
    void setCancellationCheck(std::function<bool()> check)
    {
        m_cancellationCheck = std::move(check);
    }

  private:
    void loadOnDevice(const ModelConfig &config, bool cuda);
    void reset() noexcept;
    void collectOutput();
    QJsonObject receive(int timeoutMs);
    void send(const QJsonObject &message);
    QString encodeImage(const QImage &image);
    InferenceResult decodeResult(const QJsonObject &response, const QImage &image,
                                 const QString &source);
    [[noreturn]] void processFailure(const QString &message);
    std::unique_ptr<QProcess> m_process;
    ModelConfig m_config;
    QByteArray m_stdout;
    QByteArray m_stderr;
    QString m_backendName;
    qint64 m_sequence = 0;
    bool m_loaded = false;
    int m_maximumBatchSize = 1;
    std::function<bool()> m_cancellationCheck;
};

} // namespace vision
