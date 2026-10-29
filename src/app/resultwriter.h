#pragma once

#include "core/visiontypes.h"
#include <QByteArray>
#include <QJsonObject>
#include <QObject>
#include <memory>

class QTemporaryDir;

namespace vision::app
{
struct PreparedResultExport
{
    QString base;
    InferenceResult result;
    QByteArray json, csv;
};

struct ResultWriteRequest
{
    InferenceResult result;
    quint64 ticket = 0;
    quint64 generation = 0;
    // Holds the temporary cache tree until this accepted write has completed.
    std::shared_ptr<QTemporaryDir> directory;
    QString snapshotPath;
    bool automaticExport = false;
    PreparedResultExport exported;
    QString historyPath;
    QByteArray history;
};

struct ResultWriteOutcome
{
    QString source;
    quint64 ticket = 0;
    quint64 generation = 0;
    std::shared_ptr<QTemporaryDir> directory;
    QString snapshotPath;
    bool snapshotSaved = false;
    bool automaticExport = false;
    bool exportSaved = true;
    bool historySaved = true;
    QString snapshotError, exportError, historyError;
};

// Serial background writes without a GUI-thread-owned pool whose destructor waits.
// Call requestShutdown and wait for shutdownReady before normal destruction.
class ResultWriter final : public QObject
{
    Q_OBJECT
  public:
    explicit ResultWriter(QObject *parent = nullptr);
    ~ResultWriter() override;
    static PreparedResultExport prepare(const InferenceResult &result, const QString &directory,
                                        const ModelConfig &config,
                                        const QJsonObject &historicalConfig = {});
    static bool writePrepared(const PreparedResultExport &result, QString *error = nullptr);
    static bool writeAtomically(const QString &path, const QByteArray &bytes, QString *error = nullptr);
    // Call on the owning thread. False means shutdown has rejected the new job;
    // accepted jobs are never cancelled, including when a detection task stops.
    bool enqueue(ResultWriteRequest request);
    int pendingCount() const { return pendingCount_; }
    bool isShuttingDown() const { return shuttingDown_; }
    void requestShutdown();

  signals:
    void completed(const vision::app::ResultWriteOutcome &outcome);
    void pendingChanged(int count);
    void shutdownReady();

  private:
    struct QueueState;
    static ResultWriteOutcome write(const ResultWriteRequest &request);
    void deliver(const ResultWriteOutcome &outcome);
    std::shared_ptr<QueueState> queue_;
    int pendingCount_ = 0;
    bool shuttingDown_ = false;
    bool shutdownReadyEmitted_ = false;
};
} // namespace vision::app
Q_DECLARE_METATYPE(vision::app::ResultWriteOutcome)
