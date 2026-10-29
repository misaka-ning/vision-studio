// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "core/datasetsessionstore.h"
#include "core/visiontypes.h"
#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <atomic>
#include <functional>
#include <memory>

class QThread;
class QTimer;
namespace vision { class EvaluationWorker; }

namespace vision::app
{
enum class ImageRange { All, FromCurrent, Current };
enum class InputPhase { Idle, CheckingDataset, CheckingImages, Projecting, WaitingForProjection };

struct InputSessionSnapshot
{
    SourceKind sourceKind = SourceKind::Images;
    QStringList files;
    QString streamPath;
    int selectedRow = -1;
    bool hasSelection = false;
    QJsonObject datasetManifest;
    QHash<QString, QJsonObject> datasetSamples;
    InputPhase phase = InputPhase::Idle;
    bool importing = false;
    bool queuePopulating = false;
    quint64 importGeneration = 0;
    quint64 projectionGeneration = 0;
    quint64 restoreGeneration = 0;
    bool restoreRequested = false;
    bool userInteracted = false;
};

// Owns the input session independently of widgets. Queue signals describe a
// projection of this state; the consumer must acknowledge each appended batch.
// Only the last acknowledgement commits the remembered dataset on disk.
class InputSessionController final : public QObject
{
    Q_OBJECT
  public:
    explicit InputSessionController(QString projectRoot, QString sessionFilePath,
                                    QObject *parent = nullptr);
    ~InputSessionController() override;
    InputSessionSnapshot snapshot() const { return state_; }
    bool loadDataset(const QString &source, const QString &split = QStringLiteral("val"),
                     const QString &metadata = {});
    bool requestImages(const QStringList &paths, bool append = false);
    // A validated historical report may project its stored manifest. This never
    // replaces the last dataset source remembered for ordinary startup.
    bool openDatasetSnapshot(const QJsonObject &manifest);
    void setSelection(int row, bool selected = true);
    bool selectVideo(const QString &path);
    bool selectCamera();
    void clear();
    void restore(const QString &sessionPath = {});
    void noteUserInteraction();
    void setRestoreEnabled(bool enabled);
    void setRestoreGuard(std::function<bool()> guard);
    void cancel();
    void acknowledgeQueueProjection(quint64 generation, int projectedCount);
    QStringList range(ImageRange mode, bool forEvaluation = false, QString *error = nullptr) const;
    void requestShutdown();
    bool isShutdownReady() const { return shuttingDown_ && workers_.isEmpty(); }
    int activeWorkerCount() const { return workers_.size(); }
    static constexpr int MaximumBatchItems = 96;
    static constexpr int BatchBudgetMilliseconds = 5;

  signals:
    void datasetReady(const QJsonObject &manifest);
    void stateChanged();
    void queueReset(const QStringList &files, quint64 generation);
    void queueAppend(const QStringList &paths, int firstRow, quint64 generation);
    void selectionChanged(int row, bool selected);
    void progress(int completed, int total, const QString &stage);
    void notice(const QString &message);
    void failed(const QString &message);
    void finished(bool cancelled);
    void shutdownReady();

  private:
    bool beginDataset(const QString &source, const QString &split, const QString &metadata);
    bool beginOperation(InputPhase phase);
    void beginProjection(const QStringList &images, const QJsonObject &manifest,
                         bool append, const DatasetSession &session = {});
    void appendBatch();
    void completeProjection();
    void failOperation(const QString &message);
    void cancelOperation(bool notify);
    void restorePreviousState();
    void updatePhase(InputPhase phase);
    void resetProjection();
    void notifyShutdownIfReady();
    void stopWorkers();
    QString projectRoot_, sessionFilePath_;
    InputSessionSnapshot state_, previous_;
    QHash<QThread *, std::shared_ptr<std::atomic_bool>> workers_;
    std::shared_ptr<std::atomic_bool> cancellation_;
    EvaluationWorker *datasetWorker_ = nullptr;
    QTimer *populationTimer_ = nullptr;
    QStringList pendingImages_;
    QJsonArray pendingSamples_;
    DatasetSession pendingSession_;
    int pendingIndex_ = 0, pendingTotal_ = 0, firstNewRow_ = 0, expectedProjectionCount_ = -1;
    bool transactionActive_ = false, datasetProjection_ = false, awaitingProjection_ = false;
    bool projectionStarted_ = false, appendProjection_ = false;
    bool restoreEnabled_ = true, shuttingDown_ = false, shutdownReadyEmitted_ = false;
    std::function<bool()> restoreGuard_;
};
} // namespace vision::app
Q_DECLARE_METATYPE(vision::app::InputSessionSnapshot)
Q_DECLARE_METATYPE(vision::app::ImageRange)
