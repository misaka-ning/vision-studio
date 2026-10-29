#pragma once

#include "core/evaluationreview.h"
#include "core/evaluationworker.h"
#include <QHash>
#include <QObject>
#include <QVariantMap>
#include <atomic>
#include <memory>

namespace vision::app
{
enum class EvaluationOperation { Idle, Evaluation, Reanalysis, ExternalImport };
struct EvaluationReviewState
{
    bool active = false, pending = false, available = false, previewPending = false;
    QString filter = "all", error;
    double threshold = .25;
    QVector<int> rows, errorRows;
    int evaluated = 0, falsePositive = 0, falseNegative = 0, negativeFalsePositive = 0;
    quint64 filterGeneration = 0, previewGeneration = 0;
};
struct EvaluationState
{
    QJsonObject report;
    QString directory, runFingerprint;
    ModelConfig runConfig;
    QHash<QString, int> sampleIndices;
    EvaluationOperation operation = EvaluationOperation::Idle;
    int completed = 0, total = 0;
    QStringList taskFiles;
    int rangeFirst = 0, rangeLast = 0;
    QString stage;
    EvaluationReviewState review;
};
struct LoadedEvaluationReport
{
    bool valid = false;
    QString error, directory;
    QJsonObject report, manifest;
};
struct EvaluationFilterOutcome
{
    bool valid = false;
    bool selectFirst = false;
    QString error, directory;
    QStringList files;
    QVector<bool> visible;
    int selectedRow = -1;
    quint64 generation = 0;
};
struct EvaluationPreviewOutcome
{
    int row = -1;
    QString source, directory;
    quint64 generation = 0;
    EvaluationPreview preview;
    ModelConfig config;
};
struct EvaluationConfigurationPatch
{
    bool valid = false;
    QString error;
    ModelConfig config;
    int classId = 0;
    QVariantMap profile;
};

// Owns evaluation/reanalysis/import workers and immutable report/review state.
// Dataset import remains exclusively owned by InputSessionController.
class EvaluationController final : public QObject
{
    Q_OBJECT
  public:
    explicit EvaluationController(QObject *parent = nullptr);
    ~EvaluationController() override;
    EvaluationState snapshot() const { return state_; }
    bool isBusy() const { return worker_ != nullptr; }
    bool start(const EvaluationRequest &request, const QString &fingerprint, QString *error = nullptr);
    bool reanalyze(const QJsonObject &options, const QString &outputRoot, const QString &projectRoot,
                   QString *error = nullptr);
    bool importExternal(const QString &source, const QJsonObject &options, const QString &outputRoot,
                        const QString &projectRoot, QString *error = nullptr);
    void requestStop();
    static LoadedEvaluationReport readReport(const QString &directory);
    void acceptReport(const QJsonObject &report, const QString &directory, bool historical = false);
    void clear();
    // Read-only snapshots of input authority; changes cancel obsolete callbacks.
    void setReviewContext(const QStringList &files, const QHash<QString, QJsonObject> &samples,
                          const QJsonObject &manifest = {});
    void setReviewActive(bool active);
    void resetReview();
    void suspendReview();
    bool invalidateForFingerprint(const QString &fingerprint);
    bool applyFilter(const QString &filter, double threshold, int currentRow,
                     bool selectFirst = false, QString *error = nullptr);
    bool requestPreview(int row);
    int navigationTarget(int currentRow, int step) const;
    EvaluationConfigurationPatch configurationPatch(const QString &activeModel, double threshold,
                                                     const QVariantMap &previousProfile = {});
    QJsonObject historyEntry(const QString &modelDisplayName) const;
    QVariantMap reportAssociation(const QVariantMap &previousProfile) const;
    static ModelConfig configFromJson(const QJsonObject &value);
    static QString configurationFingerprint(const EvaluationRequest &request);
    static void attachGroundTruth(InferenceResult &result, const QJsonObject &manifest,
                                  const QHash<QString, QJsonObject> &samples, int modelClassId,
                                  int datasetClassId, int modelClassCount, double matchIou);
    void requestShutdown();
    bool isShutdownReady() const { return shuttingDown_ && !worker_ && reviewThreads_.isEmpty(); }
    int activeReviewThreadCount() const { return reviewThreads_.size(); }

  signals:
    void busyChanged(bool active);
    void progress(int completed, int total, const QString &stage);
    void failed(const QString &message);
    // Ordinary/reanalysis reports are already accepted before this signal.
    // External imports are announced only after completion through externalImportReady.
    void reportReady(const QJsonObject &report, const QString &directory);
    void externalImportReady(const QString &directory);
    void finished();
    void reportChanged();
    void reviewChanged();
    void filterReady(const vision::app::EvaluationFilterOutcome &outcome);
    void previewReady(const vision::app::EvaluationPreviewOutcome &outcome);
    void shutdownReady();

  private:
    bool mayStart(QString *error) const;
    void startWorker(EvaluationWorker *worker, EvaluationOperation operation);
    void cancelReviewWork();
    void notifyShutdownIfReady();
    EvaluationState state_;
    EvaluationWorker *worker_ = nullptr;
    QHash<QThread *, std::shared_ptr<std::atomic_bool>> reviewThreads_;
    std::shared_ptr<std::atomic_bool> filterCancellation_, previewCancellation_;
    QStringList contextFiles_;
    QHash<QString, QJsonObject> contextSamples_;
    QJsonObject contextManifest_;
    bool shuttingDown_ = false, shutdownReadyEmitted_ = false;
};
} // namespace vision::app
Q_DECLARE_METATYPE(vision::app::EvaluationFilterOutcome)
Q_DECLARE_METATYPE(vision::app::EvaluationPreviewOutcome)
