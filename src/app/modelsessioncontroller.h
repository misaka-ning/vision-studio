#pragma once

#include "core/visiontypes.h"
#include <QHash>
#include <QJsonObject>
#include <QObject>
#include <QVariantMap>
#include <atomic>
#include <functional>
#include <memory>

class QThread;

namespace vision::app
{
struct ModelSessionSnapshot
{
    QStringList models;
    QString activeModel;
    QVariantMap displayNames, notes, convertedProfiles, evaluationProfiles;
    QStringList nativeLabels;
    int inputChannels = 0;
    int classCount = 0;
    QString inspectedIdentity, inspectionError;
    bool inspectionPending = false;
    quint64 inspectionGeneration = 0;
    ModelTask suggestedTask = ModelTask::YoloV8;
    bool hasSuggestedTask = false;
};

// The single authority for library entries, global model selection and inspected
// metadata. Widgets apply suggested configuration; no widget or runtime UI is used here.
class ModelSessionController final : public QObject
{
    Q_OBJECT
  public:
    using Inspector = std::function<ModelConfig(const ModelConfig &, std::function<bool()>)>;
    explicit ModelSessionController(QObject *parent = nullptr);
    explicit ModelSessionController(Inspector inspector, QObject *parent = nullptr);
    ~ModelSessionController() override;
    void initialize(const ModelSessionSnapshot &state);
    ModelSessionSnapshot snapshot() const { return state_; }
    bool validateImport(const QStringList &input, QStringList *paths, QString *error = nullptr) const;
    bool importOrdered(const QStringList &input, int *added = nullptr, QString *error = nullptr);
    bool reorder(const QStringList &paths, QString *error = nullptr);
    bool rename(const QString &path, const QString &name, QString *error = nullptr);
    bool setNote(const QString &path, const QString &note, QString *error = nullptr);
    bool remove(const QString &path, QString *error = nullptr);
    bool select(const QString &path, bool force = false, QString *error = nullptr);
    QString displayName(const QString &path) const;
    bool rememberConverted(const QString &path, const QJsonObject &metadata, QString *error = nullptr);
    QJsonObject convertedProfile(const QString &path) const;
    void setEvaluationProfile(const QString &path, const QVariantMap &profile);
    // Stored records remain opaque until convertedProfile validates their file
    // identity and schema; this also preserves old, invalidated associations.
    void setConvertedProfileRecord(const QString &path, const QVariantMap &profile);
    void addKnownModel(const QString &path);
    void acceptInferenceClassCount(int count);
    void inspect(const ModelConfig &config);
    void cancelInspection();
    // An actual loaded backend may identify a file replaced on disk afterwards.
    // Preserve that captured identity instead of relabelling the loaded model.
    bool acceptLoadedMetadata(const ModelConfig &config, const QString &capturedIdentity = {});
    void requestShutdown();
    bool isShutdownReady() const { return shuttingDown_ && inspections_.isEmpty(); }
    int activeInspectionCount() const { return inspections_.size(); }
    static QString fileIdentity(const QString &path);
    static bool isPtModel(const QString &path);
    static QStringList convertedLabels(const QJsonObject &metadata);
    static bool validConvertedMetadata(const QString &path, const QJsonObject &metadata);

  signals:
    void changed();
    void selectionChanged(const QString &path);
    void metadataChanged();
    // The same path changed while inspection ran; ask the owner for a fresh
    // configuration rather than retrying with an obsolete widget snapshot.
    void inspectionRetryRequested();
    void shutdownReady();

  private:
    bool editable(QString *error) const;
    bool contains(const QString &path, QString *error) const;
    void resetMetadata(bool advanceGeneration = false);
    void notifyShutdownIfReady();
    ModelSessionSnapshot state_;
    Inspector inspector_;
    QHash<QThread *, std::shared_ptr<std::atomic_bool>> inspections_;
    std::shared_ptr<std::atomic_bool> currentCancellation_;
    bool shuttingDown_ = false;
    bool shutdownReadyEmitted_ = false;
};
} // namespace vision::app
Q_DECLARE_METATYPE(vision::app::ModelSessionSnapshot)
