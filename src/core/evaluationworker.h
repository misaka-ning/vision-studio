#pragma once

#include "visiontypes.h"
#include <QJsonObject>
#include <QThread>
#include <atomic>

namespace vision
{
// A task owns its model configuration, ordered file set and dataset snapshot.
// Configure only while the thread is stopped, then use QThread::start().
struct EvaluationRequest
{
    ModelConfig config;
    QJsonObject manifest;
    QStringList files;
    int modelClassId = 0;
    int datasetClassId = 0;
    double confidence = .25;
    double scoreFloor = .001;
    double matchIou = .5;
    double scanStart = .05;
    double scanEnd = .95;
    double scanStep = .01;
    QString outputRoot;
    QString projectRoot;
};

class EvaluationWorker final : public QThread
{
    Q_OBJECT
  public:
    explicit EvaluationWorker(QObject *parent = nullptr);
    void configureDataset(const QString &yaml, const QString &split, const QString &metadata,
                          const QString &projectRoot);
    void configureEvaluation(const EvaluationRequest &request);
    void requestStop() noexcept;

    static QJsonObject modelConfigJson(const ModelConfig &config);
    static QJsonObject readSample(const QString &directory, int index, QString *error = nullptr);

  signals:
    void datasetReady(const QJsonObject &manifest);
    void reportReady(const QJsonObject &report, const QString &directory);
    void progress(int completed, int total, const QString &stage);
    void failed(const QString &message);

  protected:
    void run() override;

  private:
    enum class Operation { Dataset, Evaluation };
    Operation operation_ = Operation::Dataset;
    QString yaml_, split_, metadata_, projectRoot_;
    EvaluationRequest request_;
    std::atomic_bool stop_{false};
};
} // namespace vision

Q_DECLARE_METATYPE(vision::EvaluationRequest)
