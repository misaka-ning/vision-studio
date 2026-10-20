#pragma once

#include "../core/visiontypes.h"
#include <QJsonObject>
#include <QPointer>
#include <QWidget>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTableWidget;
class EvaluationCurve;

// The workbench owns model selection and inference. This view only edits an
// explicit evaluation configuration and displays immutable evaluation reports.
class EvaluationPanel final : public QWidget
{
    Q_OBJECT
public:
    explicit EvaluationPanel(QWidget *parent = nullptr);
    QWidget *controlsWidget() const { return controls_; }
    void chooseDataset();
    void setDataset(const QJsonObject &manifest);
    void clearDataset();
    void setModel(const vision::ModelConfig &config);
    void setBusy(bool busy);
    void setReport(const QJsonObject &report, const QString &directory = {});
    void clearReport();
    void setMode(bool evaluation);
    bool evaluationMode() const;
    bool canEvaluate() const;
    QString validationMessage() const;
    int modelClassId() const;
    int datasetClassId() const;
    double analysisThreshold() const;
    double matchIou() const;
    double scanStart() const;
    double scanEnd() const;
    double scanStep() const;
    QString datasetSplit() const;
    QString datasetYaml() const;
    QString metadataPath() const;
    QJsonObject dataset() const { return manifest_; }
    QJsonObject report() const { return report_; }
    void setAnalysisThreshold(double threshold);
    void setOperatingMetrics(const QJsonObject &metrics);
    void exportReportTo(const QString &parentDirectory);

signals:
    void datasetRequested(const QString &yaml, const QString &split, const QString &metadata);
    void modeChanged(bool evaluation);
    void configChanged();
    void thresholdChanged(double threshold);
    void applyRequested(double threshold);
    void imageFilterChanged(const QString &filter);
    void reportOpenRequested(const QString &path);

private:
    void updateEligibility();
    void updateMapping();
    void updateMetrics();
    void updateRecommendation();
    void requestDataset();
    QJsonObject selectedRow() const;
    void exportReport();

    QPointer<QWidget> controls_;
    QWidget *datasetControls_ = nullptr;
    QComboBox *mode_ = nullptr, *split_ = nullptr, *filter_ = nullptr, *strategy_ = nullptr;
    QSpinBox *modelId_ = nullptr, *datasetId_ = nullptr;
    QDoubleSpinBox *threshold_ = nullptr, *matchIou_ = nullptr, *scanStart_ = nullptr,
                   *scanEnd_ = nullptr, *scanStep_ = nullptr, *constraint_ = nullptr;
    QLineEdit *metadata_ = nullptr;
    QLabel *datasetHint_ = nullptr, *mappingHint_ = nullptr, *validation_ = nullptr,
           *reportStatus_ = nullptr, *reportIdentity_ = nullptr, *recommendation_ = nullptr,
           *counts_ = nullptr;
    QList<QLabel *> metrics_;
    QPushButton *reload_ = nullptr, *metadataBrowse_ = nullptr, *apply_ = nullptr,
                *useRecommended_ = nullptr, *openFolder_ = nullptr, *openReport_ = nullptr,
                *export_ = nullptr;
    EvaluationCurve *curve_ = nullptr;
    QTableWidget *table_ = nullptr;
    QJsonObject manifest_, report_, operatingOverride_;
    vision::ModelConfig model_;
    QString yaml_, directory_;
    bool busy_ = false, updating_ = false, exporting_ = false;
    double recommended_ = -1;
};
