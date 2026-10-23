#pragma once

#include "../core/visiontypes.h"
#include <QJsonObject>
#include <QPointer>
#include <QWidget>

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QSplitter;
class QScrollArea;
class QTableWidget;
class QToolButton;
class QTabWidget;
class EvaluationCurve;
class EvaluationPlot;

// The workbench owns model selection and inference. This view only edits an
// explicit evaluation configuration and displays immutable evaluation reports.
class EvaluationPanel final : public QWidget
{
    Q_OBJECT
public:
    explicit EvaluationPanel(QWidget *parent = nullptr);
    ~EvaluationPanel() override;
    QWidget *controlsWidget() const { return controls_; }
    void chooseDataset();
    void chooseDatasetFile();
    void setDatasetSource(const QString &source, const QString &split, const QString &metadata = {});
    void showDatasetImportError(const QString &message);
    void setDataset(const QJsonObject &manifest);
    void clearDataset();
    void setModel(const vision::ModelConfig &config);
    void setBusy(bool busy);
    void setReport(const QJsonObject &report, const QString &directory = {});
    void clearReport();
    void setMode(bool evaluation);
    void setModeControlVisible(bool visible);
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
    int batchSize() const;
    int cpuThreads() const;
    double scoreFloor() const;
    int maxDetections() const;
    QJsonObject analysisOptions() const;
    QJsonObject exportOptions() const;
    QString datasetSplit() const;
    QString datasetYaml() const;
    QString metadataPath() const;
    QJsonObject dataset() const { return manifest_; }
    QJsonObject report() const { return report_; }
    void setAnalysisThreshold(double threshold);
    void setOperatingMetrics(const QJsonObject &metrics);
    QString imageFilter() const;
    void setImageFilter(const QString &filter, bool notify = true);
    void setErrorSummary(int evaluated, int errors, int falsePositive, int falseNegative,
                         int negativeFalsePositive, int collection, bool pending = false,
                         bool available = true, const QString &error = {});
    void exportReportTo(const QString &parentDirectory);
    void hideDatasetSettings();

signals:
    void datasetSelectionStarted();
    void datasetRequested(const QString &yaml, const QString &split, const QString &metadata);
    void modeChanged(bool evaluation);
    void configChanged();
    void thresholdChanged(double threshold);
    void applyRequested(double threshold);
    void imageFilterChanged(const QString &filter);
    void reportOpenRequested(const QString &path);
    void reanalysisRequested(const QJsonObject &options);
    void sceneReviewRequested(const QString &group);
    void errorReviewRequested();
    void externalReportImportRequested(const QString &directory, const QJsonObject &options);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void updateEligibility();
    void updateMapping();
    void updateMetrics();
    void updateRecommendation();
    void updateDecisionMinimums();
    void updateAnalysisEligibility();
    void updateScenes();
    void requestDataset();
    void showDatasetSettings();
    void positionDatasetSettings();
    QJsonObject selectedRow() const;
    void exportReport();

    QPointer<QWidget> controls_;
    QWidget *datasetControls_ = nullptr;
    QPointer<QWidget> datasetSettings_;
    QWidget *datasetSettingsContent_ = nullptr;
    QScrollArea *datasetSettingsScroll_ = nullptr;
    QPointer<QWidget> datasetSettingsHost_;
    QPointer<QToolButton> datasetSettingsButton_, eligibilityStatus_;
    QComboBox *mode_ = nullptr, *split_ = nullptr, *filter_ = nullptr, *strategy_ = nullptr;
    QSpinBox *modelId_ = nullptr, *datasetId_ = nullptr;
    QDoubleSpinBox *threshold_ = nullptr, *matchIou_ = nullptr, *scanStart_ = nullptr,
                   *scanEnd_ = nullptr, *scanStep_ = nullptr, *constraint_ = nullptr;
    QLineEdit *metadata_ = nullptr;
    QSpinBox *batch_ = nullptr, *threads_ = nullptr, *maximumDetections_ = nullptr, *figureDpi_ = nullptr;
    QDoubleSpinBox *scoreFloor_ = nullptr, *reanalyzeIou_ = nullptr, *reanalyzeStart_ = nullptr,
                   *reanalyzeEnd_ = nullptr, *reanalyzeStep_ = nullptr, *reanalyzeConfidence_ = nullptr;
    QCheckBox *figurePng_ = nullptr, *figureSvg_ = nullptr, *figurePdf_ = nullptr;
    QComboBox *plotKind_ = nullptr;
    QTabWidget *analysisTabs_ = nullptr;
    QSplitter *decisionSplit_ = nullptr;
    QLabel *modeTitle_ = nullptr, *datasetHint_ = nullptr, *mappingHint_ = nullptr, *validation_ = nullptr,
           *reportStatus_ = nullptr, *reportIdentity_ = nullptr, *recommendation_ = nullptr,
           *constraintTitle_ = nullptr, *diagnosticThreshold_ = nullptr, *exportStatus_ = nullptr,
           *counts_ = nullptr, *errorSummary_ = nullptr, *analysisValidation_ = nullptr, *sceneHint_ = nullptr;
    QList<QLabel *> metrics_;
    QPushButton *reload_ = nullptr, *metadataBrowse_ = nullptr, *apply_ = nullptr,
                *useRecommended_ = nullptr, *reviewErrors_ = nullptr, *openFolder_ = nullptr, *openReport_ = nullptr,
                *export_ = nullptr, *reanalyze_ = nullptr, *sceneReanalysis_ = nullptr, *reviewScene_ = nullptr, *importExternal_ = nullptr;
    EvaluationCurve *curve_ = nullptr;
    EvaluationPlot *detailPlot_ = nullptr;
    QTableWidget *table_ = nullptr, *details_ = nullptr, *scenes_ = nullptr,
                 *localizationMatrix_ = nullptr, *presenceMatrix_ = nullptr;
    QJsonObject manifest_, report_, operatingOverride_;
    vision::ModelConfig model_;
    QString yaml_, directory_, importError_;
    bool busy_ = false, updating_ = false, exporting_ = false;
    double recommended_ = -1;
};
