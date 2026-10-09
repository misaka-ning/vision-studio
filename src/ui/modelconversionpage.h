#pragma once

#include <QWidget>
#include <QJsonObject>

class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProcess;
class QProgressBar;
class QPushButton;
class QSpinBox;

class ModelConversionPage final : public QWidget
{
    Q_OBJECT
  public:
    explicit ModelConversionPage(const QString &dataRoot, QWidget *parent = nullptr);
    ~ModelConversionPage() override;
    void setSourceModel(const QString &path, const QString &displayName = {});
    bool isBusy() const;
    QJsonObject completedMetadata() const { return completedMetadata_; }
    void cancel();

  signals:
    void chooseLibraryRequested();
    void convertedModelReady(const QString &path);
    void busyChanged(bool busy);

  private:
    void updateSource();
    void updateDestination(bool force = false);
    void updateControls();
    void start();
    void readOutput();
    void appendLog(const QString &message);
    void complete(int exitCode, bool normalExit);
    void cleanupJob();
    bool openFolder(const QString &path);

    QString source_, displayName_, dataRoot_, output_, completedOutput_, runningSource_, jobTemp_;
    QByteArray pendingOutput_, stderrOutput_;
    QJsonObject completedMetadata_;
    QProcess *process_ = nullptr;
    bool cancelling_ = false, receivedResult_ = false, protocolError_ = false;
    quint64 jobTempInode_ = 0, jobTempDevice_ = 0;
    QLabel *sourceName_ = nullptr, *sourcePath_ = nullptr, *supportHint_ = nullptr,
           *status_ = nullptr, *summary_ = nullptr;
    QComboBox *format_ = nullptr, *opset_ = nullptr;
    QSpinBox *imageSize_ = nullptr;
    QLineEdit *directory_ = nullptr, *filename_ = nullptr;
    QPushButton *chooseSource_ = nullptr, *chooseDirectory_ = nullptr, *start_ = nullptr,
                *cancel_ = nullptr, *openOutput_ = nullptr, *addModel_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QPlainTextEdit *log_ = nullptr;
};
