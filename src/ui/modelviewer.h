#pragma once

#include <QByteArray>
#include <QMetaType>
#include <QPointer>
#include <QString>
#include <QUrl>
#include <QWidget>

class QLabel;
class QPushButton;
class QProgressBar;
class QProcess;
class QStackedWidget;
class QTimer;
class QWebEngineProfile;
class QWebEngineView;

// Browser and Python service are deliberately created only by openModel().
class ModelViewer : public QWidget
{
    Q_OBJECT
  public:
    enum class State
    {
        Empty,
        Loading,
        Ready,
        Error
    };
    Q_ENUM(State)

    explicit ModelViewer(QWidget *parent = nullptr);
    ~ModelViewer() override;

    void openModel(const QString &path);
    void stop();
    QString modelPath() const;
    State state() const;
    QString errorString() const;
    int graphNodeCount() const;

  signals:
    void modelLoaded(QString path);
    void loadFailed(QString error);
    void stateChanged(ModelViewer::State state);

  private:
    void setState(State state, const QString &message);
    void fail(const QString &message);
    void releaseResources();
    void collectOutput(quint64 generation);
    void startBrowser(const QUrl &url, quint64 generation);
    void keepBrowserRendering();
    void checkGraph();

    State m_state = State::Empty;
    QString m_modelPath;
    QString m_canonicalPath;
    QString m_error;
    quint64 m_generation = 0;
    int m_graphNodeCount = 0;
    bool m_serverReady = false;
    bool m_checkPending = false;
    QByteArray m_stdout;
    QByteArray m_stderr;
    QPointer<QProcess> m_process;
    QStackedWidget *m_stack = nullptr;
    QWidget *m_statusPage = nullptr;
    QLabel *m_title = nullptr;
    QLabel *m_status = nullptr;
    QPushButton *m_retry = nullptr;
    QProgressBar *m_progress = nullptr;
    QTimer *m_deadline = nullptr;
    QTimer *m_poll = nullptr;
    QWebEngineProfile *m_profile = nullptr;
    QWebEngineView *m_webView = nullptr;
};

Q_DECLARE_METATYPE(ModelViewer::State)
