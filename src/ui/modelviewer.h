#pragma once

#include <QByteArray>
#include <QJsonObject>
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
class QLineEdit;
class QTreeView;
class QTableView;
class QStandardItemModel;
class QSortFilterProxyModel;

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
    enum class DisplayMode
    {
        Graph,
        Hierarchy,
        Parameters
    };
    Q_ENUM(DisplayMode)

    explicit ModelViewer(QWidget *parent = nullptr);
    ~ModelViewer() override;

    void openModel(const QString &path);
    void stop();
    QString modelPath() const;
    State state() const;
    QString errorString() const;
    int graphNodeCount() const;
    void setDisplayMode(DisplayMode mode);
    DisplayMode displayMode() const;
    int hierarchyItemCount() const;
    int parameterCount() const;

  signals:
    void modelLoaded(QString path);
    void loadFailed(QString error);
    void stateChanged(ModelViewer::State state);
    void displayModeChanged(ModelViewer::DisplayMode mode);

  private:
    void setState(State state, const QString &message);
    void fail(const QString &message);
    void releaseResources();
    void collectOutput(quint64 generation);
    void startBrowser(const QUrl &url, quint64 generation);
    void keepBrowserRendering();
    void checkGraph();
    void extractStructure(quint64 generation);
    void applyStructure(const QJsonObject &structure);
    void clearStructure();
    void updateDisplay();
    void updateSelectionDetails(bool hierarchy);

    State m_state = State::Empty;
    QString m_modelPath;
    QString m_canonicalPath;
    QString m_error;
    QString m_hierarchyHint;
    quint64 m_generation = 0;
    int m_graphNodeCount = 0;
    DisplayMode m_mode = DisplayMode::Graph;
    int m_hierarchyCount = 0, m_parameterCount = 0;
    bool m_structurePending = false, m_structureReady = false;
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
    QWidget *m_hierarchyPage = nullptr, *m_parametersPage = nullptr;
    QTreeView *m_tree = nullptr;
    QTableView *m_parameters = nullptr;
    QStandardItemModel *m_treeModel = nullptr, *m_parameterModel = nullptr;
    QSortFilterProxyModel *m_treeFilter = nullptr, *m_parameterFilter = nullptr;
    QLineEdit *m_search = nullptr;
    QLabel *m_summary = nullptr, *m_hint = nullptr, *m_details = nullptr;
    QList<QPushButton *> m_modeButtons;
};

Q_DECLARE_METATYPE(ModelViewer::State)
Q_DECLARE_METATYPE(ModelViewer::DisplayMode)
