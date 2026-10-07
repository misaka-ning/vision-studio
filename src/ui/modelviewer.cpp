#include "modelviewer.h"

#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWebEngineDownloadRequest>
#include <QWebEnginePage>
#include <QWebEngineProfile>
#include <QWebEngineScript>
#include <QWebEngineScriptCollection>
#include <QWebEngineSettings>
#include <QWebEngineUrlRequestInfo>
#include <QWebEngineUrlRequestInterceptor>
#include <QWebEngineView>

namespace
{
constexpr qsizetype outputLimit = 64 * 1024;
constexpr auto netronVersion = "9.3.1";

QString projectDirectory()
{
#ifdef VISION_PROJECT_DIR
    return QString::fromUtf8(VISION_PROJECT_DIR);
#else
    return QDir::currentPath();
#endif
}

QString helperPath()
{
    const QString overridePath = qEnvironmentVariable("VISION_STUDIO_NETRON_WORKER");
    if (!overridePath.isEmpty())
        return QFileInfo(overridePath).absoluteFilePath();
    const QString applicationPath = QCoreApplication::applicationDirPath();
    const QString home = qEnvironmentVariable("VISION_STUDIO_HOME");
    const QStringList candidates = {home.isEmpty() ? QString() : home + "/scripts/netron_server.py",
                                    applicationPath + "/../../scripts/netron_server.py",
                                    applicationPath + "/../scripts/netron_server.py",
                                    projectDirectory() + "/scripts/netron_server.py"};
    for (const QString &path : candidates)
        if (!path.isEmpty() && QFileInfo(path).isFile())
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    return {};
}

QString interpreterPath(const QString &script)
{
    const QString overridePath = qEnvironmentVariable("VISION_STUDIO_PYTHON");
    if (!overridePath.isEmpty())
        return QFileInfo(overridePath).isAbsolute() ? overridePath
                                                    : QStandardPaths::findExecutable(overridePath);
    const QString home = qEnvironmentVariable("VISION_STUDIO_HOME");
    const QStringList candidates = {home.isEmpty() ? QString() : home + "/runtime/bin/python",
                                    QFileInfo(script).absolutePath() + "/../runtime/bin/python",
                                    projectDirectory() + "/runtime/bin/python",
                                    QDir::homePath() + "/VisionStudio/runtime/bin/python"};
    for (const QString &path : candidates)
        if (!path.isEmpty() && QFileInfo(path).isFile() && QFileInfo(path).isExecutable())
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    return {};
}

bool sameOrigin(const QUrl &url, const QUrl &origin)
{
    return url.scheme() == "http" && url.host() == "127.0.0.1" && url.port() == origin.port() &&
           url.userName().isEmpty() && url.password().isEmpty();
}

class LocalRequests final : public QWebEngineUrlRequestInterceptor
{
  public:
    explicit LocalRequests(const QUrl &origin, QObject *parent)
        : QWebEngineUrlRequestInterceptor(parent), m_origin(origin)
    {
    }

    void interceptRequest(QWebEngineUrlRequestInfo &info) override
    {
        const QUrl url = info.requestUrl();
        // Blob workers inherit this service's origin. No file or external network access is allowed.
        const bool blob = url.scheme() == "blob" && sameOrigin(QUrl(url.path()), m_origin);
        const bool local = sameOrigin(url, m_origin) || url == QUrl("about:blank") || blob ||
                           (url.scheme() == "data" &&
                            info.resourceType() != QWebEngineUrlRequestInfo::ResourceTypeMainFrame);
        if (!local)
            info.block(true);
    }

  private:
    const QUrl m_origin;
};

class LocalPage final : public QWebEnginePage
{
  public:
    LocalPage(QWebEngineProfile *profile, const QUrl &origin, QObject *parent)
        : QWebEnginePage(profile, parent), m_origin(origin)
    {
    }

  protected:
    bool acceptNavigationRequest(const QUrl &url, NavigationType, bool) override
    {
        return sameOrigin(url, m_origin) || url == QUrl("about:blank");
    }

    QWebEnginePage *createWindow(WebWindowType) override
    {
        return nullptr;
    }

    QStringList chooseFiles(FileSelectionMode, const QStringList &, const QStringList &) override
    {
        return {};
    }

  private:
    const QUrl m_origin;
};

QString bridgeScript()
{
    // Adapt Netron's supported browser Host rather than editing vendor files or accepting tracking cookies.
    // Version is pinned: these hooks are verified against the shipped Netron 9.3.1 browser/view modules.
    return QString::fromUtf8(R"JS((() => {
        'use strict';
        // Qt can keep Page::visible true while its native QWidget is hidden. In that state
        // Chromium supplies timers but no compositor animation frames. Race the native clock
        // with a timer so Netron's SVG measurement can finish in a background application page.
        // Visible pages retain the native frame timing; each request is delivered exactly once.
        const requestFrame = window.requestAnimationFrame.bind(window);
        const cancelFrame = window.cancelAnimationFrame.bind(window);
        const frames = new Map();
        let nextFrame = 1;
        window.requestAnimationFrame = callback => {
            if (typeof callback !== 'function') return requestFrame(callback);
            const id = nextFrame++;
            const entry = { native: 0, timer: 0 };
            const complete = timestamp => {
                if (!frames.has(id)) return;
                frames.delete(id);
                window.clearTimeout(entry.timer);
                cancelFrame(entry.native);
                callback(timestamp);
            };
            frames.set(id, entry);
            entry.native = requestFrame(complete);
            entry.timer = window.setTimeout(() => complete(window.performance.now()), 50);
            return id;
        };
        window.cancelAnimationFrame = id => {
            const entry = frames.get(id);
            if (entry) {
                frames.delete(id);
                window.clearTimeout(entry.timer);
                cancelFrame(entry.native);
            }
        };
        const bridge = {
            parsed: false, error: '', hostAdapted: false,
            fail(error) {
                if (!this.error) this.error = String(error && error.message ? error.message : error || 'Unknown error');
            },
            snapshot() {
                const view = window.__view__;
                const model = view && view.model;
                const active = view && view.activeTarget;
                const nodes = active && Array.isArray(active.nodes) ? active.nodes.length : 0;
                const rendered = !!(view && view.target && document.querySelector('#target svg'));
                return JSON.stringify({ error: this.error, parsed: this.parsed,
                    ready: !!(this.parsed && model && active && rendered && document.body &&
                              document.body.classList.contains('default')),
                    empty: !!(this.parsed && model && !active),
                    nodes, rendered, format: model && model.format ? String(model.format) : '',
                    hostAdapted: this.hostAdapted });
            }
        };
        Object.defineProperty(window, '__visionStudioNetron', { value: bridge });
        window.addEventListener('error', event => {
            if (event.error || event.message) bridge.fail(event.error || event.message);
        });
        window.addEventListener('unhandledrejection', event => bridge.fail(event.reason));
        const hook = (exports) => {
            if (!exports || exports.__visionStudioHooked) return;
            Object.defineProperty(exports, '__visionStudioHooked', { value: true });
            let browser;
            Object.defineProperty(exports, 'browser', { configurable: true,
                get: () => browser,
                set(namespace) {
                    browser = namespace;
                    if (!namespace || !namespace.Host) return;
                    const Original = namespace.Host;
                    namespace.Host = class extends Original {
                        async view(view) {
                            this._view = view;
                            this._environment.packaged = false;
                            bridge.hostAdapted = true;
                        }
                        async start() {
                            if (this._meta.file && !this._view.accept(this._meta.file[0])) {
                                bridge.fail('Unsupported file format.');
                                return;
                            }
                            try { return await super.start(); }
                            catch (error) { bridge.fail(error); throw error; }
                        }
                        event() {}
                        exception(error, fatal) { if (fatal) bridge.fail(error); }
                        openURL() {}
                    };
                }
            });
            let view;
            Object.defineProperty(exports, 'view', { configurable: true,
                get: () => view,
                set(namespace) {
                    view = namespace;
                    if (!namespace || !namespace.View) return;
                    const prototype = namespace.View.prototype;
                    const open = prototype.open;
                    prototype.open = async function(...args) {
                        try {
                            const model = await open.apply(this, args);
                            if (model) bridge.parsed = true;
                            return model;
                        } catch (error) { bridge.fail(error); throw error; }
                    };
                    const error = prototype.error;
                    prototype.error = function(...args) {
                        bridge.fail(args[0]);
                        return error.apply(this, args);
                    };
                }
            });
            let terminate;
            Object.defineProperty(exports, 'terminate', { configurable: true,
                get: () => terminate,
                set(fn) {
                    terminate = function(message) { bridge.fail(message); return fn.call(this, message); };
                }
            });
        };
        let exports;
        Object.defineProperty(window, 'exports', { configurable: true,
            get: () => exports, set(value) { exports = value; hook(value); }
        });
        document.addEventListener('DOMContentLoaded', () => {
            document.documentElement.style.colorScheme = 'dark';
            const style = document.createElement('style');
            style.textContent = `
                html, body, #target, .default { background: #0b1420 !important; color: #dce6ef; }
                #sidebar, #menu, #toolbar, .sidebar, .menu { background-color: #152330 !important; color: #dce6ef; }
                .sidebar-item, .sidebar-item-name, .sidebar-item-value { color: #dce6ef !important; }
                .sidebar-item-value, .sidebar-item-selector, .sidebar-find-search {
                    background: #1d3143 !important; border-color: #30485c !important; color: #dce6ef !important;
                }
                .sidebar-item-value-content { background: #192c3d !important; border-color: #30485c !important; }
                .sidebar-find-query, .sidebar-find-content li, .sidebar-closebutton { color: #bacbdb; }
                .sidebar-find-content li.focus { background: #294960; color: #fff; }
                .sidebar-documentation a { color: #70c9c6; }
                .menu .menu-command { color: #dce6ef !important; }
                .menu .menu-command:disabled { color: #6c8295 !important; }
                .menu .menu-command:focus { background: #286958; }
                .toolbar-select select, .toolbar-path-name-button, .toolbar-path-back-button {
                    background: #254157; border-color: #365b73; color: #e4edf6;
                }
                .toolbar-icon .stroke { stroke: #b4c9da; }
                .toolbar-icon .fill { fill: #b4c9da; }
                .toolbar-icon .border { stroke: #0b1420; }
                .node path, .node line { stroke: #3b5164; }
                .node-item path { stroke: #3b5164; }
                .node-item:not(.node-item-type) path { fill: #192a39; }
                .node-item text { fill: #dce6ef; }
                .node-item-type text { fill: #fff; }
                .node-item-type-constant path, .node-item-type-control path, .node-item-function path { fill: #23394b; }
                .node-item-type-constant text, .node-item-type-control text, .node-item-function text { fill: #e5eef7; }
                .node-item-type:hover path { fill: #35516a; }
                .node-item-type:hover text { fill: #fff; }
                .node-item-undefined path { fill: #a4323c !important; }
                .node-argument > text, .edge-label { fill: #bacbdb; }
                .node-argument-list > path, .graph-item-input path, .graph-item-output path { fill: #23394b; }
                .node-block > .node-block-background { fill: #0b1420; }
                .node-block .edge-path { stroke: #8aa2b8; }
                .edge-path { stroke: #8aa2b8; }
                .edge-label text { fill: #bacbdb; }
                #arrowhead, #arrowhead-tunnel { fill: #8aa2b8; }
                #logo-github, #logo-netron { display: none !important; }
            `;
            document.head.appendChild(style);
        }, { once: true });
    })();)JS");
}
} // namespace

ModelViewer::ModelViewer(QWidget *parent) : QWidget(parent)
{
    setObjectName("modelViewer");
    qRegisterMetaType<State>();
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    m_stack = new QStackedWidget(this);
    // Netron measures its SVG after three animation frames. Keep its browser visible beneath
    // the native loading panel so Chromium can deliver those frames before State::Ready.
    static_cast<QStackedLayout *>(m_stack->layout())->setStackingMode(QStackedLayout::StackAll);
    layout->addWidget(m_stack);
    m_statusPage = new QWidget(m_stack);
    m_statusPage->setObjectName("modelViewerStatePage");
    m_statusPage->setStyleSheet("#modelViewerStatePage { background: #0b1420; border: 1px solid #253747; "
                                "border-radius: 12px; }");
    auto *statusLayout = new QVBoxLayout(m_statusPage);
    statusLayout->setContentsMargins(36, 40, 36, 40);
    statusLayout->setSpacing(16);
    statusLayout->addStretch();
    m_title = new QLabel(QStringLiteral("查看模型结构"), m_statusPage);
    m_title->setAlignment(Qt::AlignCenter);
    m_title->setStyleSheet("color: #e4edf6; font-size: 22px; font-weight: 600;");
    statusLayout->addWidget(m_title);
    m_status = new QLabel(m_statusPage);
    m_status->setObjectName("modelViewerStatus");
    m_status->setAlignment(Qt::AlignCenter);
    m_status->setWordWrap(true);
    m_status->setTextFormat(Qt::PlainText);
    m_status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_status->setStyleSheet("color: #98afc3; font-size: 14px;");
    statusLayout->addWidget(m_status);
    m_progress = new QProgressBar(m_statusPage);
    m_progress->setRange(0, 0);
    m_progress->setMaximumWidth(300);
    m_progress->setFixedHeight(4);
    m_progress->setTextVisible(false);
    m_progress->setStyleSheet("QProgressBar { background: #213346; border: 0; border-radius: 2px; } "
                              "QProgressBar::chunk { background: #38c9a7; }");
    statusLayout->addWidget(m_progress, 0, Qt::AlignHCenter);
    m_retry = new QPushButton(QStringLiteral("重试加载"), m_statusPage);
    m_retry->setObjectName("modelViewerRetry");
    m_retry->setMinimumWidth(140);
    m_retry->setCursor(Qt::PointingHandCursor);
    statusLayout->addWidget(m_retry, 0, Qt::AlignHCenter);
    statusLayout->addStretch();
    m_stack->addWidget(m_statusPage);
    connect(m_retry, &QPushButton::clicked, this, [this] { openModel(m_modelPath); });
    m_deadline = new QTimer(this);
    m_deadline->setSingleShot(true);
    connect(m_deadline, &QTimer::timeout, this,
            [this] { fail(QStringLiteral("模型结构加载超时。请重试；大型模型可能需要更长的解析时间。")); });
    m_poll = new QTimer(this);
    m_poll->setInterval(200);
    connect(m_poll, &QTimer::timeout, this, &ModelViewer::checkGraph);
    setState(State::Empty, QStringLiteral("在模型库中选择模型，即可查看节点、张量与连接。"));
}

ModelViewer::~ModelViewer()
{
    stop();
    // Retiring processes may still be waiting for their graceful quit timer. Bound shutdown and reap all of
    // them.
    for (QProcess *process : findChildren<QProcess *>(QString(), Qt::FindDirectChildrenOnly))
    {
        if (process->state() != QProcess::NotRunning)
        {
            process->closeWriteChannel();
            if (!process->waitForFinished(150))
            {
                process->terminate();
                if (!process->waitForFinished(150))
                {
                    process->kill();
                    process->waitForFinished(150);
                }
            }
        }
    }
}

QString ModelViewer::modelPath() const
{
    return m_modelPath;
}

ModelViewer::State ModelViewer::state() const
{
    return m_state;
}

QString ModelViewer::errorString() const
{
    return m_error;
}

int ModelViewer::graphNodeCount() const
{
    return m_graphNodeCount;
}

void ModelViewer::setState(State state, const QString &message)
{
    const bool changed = m_state != state;
    m_state = state;
    m_status->setText(message);
    m_title->setText(state == State::Error     ? QStringLiteral("无法显示模型结构")
                     : state == State::Loading ? QStringLiteral("正在打开模型结构")
                                               : QStringLiteral("查看模型结构"));
    m_progress->setVisible(state == State::Loading);
    m_retry->setVisible(state == State::Error && !m_modelPath.isEmpty());
    m_stack->setCurrentWidget(state == State::Ready && m_webView ? static_cast<QWidget *>(m_webView)
                                                                 : m_statusPage);
    if (changed)
        emit stateChanged(state);
}

void ModelViewer::releaseResources()
{
    m_deadline->stop();
    m_poll->stop();
    m_checkPending = false;
    m_serverReady = false;
    m_graphNodeCount = 0;
    m_poll->setInterval(200);
    if (m_webView)
    {
        QWebEngineView *view = m_webView;
        m_webView = nullptr;
        m_stack->removeWidget(view);
        delete view; // Page must die before its off-the-record profile.
    }
    delete m_profile;
    m_profile = nullptr;
    if (m_process)
    {
        QProcess *process = m_process;
        m_process = nullptr;
        process->disconnect(this);
        connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), process,
                &QObject::deleteLater);
        if (process->state() == QProcess::NotRunning)
            process->deleteLater();
        else
        {
            process->write("{\"type\":\"quit\"}\n");
            process->closeWriteChannel();
            QTimer::singleShot(800, process,
                               [process]
                               {
                                   if (process->state() != QProcess::NotRunning)
                                   {
                                       process->terminate();
                                       QTimer::singleShot(400, process,
                                                          [process]
                                                          {
                                                              if (process->state() != QProcess::NotRunning)
                                                                  process->kill();
                                                          });
                                   }
                               });
        }
    }
    m_stdout.clear();
    m_stderr.clear();
}

void ModelViewer::stop()
{
    ++m_generation;
    releaseResources();
    m_error.clear();
    setState(State::Empty, QStringLiteral("在模型库中选择模型，即可查看节点、张量与连接。"));
}

void ModelViewer::fail(const QString &message)
{
    ++m_generation;
    const quint64 generation = m_generation;
    releaseResources();
    m_error = message;
    setState(State::Error, message);
    if (generation == m_generation && m_state == State::Error)
        emit loadFailed(message);
}

void ModelViewer::openModel(const QString &path)
{
    ++m_generation;
    releaseResources();
    m_error.clear();
    m_modelPath = path.isEmpty() ? QString() : QFileInfo(path).absoluteFilePath();
    if (path.isEmpty())
    {
        m_canonicalPath.clear();
        setState(State::Empty, QStringLiteral("在模型库中选择模型，即可查看节点、张量与连接。"));
        return;
    }
    const QFileInfo model(m_modelPath);
    m_canonicalPath = model.canonicalFilePath();
    if (!model.isFile() || !model.isReadable())
    {
        fail(QStringLiteral("找不到可读取的模型文件。请重新选择文件。\n%1").arg(m_modelPath));
        return;
    }
    if (model.size() == 0)
    {
        fail(QStringLiteral("模型文件为空，无法解析结构。"));
        return;
    }
    const QString helper = helperPath();
    if (helper.isEmpty() || !QFileInfo(helper).isFile())
    {
        fail(QStringLiteral("找不到模型结构服务 scripts/netron_server.py。请保留完整的安装目录。"));
        return;
    }
    const QString interpreter = interpreterPath(helper);
    if (interpreter.isEmpty() || !QFileInfo(interpreter).isFile() || !QFileInfo(interpreter).isExecutable())
    {
        fail(QStringLiteral("找不到模型结构运行环境。请安装随附的 Python / Netron，或设置 "
                            "VISION_STUDIO_PYTHON。"));
        return;
    }
    const quint64 generation = m_generation;
    setState(State::Loading, QStringLiteral("正在启动本机模型解析服务…"));
    if (generation != m_generation)
        return;
    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::SeparateChannels);
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    environment.insert("PYTHONNOUSERSITE", "1");
    environment.insert("PYTHONDONTWRITEBYTECODE", "1");
    m_process->setProcessEnvironment(environment);
    connect(m_process, &QProcess::readyReadStandardOutput, this,
            [this, generation] { collectOutput(generation); });
    connect(m_process, &QProcess::readyReadStandardError, this,
            [this, generation]
            {
                if (generation == m_generation && m_process)
                    m_stderr = (m_stderr + m_process->readAllStandardError()).right(outputLimit);
            });
    connect(m_process, &QProcess::errorOccurred, this,
            [this, generation](QProcess::ProcessError error)
            {
                if (generation == m_generation && error == QProcess::FailedToStart)
                    fail(QStringLiteral("无法启动模型结构服务。请检查 Python 运行环境与安装权限。"));
            });
    connect(m_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this, generation](int, QProcess::ExitStatus)
            {
                if (generation != m_generation)
                    return;
                collectOutput(generation);
                if (generation == m_generation)
                {
                    const QString detail = QString::fromUtf8(m_stderr).trimmed().right(1000);
                    fail(QStringLiteral("本机模型结构服务已意外结束。请重试。") +
                         (detail.isEmpty() ? QString() : "\n" + detail));
                }
            });
    m_deadline->start(15000);
    m_process->start(interpreter, {"-u", helper, "--model", m_canonicalPath});
}

void ModelViewer::collectOutput(quint64 generation)
{
    if (generation != m_generation || !m_process)
        return;
    m_stdout += m_process->readAllStandardOutput();
    if (m_stdout.size() > outputLimit)
    {
        fail(QStringLiteral("模型结构服务返回的数据异常，已停止加载。"));
        return;
    }
    qsizetype newline;
    while ((newline = m_stdout.indexOf('\n')) >= 0)
    {
        const QByteArray line = m_stdout.left(newline).trimmed();
        m_stdout.remove(0, newline + 1);
        if (line.isEmpty())
            continue;
        const QJsonDocument document = QJsonDocument::fromJson(line);
        const QJsonObject response = document.object();
        if (!document.isObject() || response.value("protocol").toInt() != 1)
        {
            fail(QStringLiteral("模型结构服务协议不匹配。请重新安装完整版本。"));
            return;
        }
        if (response.value("type").toString() == "error")
        {
            const QString message = response.value("message").toString().left(1500);
            fail(QStringLiteral("模型结构加载失败。\n%1").arg(message));
            return;
        }
        const QUrl url(response.value("url").toString());
        if (response.value("type").toString() != "ready" || m_serverReady ||
            response.value("version").toString() != netronVersion ||
            QFileInfo(response.value("model_path").toString()).canonicalFilePath() != m_canonicalPath ||
            !url.isValid() || url.scheme() != "http" || url.host() != "127.0.0.1" || url.port() <= 0 ||
            !url.userName().isEmpty() || !url.password().isEmpty() || url.path() != "/" || url.hasQuery() ||
            url.hasFragment())
        {
            fail(QStringLiteral("模型结构服务返回了无效的本机地址或模型信息，已停止加载。"));
            return;
        }
        m_serverReady = true;
        m_deadline->start(90000);
        startBrowser(url, generation);
    }
}

void ModelViewer::startBrowser(const QUrl &url, quint64 generation)
{
    m_profile = new QWebEngineProfile(this); // No storage name: private, memory-only profile.
    m_profile->setHttpCacheType(QWebEngineProfile::NoCache);
    m_profile->setPersistentCookiesPolicy(QWebEngineProfile::NoPersistentCookies);
    m_profile->setSpellCheckEnabled(false);
    m_profile->setUrlRequestInterceptor(new LocalRequests(url, m_profile));
    connect(m_profile, &QWebEngineProfile::downloadRequested, this,
            [](QWebEngineDownloadRequest *download) { download->cancel(); });
    m_webView = new QWebEngineView(m_stack);
    m_webView->setObjectName("netronWebView");
    m_webView->setAcceptDrops(false);
    auto *page = new LocalPage(m_profile, url, m_webView);
    page->setBackgroundColor(QColor("#0b1420"));
    m_webView->setPage(page);
    auto *settings = page->settings();
    settings->setAttribute(QWebEngineSettings::JavascriptCanOpenWindows, false);
    settings->setAttribute(QWebEngineSettings::JavascriptCanAccessClipboard, false);
    settings->setAttribute(QWebEngineSettings::LocalContentCanAccessRemoteUrls, false);
    settings->setAttribute(QWebEngineSettings::LocalContentCanAccessFileUrls, false);
    settings->setAttribute(QWebEngineSettings::LocalStorageEnabled, false);
    settings->setAttribute(QWebEngineSettings::PluginsEnabled, false);
    settings->setAttribute(QWebEngineSettings::PdfViewerEnabled, false);
    settings->setAttribute(QWebEngineSettings::NavigateOnDropEnabled, false);
    QWebEngineScript script;
    script.setName("VisionStudio offline Netron host and parse feedback");
    script.setInjectionPoint(QWebEngineScript::DocumentCreation);
    script.setWorldId(QWebEngineScript::MainWorld);
    script.setRunsOnSubFrames(false);
    script.setSourceCode(bridgeScript());
    page->scripts().insert(script);
    m_stack->addWidget(m_webView);
    m_webView->show();
    m_stack->setCurrentWidget(m_statusPage);
    connect(page, &QWebEnginePage::loadFinished, this,
            [this, generation](bool ok)
            {
                if (generation != m_generation || m_state != State::Loading)
                    return;
                if (!ok)
                {
                    fail(QStringLiteral("无法打开本机模型结构页面。请重试。"));
                    return;
                }
                m_status->setText(QStringLiteral("正在解析模型与排列节点…"));
                m_poll->start();
                checkGraph();
            });
    connect(page, &QWebEnginePage::renderProcessTerminated, this,
            [this, generation](QWebEnginePage::RenderProcessTerminationStatus, int)
            {
                if (generation == m_generation)
                    fail(QStringLiteral("模型结构浏览器已停止响应。请重试。"));
            });
    connect(page, &QWebEnginePage::visibleChanged, this,
            [this, generation](bool visible)
            {
                if (!visible && generation == m_generation)
                {
                    // Qt's view hideEvent changes Page::visible. Restore it after that event has
                    // completed so a non-current application page can still finish Netron's rAF layout.
                    QTimer::singleShot(0, this,
                                       [this, generation]
                                       {
                                           if (generation == m_generation)
                                               keepBrowserRendering();
                                       });
                }
            });
    keepBrowserRendering();
    page->load(url);
}

void ModelViewer::keepBrowserRendering()
{
    if (!m_webView)
        return;
    if (!m_webView->isVisible())
    {
        // A never-opened QStackedWidget page may not yet have a useful layout geometry.
        // Give its retained offscreen browser a real viewport; the layout supplies the actual
        // page dimensions when the user opens it, without reparsing the model.
        const QSize viewport(qMax(1100, m_stack->width()), qMax(720, m_stack->height()));
        if (m_webView->size() != viewport)
            m_webView->resize(viewport);
    }
    // Public Qt API: keep the renderer's animation clock active even while the widget is hidden.
    // This leaves the QWidget hidden and never creates another window or modifies Netron assets.
    if (!m_webView->page()->isVisible())
        m_webView->page()->setVisible(true);
}

void ModelViewer::checkGraph()
{
    if (!m_webView || m_checkPending || (m_state != State::Loading && m_state != State::Ready))
        return;
    keepBrowserRendering();
    const quint64 generation = m_generation;
    const QPointer<ModelViewer> self(this);
    const QPointer<QWebEnginePage> page(m_webView->page());
    m_checkPending = true;
    page->runJavaScript(
        "window.__visionStudioNetron ? window.__visionStudioNetron.snapshot() : ''",
        QWebEngineScript::MainWorld,
        [self, page, generation](const QVariant &result)
        {
            if (!self || generation != self->m_generation || !page || !self->m_webView ||
                self->m_webView->page() != page)
                return;
            self->m_checkPending = false;
            const QJsonObject snapshot = QJsonDocument::fromJson(result.toString().toUtf8()).object();
            const QString detail = snapshot.value("error").toString();
            if (!detail.isEmpty())
            {
                self->fail(QStringLiteral("Netron 无法解析或绘制此模型结构。文件可能损坏，或格式尚不受支持。"
                                          "\n解析器详情：%1")
                               .arg(detail.left(1500)));
                return;
            }
            if (snapshot.value("empty").toBool())
            {
                self->fail(QStringLiteral("模型已解析，但该文件没有可显示的运算图。"
                                          "请尝试导出 ONNX 后查看完整结构。"));
                return;
            }
            if (snapshot.value("ready").toBool())
                self->m_graphNodeCount = snapshot.value("nodes").toInt();
            if (self->m_state == State::Loading && snapshot.value("ready").toBool() &&
                snapshot.value("hostAdapted").toBool())
            {
                self->m_deadline->stop();
                const int nodes = snapshot.value("nodes").toInt();
                const QString summary = nodes > 0
                                            ? QStringLiteral("已解析模型结构 · %1 个节点").arg(nodes)
                                            : QStringLiteral("模型已解析；该文件未提供可视化运算节点。");
                self->setState(State::Ready, summary);
                if (!self || generation != self->m_generation || self->m_state != State::Ready)
                    return;
                self->m_poll->setInterval(500);
                emit self->modelLoaded(self->m_modelPath);
            }
        });
}
