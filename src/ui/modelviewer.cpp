#include "modelviewer.h"

#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QTableView>
#include <QTimer>
#include <QTreeView>
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

class ParameterFilter final : public QSortFilterProxyModel
{
  public:
    using QSortFilterProxyModel::QSortFilterProxyModel;

  protected:
    bool lessThan(const QModelIndex &left, const QModelIndex &right) const override
    {
        if (left.column() != 3 || right.column() != 3)
            return QSortFilterProxyModel::lessThan(left, right);
        const QString a = left.data().toString();
        const QString b = right.data().toString();
        const auto integer = [](const QString &value)
        {
            if (value.isEmpty())
                return false;
            for (QChar character : value)
                if (character < QLatin1Char('0') || character > QLatin1Char('9'))
                    return false;
            return true;
        };
        const bool aKnown = integer(a), bKnown = integer(b);
        if (aKnown != bKnown)
            return sortOrder() == Qt::AscendingOrder ? aKnown : !aKnown;
        if (!aKnown)
            return QString::compare(a, b, Qt::CaseSensitive) < 0;
        const auto significant = [](const QString &value)
        {
            int start = 0;
            while (start < value.size() - 1 && value[start] == QLatin1Char('0'))
                ++start;
            return value.mid(start);
        };
        const QString first = significant(a), second = significant(b);
        return first.size() != second.size() ? first.size() < second.size()
                                             : QString::compare(first, second, Qt::CaseSensitive) < 0;
    }
};

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
                html, body, #target, .default { background: #202020 !important; color: #F5F5F5; }
                #sidebar, #menu, #toolbar, .sidebar, .menu { background-color: #272727 !important; color: #F5F5F5; }
                .sidebar-item, .sidebar-item-name, .sidebar-item-value { color: #F5F5F5 !important; }
                .sidebar-item-value, .sidebar-item-selector, .sidebar-find-search {
                    background: #303030 !important; border-color: #454545 !important; color: #F5F5F5 !important;
                }
                .sidebar-item-value-content { background: #303030 !important; border-color: #454545 !important; }
                .sidebar-find-query, .sidebar-find-content li, .sidebar-closebutton { color: #B8B8B8; }
                .sidebar-find-content li.focus { background: #35434a; color: #fff; }
                .sidebar-documentation a { color: #60CDFF; }
                .menu .menu-command { color: #F5F5F5 !important; }
                .menu .menu-command:disabled { color: #808080 !important; }
                .menu .menu-command:focus { background: #35434a; }
                .toolbar-select select, .toolbar-path-name-button, .toolbar-path-back-button {
                    background: #303030; border-color: #454545; color: #F5F5F5;
                }
                .toolbar-icon .stroke { stroke: #B8B8B8; }
                .toolbar-icon .fill { fill: #B8B8B8; }
                .toolbar-icon .border { stroke: #202020; }
                .node path, .node line { stroke: #626262; }
                .node-item path { stroke: #626262; }
                .node-item:not(.node-item-type) > path { fill: #272727; }
                .node-item text { fill: #F5F5F5 !important; }
                .node-item-type text { fill: #fff !important; }
                .node-item-type-constant > path, .node-item-type-control > path, .node-item-function > path { fill: #303030 !important; }
                .node-item-type-constant > text, .node-item-type-control > text, .node-item-function > text { fill: #F5F5F5 !important; }
                .node-item-type:hover > path { fill: #454545 !important; }
                .node-item-type:hover text { fill: #fff; }
                .node-item-undefined > path { fill: #a4323c !important; }
                .node-argument > text, .edge-label { fill: #F5F5F5 !important; }
                .node-argument-list > path, .node-argument-list:hover > path,
                .node-item-input > path, .node-item-input:hover > path,
                .node-item-constant > path, .node-item-constant:hover > path,
                .node-item-function > path, .node-item-function:hover > path,
                .graph-item-input > path, .graph-item-input:hover > path,
                .graph-item-output > path, .graph-item-output:hover > path { fill: #303030 !important; }
                .node-block > .node-block-background { fill: #202020 !important; }
                .node-block .edge-path { stroke: #B8B8B8; }
                .edge-path { stroke: #B8B8B8; }
                .edge-label text { fill: #B8B8B8; }
                #arrowhead, #arrowhead-tunnel { fill: #B8B8B8; }
                .select > .node.node-border, .select.edge-path, .select.node-argument > rect,
                .node-block .select.edge-path { stroke: #60CDFF !important; }
                #arrowhead-hover, #arrowhead-select { fill: #60CDFF !important; }
                #logo-github, #logo-netron { display: none !important; }
            `;
            document.head.appendChild(style);
        }, { once: true });
    })();)JS");
}

QString structureScript()
{
    // Only metadata already decoded by Netron is read. Tensor values/storage and the original
    // checkpoint are never serialized, evaluated, or loaded by another Python backend.
    return QString::fromUtf8(R"JS((() => {
        try {
            const model = window.__view__ && window.__view__.model;
            if (!model) return JSON.stringify({ error: 'Model is not parsed.' });
            const result = { schema: 1, nodes: [], parameters: [], kind: 'graph', notes: [], truncated: false };
            const maxNodes = 5000, maxParameters = 10000, maxDepth = 48, maxBytes = 4 * 1024 * 1024;
            let bytes = 0, saturated = false, moduleCount = 0, weightsDetected = false;
            const reasons = new Set();
            const seenNodes = new WeakMap(), seenTensors = new WeakMap();
            const isPyTorch = /PyTorch|TorchScript/.test(String(model.format || ''));
            const isTorchScript = /TorchScript/.test(String(model.format || ''));
            const limit = message => { result.truncated = true; reasons.add(message); };
            const text = (value, maximum = 512) => {
                if (typeof value !== 'string') return '';
                if (value.length > maximum) limit('元信息字段过长，已截断');
                return value.slice(0, maximum);
            };
            const append = (array, record) => {
                // Three bytes per UTF-16 unit conservatively bounds UTF-8, including CJK strings.
                bytes += JSON.stringify(record).length * 3;
                if (bytes > maxBytes - 8192) {
                    limit('结构数据超过 4 MiB 上限'); saturated = true; return false;
                }
                array.push(record); return true;
            };
            const shape = type => {
                const source = type && type.shape && type.shape.dimensions;
                if (!Array.isArray(source)) return { shape: '未知', elements: '—' };
                if (source.length > 32) {
                    limit('张量维度超过 32 维'); return { shape: '维度过多', elements: '—' };
                }
                let precise = true;
                const dimensions = source.map(value => {
                    let dimension = '?';
                    if (typeof value === 'bigint') dimension = value.toString();
                    else if (typeof value === 'number' && Number.isFinite(value)) {
                        dimension = String(value);
                        if (!Number.isSafeInteger(value) || value < 0) precise = false;
                    } else if (typeof value === 'string') dimension = value || '?';
                    else precise = false;
                    if (dimension.length > 80) {
                        limit('张量维度文本超过 80 字符，已截断'); precise = false;
                        return dimension.slice(0, 79) + '…';
                    }
                    return dimension;
                });
                let elements = '—';
                if (precise && dimensions.every(value => /^\d+$/.test(value))) {
                    elements = dimensions.reduce((count, value) => count * BigInt(value), 1n).toString();
                    if (elements.length > 160) elements = '—';
                }
                return { shape: '[' + dimensions.join(',') + ']', elements };
            };
            const tensor = (value, owner, argument, fallback) => {
                if (!value || typeof value !== 'object' || saturated) return;
                const initializer = value.initializer ||
                    ((fallback === 'tensor' || fallback === 'tensor[]') && value.type &&
                     ('values' in value || typeof value.encoding === 'string') ? value : null);
                if (!initializer || !initializer.type) return;
                const name = text(initializer.name || value.name || argument);
                const path = name.includes('.') || name.includes('/') ? name :
                    (owner ? owner + '.' + (name || argument) : name || argument || 'tensor');
                const known = seenTensors.get(initializer);
                if (known) {
                    if (known.path !== path) known.note = '共享张量；多个引用仅列一次';
                    return;
                }
                if (result.parameters.length >= maxParameters) { limit('张量表超过 10000 行'); return; }
                const type = initializer.type;
                const info = shape(type);
                const record = { path: text(path), type: text(type.dataType, 120) || '未知',
                    shape: info.shape, elements: info.elements, note: '文件中保存的权重 / 常量 / 缓冲' };
                if (append(result.parameters, record)) {
                    seenTensors.set(initializer, record);
                }
            };
            const argumentsOf = node => (Array.isArray(node.inputs) ? node.inputs : [])
                .concat(Array.isArray(node.attributes) ? node.attributes : []);
            const scanParameters = (arguments, owner) => {
                for (const argument of arguments) {
                    if (!argument || saturated) break;
                    const values = Array.isArray(argument.value) ? argument.value : [argument.value];
                    for (const value of values)
                        tensor(value, owner, text(argument.name, 160), argument.type);
                }
            };
            const addNode = (parent, name, path, type, extra = {}) => {
                if (saturated) return null;
                if (result.nodes.length >= maxNodes) { limit('层级树超过 5000 项'); return null; }
                const record = { id: 'n' + result.nodes.length, parent, name: text(name, 160) || '未命名',
                    path: text(path), type: text(type, 180) || '未知', shape: '', note: '', ...extra };
                return append(result.nodes, record) ? record : null;
            };
            const walkNode = (node, parent, depth, fallback, scope = '') => {
                if (!node || typeof node !== 'object' || saturated) return;
                const type = node.type || {};
                const identifier = text(type.identifier || type.name, 180) || '未知';
                const ownPath = node.name || node.identifier || fallback;
                const path = text(!isPyTorch && scope ? scope + '/' + ownPath : ownPath);
                if (type.type === 'weights' || identifier === 'Weights') weightsDetected = true;
                const name = text(node.name || node.identifier, 160).split('.').pop() ||
                    identifier.split('.').pop() || '未命名';
                const prior = seenNodes.get(node);
                const arguments = argumentsOf(node);
                const before = result.parameters.length;
                if (!prior && result.nodes.length < maxNodes) scanParameters(arguments, path);
                const own = result.parameters.slice(before, before + 3);
                const record = addNode(parent, name, path, identifier,
                    prior ? { note: '共享对象引用：' + prior.path } :
                            { shape: text(own.map(parameter => parameter.shape).join(' · ')) });
                if (!record || prior) return;
                seenNodes.set(node, record);
                if (depth >= maxDepth) { limit('层级深度超过 48 层'); record.note = '更深层级已截断'; return; }
                for (const argument of arguments) {
                    if (!argument || saturated) break;
                    if (argument.name === '_modules') moduleCount++;
                    if (argument.type === 'object' && argument.value)
                        walkNode(argument.value, record.id, depth + 1, path + '.' + argument.name);
                    else if (argument.type === 'object[]' && Array.isArray(argument.value))
                        argument.value.forEach((child, index) => walkNode(child, record.id, depth + 1,
                            path + '.' + argument.name + '.' + index));
                    else if (argument.type === 'graph' || argument.type === 'function')
                        walkGraph(argument.value, record.id, depth + 1, path + '.' + argument.name);
                }
                if (Array.isArray(node.nodes))
                    node.nodes.forEach((child, index) => walkNode(child, record.id, depth + 1, path + '.' + index));
                if (Array.isArray(type.nodes) && type.nodes.length)
                    walkGraph(type, record.id, depth + 1, path + '.weights');
            };
            const walkGraph = (graph, parent, depth, fallback) => {
                if (!graph || saturated) return;
                const graphPath = parent ? fallback + (graph.name ? '[' + graph.name + ']' : '') : graph.name || fallback || '模型';
                const root = addNode(parent, graph.name || fallback || '模型', graphPath,
                    graph.type === 'weights' ? '权重分组' : '模型 / 图');
                if (graph.type === 'weights') weightsDetected = true;
                if (!root) return;
                if (depth >= maxDepth) { limit('层级深度超过 48 层'); return; }
                scanParameters(Array.isArray(graph.inputs) ? graph.inputs : [], text(graphPath));
                let nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
                if (isPyTorch && nodes.length === 1) {
                    const arguments = argumentsOf(nodes[0]);
                    const checkpoint = ['model', 'ema', 'state_dict', 'model_state_dict']
                        .map(name => arguments.find(argument => argument.name === name &&
                            argument.type === 'object' && argument.value)).find(Boolean);
                    if (checkpoint) {
                        nodes = [checkpoint.value];
                        result.notes.push('显示 checkpoint 的 ' + checkpoint.name + ' 分支；训练元数据与重复分支未列入');
                    }
                }
                for (let index = 0; index < nodes.length; index++) {
                    if (saturated || result.nodes.length >= maxNodes) { limit('层级树超过 5000 项'); break; }
                    walkNode(nodes[index], root.id, depth + 1, 'node.' + index, graphPath);
                }
            };
            let modules = Array.isArray(model.modules) ? model.modules : [];
            if (isPyTorch && modules.length > 1) {
                const primary = modules.find(graph => graph.name === 'model') || modules.find(graph => graph.name === 'ema');
                if (primary) { modules = [primary]; result.notes.push('仅显示 checkpoint 的主模型分支'); }
            }
            modules.forEach((graph, index) => walkGraph(graph, '', 0, graph.name || '模型' + (index ? ' ' + index : '')));
            result.kind = moduleCount ? 'modules' : isPyTorch && !isTorchScript && weightsDetected && result.parameters.length ? 'parameters' : 'graph';
            if (result.kind === 'parameters') {
                // A state_dict does not describe execution order. Build groups from saved names only.
                result.nodes = []; bytes = JSON.stringify(result.parameters).length * 3; saturated = false;
                const root = addNode('', '参数名称分组', 'state_dict', '权重字典');
                const groups = new Map([['', root]]);
                if (root) for (const parameter of result.parameters) {
                    if (saturated) break;
                    const parts = parameter.path.split('.');
                    let prefix = '', parent = root;
                    for (let index = 0; index < Math.min(parts.length, maxDepth); index++) {
                        prefix += (prefix ? '.' : '') + parts[index];
                        let item = groups.get(prefix);
                        if (!item) {
                            const leaf = index === parts.length - 1;
                            item = addNode(parent.id, parts[index], prefix, leaf ? parameter.type : '参数名称分组',
                                leaf ? { shape: parameter.shape, note: '按保存的参数名称分组，不代表计算连接' } : {});
                            if (!item) break;
                            groups.set(prefix, item);
                        }
                        parent = item;
                    }
                    if (parts.length > maxDepth) limit('参数名称分组超过 48 层');
                }
            }
            result.notes.push(...reasons);
            let encoded = JSON.stringify(result);
            while (new TextEncoder().encode(encoded).length > maxBytes) {
                result.truncated = true;
                if (!result.notes.includes('结构数据超过 4 MiB 上限')) result.notes.push('结构数据超过 4 MiB 上限');
                if (result.nodes.length > 1) result.nodes.splice(Math.max(1, result.nodes.length - 128));
                else if (result.parameters.length) result.parameters.splice(Math.max(0, result.parameters.length - 128));
                else break;
                encoded = JSON.stringify(result);
            }
            return encoded;
        } catch (error) { return JSON.stringify({ error: String(error && error.message || error) }); }
    })())JS");
}
} // namespace

ModelViewer::ModelViewer(QWidget *parent) : QWidget(parent)
{
    setObjectName("modelViewer");
    qRegisterMetaType<State>();
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    auto *toolbar = new QHBoxLayout;
    toolbar->setSpacing(6);
    const QStringList names = {QStringLiteral("结构图"), QStringLiteral("层级树"), QStringLiteral("参数表")};
    const QStringList objects = {"modelGraphModeButton", "modelHierarchyModeButton",
                                 "modelParametersModeButton"};
    for (int index = 0; index < names.size(); ++index)
    {
        auto *mode = new QPushButton(names[index], this);
        mode->setObjectName(objects[index]);
        mode->setCheckable(true);
        mode->setCursor(Qt::PointingHandCursor);
        mode->setMinimumHeight(34);
        mode->setStyleSheet(
            "QPushButton { background:#303030; color:#B8B8B8; border:1px solid #454545; "
            "border-radius:4px; padding:6px 15px; } QPushButton:checked { background:#35434a; "
            "color:#60CDFF; border-color:#60CDFF; } QPushButton:hover { background:#383838; "
            "border-color:#626262; } QPushButton:checked:hover { background:#3b4d57; "
            "border-color:#60CDFF; } QPushButton:focus { border-color:#60CDFF; } "
            "QPushButton:disabled { background:#272727; color:#808080; border-color:#383838; }");
        toolbar->addWidget(mode);
        m_modeButtons.append(mode);
        connect(mode, &QPushButton::clicked, this,
                [this, index] { setDisplayMode(static_cast<DisplayMode>(index)); });
    }
    toolbar->addStretch();
    m_search = new QLineEdit(this);
    m_search->setObjectName("modelStructureSearch");
    m_search->setPlaceholderText(QStringLiteral("搜索名称、类型或路径"));
    m_search->setClearButtonEnabled(true);
    m_search->setMinimumWidth(180);
    m_search->setMaximumWidth(340);
    m_search->setMinimumHeight(34);
    m_search->setStyleSheet(
        "QLineEdit { background:#303030; color:#F5F5F5; border:1px solid #454545; "
        "border-radius:4px; padding:6px 10px; } QLineEdit:focus { border-color:#60CDFF; }");
    toolbar->addWidget(m_search);
    layout->addLayout(toolbar);
    m_summary = new QLabel(this);
    m_summary->setObjectName("modelStructureSummary");
    m_summary->setTextFormat(Qt::PlainText);
    m_summary->setWordWrap(true);
    m_summary->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_summary->setStyleSheet("color:#B8B8B8; font-size:12px;");
    layout->addWidget(m_summary);
    m_hint = new QLabel(this);
    m_hint->setObjectName("modelHierarchyHint");
    m_hint->setWordWrap(true);
    m_hint->setTextFormat(Qt::PlainText);
    m_hint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_hint->setStyleSheet("color:#B8B8B8; font-size:12px;");
    layout->addWidget(m_hint);
    m_stack = new QStackedWidget(this);
    // Netron measures its SVG after three animation frames. Keep its browser visible beneath
    // the native loading panel so Chromium can deliver those frames before State::Ready.
    static_cast<QStackedLayout *>(m_stack->layout())->setStackingMode(QStackedLayout::StackAll);
    layout->addWidget(m_stack);
    m_statusPage = new QWidget(m_stack);
    m_statusPage->setObjectName("modelViewerStatePage");
    m_statusPage->setStyleSheet("#modelViewerStatePage { background: #272727; border: 1px solid #454545; "
                                "border-radius: 8px; }");
    auto *statusLayout = new QVBoxLayout(m_statusPage);
    statusLayout->setContentsMargins(36, 40, 36, 40);
    statusLayout->setSpacing(16);
    statusLayout->addStretch();
    m_title = new QLabel(QStringLiteral("查看模型结构"), m_statusPage);
    m_title->setAlignment(Qt::AlignCenter);
    m_title->setStyleSheet("color: #F5F5F5; font-size: 22px; font-weight: 600;");
    statusLayout->addWidget(m_title);
    m_status = new QLabel(m_statusPage);
    m_status->setObjectName("modelViewerStatus");
    m_status->setAlignment(Qt::AlignCenter);
    m_status->setWordWrap(true);
    m_status->setTextFormat(Qt::PlainText);
    m_status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_status->setStyleSheet("color: #B8B8B8; font-size: 14px;");
    statusLayout->addWidget(m_status);
    m_progress = new QProgressBar(m_statusPage);
    m_progress->setRange(0, 0);
    m_progress->setMaximumWidth(300);
    m_progress->setFixedHeight(4);
    m_progress->setTextVisible(false);
    m_progress->setStyleSheet("QProgressBar { background: #303030; border: 0; border-radius: 2px; } "
                              "QProgressBar::chunk { background: #60CDFF; }");
    statusLayout->addWidget(m_progress, 0, Qt::AlignHCenter);
    m_retry = new QPushButton(QStringLiteral("重试加载"), m_statusPage);
    m_retry->setObjectName("modelViewerRetry");
    m_retry->setMinimumWidth(140);
    m_retry->setCursor(Qt::PointingHandCursor);
    statusLayout->addWidget(m_retry, 0, Qt::AlignHCenter);
    statusLayout->addStretch();
    m_stack->addWidget(m_statusPage);
    m_hierarchyPage = new QWidget(m_stack);
    auto *treeLayout = new QVBoxLayout(m_hierarchyPage);
    treeLayout->setContentsMargins(0, 0, 0, 0);
    m_tree = new QTreeView(m_hierarchyPage);
    m_tree->setObjectName("modelHierarchyTree");
    m_treeModel = new QStandardItemModel(this);
    m_treeFilter = new QSortFilterProxyModel(this);
    m_treeFilter->setSourceModel(m_treeModel);
    m_treeFilter->setFilterCaseSensitivity(Qt::CaseInsensitive);
    m_treeFilter->setFilterKeyColumn(-1);
    m_treeFilter->setRecursiveFilteringEnabled(true);
    m_tree->setModel(m_treeFilter);
    m_tree->setAlternatingRowColors(true);
    m_tree->setUniformRowHeights(true);
    m_tree->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tree->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tree->setSelectionMode(QAbstractItemView::SingleSelection);
    m_tree->setIndentation(22);
    treeLayout->addWidget(m_tree);
    m_stack->addWidget(m_hierarchyPage);
    m_parametersPage = new QWidget(m_stack);
    auto *parameterLayout = new QVBoxLayout(m_parametersPage);
    parameterLayout->setContentsMargins(0, 0, 0, 0);
    m_parameters = new QTableView(m_parametersPage);
    m_parameters->setObjectName("modelParameterTable");
    m_parameterModel = new QStandardItemModel(this);
    m_parameterFilter = new ParameterFilter(this);
    m_parameterFilter->setSourceModel(m_parameterModel);
    m_parameterFilter->setFilterCaseSensitivity(Qt::CaseInsensitive);
    m_parameterFilter->setFilterKeyColumn(-1);
    m_parameters->setModel(m_parameterFilter);
    m_parameters->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_parameters->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_parameters->setSelectionMode(QAbstractItemView::SingleSelection);
    m_parameters->setAlternatingRowColors(true);
    m_parameters->setSortingEnabled(true);
    m_parameters->verticalHeader()->hide();
    m_parameters->verticalHeader()->setDefaultSectionSize(32);
    parameterLayout->addWidget(m_parameters);
    m_stack->addWidget(m_parametersPage);
    const QString viewStyle =
        "QTreeView, QTableView { background:#272727; alternate-background-color:#2c2c2c; color:#F5F5F5; "
        "border:1px solid #454545; border-radius:8px; font-size:13px; gridline-color:#404040; outline:0; } "
        "QTreeView::item, QTableView::item { padding:6px; } "
        "QTreeView::item:selected, QTableView::item:selected { background:#35434a; color:#F5F5F5; } "
        "QTreeView::item:hover, QTableView::item:hover { background:#353535; } "
        "QHeaderView::section { background:#303030; color:#B8B8B8; border:0; border-bottom:1px solid "
        "#454545; "
        "padding:8px; font-weight:600; } QTableView QTableCornerButton::section { background:#303030; "
        "border:0; }";
    m_tree->setStyleSheet(viewStyle);
    m_parameters->setStyleSheet(viewStyle);
    m_details = new QLabel(this);
    m_details->setObjectName("modelSelectionDetails");
    m_details->setTextFormat(Qt::PlainText);
    m_details->setWordWrap(true);
    m_details->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_details->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_details->setMinimumHeight(78);
    m_details->setMaximumHeight(140);
    m_details->setStyleSheet("background:#303030; color:#F5F5F5; border:1px solid #454545; "
                             "border-radius:8px; padding:10px 14px; font-size:12px;");
    layout->addWidget(m_details);
    connect(m_search, &QLineEdit::textChanged, this,
            [this](const QString &query)
            {
                m_treeFilter->setFilterFixedString(query);
                m_parameterFilter->setFilterFixedString(query);
                if (!query.isEmpty())
                    m_tree->expandAll();
                else
                    m_tree->expandToDepth(2);
                if (m_mode == DisplayMode::Hierarchy)
                    updateSelectionDetails(true);
                else if (m_mode == DisplayMode::Parameters)
                    updateSelectionDetails(false);
            });
    connect(m_tree->selectionModel(), &QItemSelectionModel::currentChanged, this,
            [this]
            {
                if (m_mode == DisplayMode::Hierarchy)
                    updateSelectionDetails(true);
            });
    connect(m_parameters->selectionModel(), &QItemSelectionModel::currentChanged, this,
            [this]
            {
                if (m_mode == DisplayMode::Parameters)
                    updateSelectionDetails(false);
            });
    clearStructure();
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

void ModelViewer::setDisplayMode(DisplayMode mode)
{
    if (mode != DisplayMode::Graph && mode != DisplayMode::Hierarchy && mode != DisplayMode::Parameters)
        return;
    const bool changed = m_mode != mode;
    m_mode = mode;
    updateDisplay();
    if (changed)
        emit displayModeChanged(mode);
}

ModelViewer::DisplayMode ModelViewer::displayMode() const
{
    return m_mode;
}

int ModelViewer::hierarchyItemCount() const
{
    return m_hierarchyCount;
}

int ModelViewer::parameterCount() const
{
    return m_parameterCount;
}

void ModelViewer::clearStructure()
{
    m_structurePending = false;
    m_structureReady = false;
    m_hierarchyCount = 0;
    m_parameterCount = 0;
    m_hierarchyHint.clear();
    m_search->clear();
    m_treeModel->clear();
    m_treeModel->setHorizontalHeaderLabels(
        {QStringLiteral("模块 / 对象"), QStringLiteral("类型"), QStringLiteral("路径")});
    m_parameterModel->clear();
    m_parameterModel->setHorizontalHeaderLabels({QStringLiteral("张量 / 参数路径"),
                                                 QStringLiteral("数据类型"), QStringLiteral("形状"),
                                                 QStringLiteral("元素数"), QStringLiteral("备注")});
    m_summary->clear();
    m_hint->clear();
    m_details->clear();
}

void ModelViewer::updateDisplay()
{
    for (int index = 0; index < m_modeButtons.size(); ++index)
        m_modeButtons[index]->setChecked(index == static_cast<int>(m_mode));
    const bool ready = m_state == State::Ready && m_structureReady;
    const bool native = m_mode != DisplayMode::Graph;
    m_search->setVisible(native);
    m_search->setEnabled(ready);
    m_search->setPlaceholderText(m_mode == DisplayMode::Parameters
                                     ? QStringLiteral("搜索路径、类型、形状或备注")
                                     : QStringLiteral("搜索名称、类型或路径"));
    m_summary->setVisible(ready);
    m_hint->setVisible(ready && native);
    m_details->setVisible(ready && native);
    if (ready && m_mode == DisplayMode::Hierarchy)
    {
        m_stack->setCurrentWidget(m_hierarchyPage);
        m_hint->setText(m_hierarchyHint);
        if (!m_tree->currentIndex().isValid() && m_treeFilter->rowCount())
            m_tree->setCurrentIndex(m_treeFilter->index(0, 0));
        updateSelectionDetails(true);
    }
    else if (ready && m_mode == DisplayMode::Parameters)
    {
        m_stack->setCurrentWidget(m_parametersPage);
        m_hint->setText(
            QStringLiteral("只读张量元信息，包含文件中保存的权重、常量与缓冲；元素数未知时显示 —，"
                           "此表不等同可训练参数总量。"));
        if (!m_parameters->currentIndex().isValid() && m_parameterFilter->rowCount())
            m_parameters->setCurrentIndex(m_parameterFilter->index(0, 0));
        updateSelectionDetails(false);
    }
    else
        m_stack->setCurrentWidget(ready && m_webView ? static_cast<QWidget *>(m_webView) : m_statusPage);
}

void ModelViewer::updateSelectionDetails(bool hierarchy)
{
    const QModelIndex index = (hierarchy ? m_tree->currentIndex() : m_parameters->currentIndex());
    const QModelIndex first = index.isValid() ? index.siblingAtColumn(0) : QModelIndex();
    m_details->setText(first.isValid() ? first.data(Qt::UserRole + 4).toString()
                                       : QStringLiteral("选择一项查看只读元信息。"));
}

void ModelViewer::applyStructure(const QJsonObject &structure)
{
    const QJsonArray nodes = structure.value("nodes").toArray();
    const QJsonArray parameters = structure.value("parameters").toArray();
    QHash<QString, QStandardItem *> parents;
    QModelIndex initialModule;
    for (const QJsonValue &value : nodes)
    {
        const QJsonObject node = value.toObject();
        const QString path = node.value("path").toString();
        const QString type = node.value("type").toString();
        const QString shape = node.value("shape").toString();
        const QString note = node.value("note").toString();
        const QString details =
            QStringLiteral("路径：%1\n类型：%2%3%4")
                .arg(path, type, shape.isEmpty() ? QString() : QStringLiteral("\n保存的张量形状：") + shape,
                     note.isEmpty() ? QString() : QStringLiteral("\n") + note);
        QList<QStandardItem *> row = {new QStandardItem(node.value("name").toString()),
                                      new QStandardItem(type), new QStandardItem(path)};
        for (QStandardItem *item : row)
            item->setEditable(false);
        row[0]->setData(path, Qt::UserRole + 1);
        row[0]->setData(type, Qt::UserRole + 2);
        row[0]->setData(shape, Qt::UserRole + 3);
        row[0]->setData(details, Qt::UserRole + 4);
        row[0]->setToolTip(details);
        QStandardItem *parent =
            parents.value(node.value("parent").toString(), m_treeModel->invisibleRootItem());
        parent->appendRow(row);
        parents.insert(node.value("id").toString(), row[0]);
        if (!initialModule.isValid() && !node.value("parent").toString().isEmpty() &&
            type != QStringLiteral("模型 / 图") && type != QStringLiteral("参数名称分组") &&
            type != QStringLiteral("权重分组") && type != QStringLiteral("builtins.object"))
            initialModule = row[0]->index();
    }
    for (const QJsonValue &value : parameters)
    {
        const QJsonObject parameter = value.toObject();
        const QString path = parameter.value("path").toString();
        const QString type = parameter.value("type").toString();
        const QString shape = parameter.value("shape").toString();
        const QString elements = parameter.value("elements").toString(QStringLiteral("—"));
        const QString note = parameter.value("note").toString();
        const QString details =
            QStringLiteral("路径：%1\n数据类型：%2    形状：%3    元素数：%4\n%5 · 只读元信息")
                .arg(path, type, shape, elements, note);
        QList<QStandardItem *> row = {new QStandardItem(path), new QStandardItem(type),
                                      new QStandardItem(shape), new QStandardItem(elements),
                                      new QStandardItem(note)};
        for (QStandardItem *item : row)
            item->setEditable(false);
        row[0]->setData(path, Qt::UserRole + 1);
        row[0]->setData(type, Qt::UserRole + 2);
        row[0]->setData(shape, Qt::UserRole + 3);
        row[0]->setData(details, Qt::UserRole + 4);
        row[0]->setToolTip(details);
        m_parameterModel->appendRow(row);
    }
    m_hierarchyCount = nodes.size();
    m_parameterCount = parameters.size();
    m_tree->header()->setStretchLastSection(true);
    m_tree->setColumnWidth(0, 340);
    m_tree->setColumnWidth(1, 300);
    m_parameters->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < m_parameterModel->columnCount(); ++column)
        m_parameters->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    m_parameters->horizontalHeader()->setMaximumSectionSize(400);
    m_parameters->sortByColumn(0, Qt::AscendingOrder);
    m_tree->expandToDepth(2);
    if (initialModule.isValid())
        m_tree->setCurrentIndex(m_treeFilter->mapFromSource(initialModule));
    else if (m_treeFilter->rowCount())
        m_tree->setCurrentIndex(m_treeFilter->index(0, 0));
    if (m_parameterFilter->rowCount())
        m_parameters->setCurrentIndex(m_parameterFilter->index(0, 0));
    const QString kind = structure.value("kind").toString();
    m_hierarchyHint = kind == "parameters" ? QStringLiteral("按保存的参数名称分组，不代表前向计算连接。")
                      : kind == "modules"
                          ? QStringLiteral("层级树显示模块 / 对象的包含关系，不代表前向计算连接。")
                          : QStringLiteral("层级树显示文件中的包含关系，运算连接请查看结构图。");
    QStringList notes;
    for (const QJsonValue &value : structure.value("notes").toArray())
        if (!value.toString().isEmpty())
            notes.append(value.toString().left(512));
    QString summary = QStringLiteral("%1 个图节点 · %2 个层级项 · %3 个张量条目 · 共享一次解析缓存")
                          .arg(m_graphNodeCount)
                          .arg(m_hierarchyCount)
                          .arg(m_parameterCount);
    if (structure.value("truncated").toBool())
        summary += QStringLiteral("\n已达到展示上限，部分内容已截断。");
    if (!notes.isEmpty())
        summary += QStringLiteral("\n") + notes.join(QStringLiteral("；"));
    m_summary->setText(summary);
}

void ModelViewer::extractStructure(quint64 generation)
{
    if (!m_webView || m_structurePending || m_structureReady)
        return;
    m_structurePending = true;
    m_status->setText(QStringLiteral("正在整理层级与张量元信息…"));
    const QPointer<ModelViewer> self(this);
    const QPointer<QWebEnginePage> page(m_webView->page());
    page->runJavaScript(structureScript(), QWebEngineScript::MainWorld,
                        [self, page, generation](const QVariant &result)
                        {
                            if (!self || generation != self->m_generation || !page || !self->m_webView ||
                                self->m_webView->page() != page)
                                return;
                            self->m_structurePending = false;
                            const QByteArray encoded = result.toString().toUtf8();
                            if (encoded.size() > 4 * 1024 * 1024)
                            {
                                self->fail(QStringLiteral("模型结构元信息超过 4 MiB 展示上限。"));
                                return;
                            }
                            const QJsonObject structure = QJsonDocument::fromJson(encoded).object();
                            if (structure.value("schema").toInt() != 1 ||
                                !structure.value("nodes").isArray() ||
                                !structure.value("parameters").isArray() ||
                                structure.value("nodes").toArray().size() > 5000 ||
                                structure.value("parameters").toArray().size() > 10000)
                            {
                                self->fail(QStringLiteral("模型已解析，但无法整理可显示的结构元信息。\n%1")
                                               .arg(structure.value("error").toString().left(1500)));
                                return;
                            }
                            self->applyStructure(structure);
                            self->m_structureReady = true;
                            self->m_deadline->stop();
                            self->setState(State::Ready, QStringLiteral("模型结构与只读元信息已缓存。"));
                            if (!self || generation != self->m_generation || self->m_state != State::Ready)
                                return;
                            self->m_poll->setInterval(500);
                            emit self->modelLoaded(self->m_modelPath);
                        });
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
    updateDisplay();
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
    clearStructure();
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
    page->setBackgroundColor(QColor("#202020"));
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
                self->extractStructure(generation);
            }
        });
}
