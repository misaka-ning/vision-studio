#include "ui/modelviewer.h"

#include <QApplication>
#include <QAbstractButton>
#include <QAbstractItemModel>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPointer>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QTableView>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWebEnginePage>
#include <QWebEngineProfile>
#include <QWebEngineView>
#include <memory>
#include <functional>

namespace
{
constexpr int deadline = 30000;
constexpr int pathRole = Qt::UserRole + 1;
constexpr int typeRole = Qt::UserRole + 2;
constexpr int shapeRole = Qt::UserRole + 3;

QModelIndex findItem(QAbstractItemModel *model,
                    const std::function<bool(const QModelIndex &)> &matches,
                    const QModelIndex &parent = {})
{
    for (int row = 0; row < model->rowCount(parent); ++row)
    {
        const QModelIndex index = model->index(row, 0, parent);
        if (matches(index))
            return index;
        const QModelIndex child = findItem(model, matches, index);
        if (child.isValid())
            return child;
    }
    return {};
}

int itemCount(QAbstractItemModel *model, const QModelIndex &parent = {})
{
    int count = model->rowCount(parent);
    for (int row = 0; row < model->rowCount(parent); ++row)
        count += itemCount(model, model->index(row, 0, parent));
    return count;
}

class EnvironmentOverride
{
  public:
    EnvironmentOverride(const char *name, const QByteArray &value)
        : m_name(name), m_wasSet(qEnvironmentVariableIsSet(name)), m_previous(qgetenv(name))
    {
        qputenv(name, value);
    }
    ~EnvironmentOverride()
    {
        if (m_wasSet)
            qputenv(m_name.constData(), m_previous);
        else
            qunsetenv(m_name.constData());
    }

    EnvironmentOverride(const EnvironmentOverride &) = delete;
    EnvironmentOverride &operator=(const EnvironmentOverride &) = delete;

  private:
    QByteArray m_name;
    bool m_wasSet;
    QByteArray m_previous;
};

QJsonObject runJson(QWebEngineView *browser, const QString &expression)
{
    auto result = std::make_shared<QJsonObject>();
    QEventLoop loop;
    QPointer<QEventLoop> guard(&loop);
    browser->page()->runJavaScript(
        expression,
        [result, guard](const QVariant &value)
        {
            *result = QJsonDocument::fromJson(value.toString().toUtf8()).object();
            if (guard)
                guard->quit();
        });
    QTimer::singleShot(3000, &loop, &QEventLoop::quit);
    loop.exec();
    return *result;
}

QJsonObject graphSnapshot(QWebEngineView *browser)
{
    return runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
            const view = window.__view__;
            const graph = view && view.activeTarget;
            return { model: !!(view && view.model), active: !!graph,
                nodes: graph && Array.isArray(graph.nodes) ? graph.nodes.length : 0,
                rendered: !!document.querySelector('#target svg'),
                bodyDefault: document.body.classList.contains('default'),
                format: view && view.model ? String(view.model.format || '') : '' };
        })()))JS"));
}

void clickBrowserPoint(QWebEngineView *browser, const QJsonObject &position)
{
    const QPoint point(qRound(position.value("x").toDouble() * browser->zoomFactor()),
                       qRound(position.value("y").toDouble() * browser->zoomFactor()));
    QWidget *receiver = browser->focusProxy() ? browser->focusProxy() : browser;
    QTest::mouseClick(receiver, Qt::LeftButton, Qt::NoModifier, receiver->mapFrom(browser, point));
}

bool containsChinese(const QString &value)
{
    return value.contains(QRegularExpression(QStringLiteral("[\\x{4e00}-\\x{9fff}]")));
}
} // namespace

class ModelViewerTests : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        m_temporary = std::make_unique<QTemporaryDir>();
        QVERIFY(m_temporary->isValid());
        const QString project = QString::fromUtf8(VISION_PROJECT_DIR);
        const QString portablePython = project + "/runtime/bin/python";
        const QString defaultPython = QFileInfo(portablePython).isFile()
                                          ? portablePython
                                          : project + "/build/release-runtime/bin/python";
        const QString python = qEnvironmentVariable("VISION_STUDIO_PYTHON", defaultPython);
        QVERIFY2(QFileInfo(python).isExecutable(), qPrintable("Missing Netron runtime: " + python));
        const QString helper = project + "/scripts/netron_server.py";
        QVERIFY(QFileInfo(helper).isFile());
        m_python = std::make_unique<EnvironmentOverride>("VISION_STUDIO_PYTHON", python.toUtf8());
        m_helper = std::make_unique<EnvironmentOverride>("VISION_STUDIO_NETRON_WORKER", helper.toUtf8());

        QProcess probe;
        probe.start(python, {"-c", "import netron; print(netron.__version__)"});
        QVERIFY(probe.waitForFinished(5000));
        QCOMPARE(probe.exitStatus(), QProcess::NormalExit);
        QCOMPARE(probe.exitCode(), 0);
        QCOMPARE(probe.readAllStandardOutput().trimmed(), QByteArray("9.3.1"));

        const QString fixtures = m_temporary->path() + QStringLiteral("/中文 模型目录");
        QVERIFY(QDir().mkpath(fixtures));
        m_onnx = fixtures + QStringLiteral("/单目 检测.onnx");
        m_pt = fixtures + QStringLiteral("/YOLO 中文模型.pt");
        QVERIFY(QFile::copy(project + "/models/yolov5n.onnx", m_onnx));
        QVERIFY(QFile::copy(project + "/models/yolov8n.pt", m_pt));
        m_final = fixtures + QStringLiteral("/最终 模型.onnx");
        QVERIFY(QFile::copy(m_onnx, m_final));
        m_weights = fixtures + "/weights-only.pt";
        m_script = fixtures + "/linear-script.pt";
        m_scopedOnnx = fixtures + "/scoped-if.onnx";
        QProcess generated;
        generated.start(python, {project + "/tests/generate_modelviewer_fixtures.py", "--output", fixtures});
        QVERIFY(generated.waitForFinished(15000));
        QCOMPARE(generated.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generated.exitCode() == 0, generated.readAllStandardError().constData());
        for (const QString &path : {m_weights, m_script, m_scopedOnnx})
            QVERIFY(QFileInfo(path).size() > 0);
        m_corrupt = fixtures + "/corrupt.onnx";
        m_unknown = fixtures + "/model.unsupported-vision-format";
        for (const QString &path : {m_corrupt, m_unknown})
        {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
            QVERIFY(file.write("not a neural network protobuf or checkpoint\n") > 0);
        }
    }

    void realModelsRenderGraphs_data()
    {
        QTest::addColumn<QString>("path");
        QTest::addColumn<QString>("format");
        QTest::newRow("onnx-structure") << m_onnx << QStringLiteral("ONNX");
        QTest::newRow("pt-module-hierarchy") << m_pt << QStringLiteral("PyTorch");
    }

    void realModelsRenderGraphs()
    {
        QFETCH(QString, path);
        QFETCH(QString, format);
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(path);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
        QCOMPARE(loaded.first().first().toString(), QFileInfo(path).absoluteFilePath());
        QVERIFY(viewer.graphNodeCount() > 0);
        auto *browser = viewer.findChild<QWebEngineView *>("netronWebView");
        QVERIFY(browser);
        const QJsonObject graph = graphSnapshot(browser);
        QVERIFY(graph.value("model").toBool());
        QVERIFY(graph.value("active").toBool());
        QVERIFY(graph.value("rendered").toBool());
        QVERIFY(graph.value("bodyDefault").toBool());
        QVERIFY(graph.value("nodes").toInt() > 0);
        QCOMPARE(viewer.graphNodeCount(), graph.value("nodes").toInt());
        QVERIFY2(graph.value("format").toString().contains(format, Qt::CaseInsensitive),
                 qPrintable(graph.value("format").toString()));
        if (format == QStringLiteral("ONNX"))
        {
            // Hit-test an actual visible operator header, then send a native
            // Qt click to Chromium's render widget at those DOM coordinates.
            const QJsonObject node = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                for (const header of document.querySelectorAll('#target .graph-node .node-item-type')) {
                    const rect = header.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    if (x < 8 || x > innerWidth - 8 || y < 48 || y > innerHeight - 8) continue;
                    const hit = document.elementFromPoint(x, y);
                    if (!hit || hit.closest('.node-item-type') !== header) continue;
                    const container = header.closest('.graph-node');
                    const entry = Array.from(window.__view__.target.nodes.values())
                        .find(entry => entry.label && entry.label.element === container);
                    const value = entry && entry.label.value;
                    if (value && value.name && value.type) return { x, y,
                        name: value.name, type: value.type.identifier || value.type.name };
                }
                return {};
            })()))JS"));
            QVERIFY2(node.contains("x"), "No visible, unobstructed ONNX operator header was found");
            QVERIFY(!node.value("name").toString().isEmpty());
            QVERIFY(!node.value("type").toString().isEmpty());
            clickBrowserPoint(browser, node);
            auto sidebar = [browser]
            {
                return runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                    const sidebar = document.getElementById('sidebar');
                    const rect = sidebar.getBoundingClientRect();
                    return { visible: getComputedStyle(sidebar).opacity === '1' &&
                        rect.width > 0 && rect.left < innerWidth && rect.right > 0,
                        title: document.getElementById('sidebar-title').innerText,
                        text: document.getElementById('sidebar-content').innerText };
                })()))JS"));
            };
            QTRY_VERIFY_WITH_TIMEOUT(sidebar().value("visible").toBool(), 5000);
            const QJsonObject properties = sidebar();
            QCOMPARE(properties.value("title").toString().toCaseFolded(), QStringLiteral("node properties"));
            const QString text = properties.value("text").toString();
            QVERIFY2(text.contains(node.value("type").toString()), qPrintable(text));
            QVERIFY2(text.contains(node.value("name").toString()), qPrintable(text));
            QVERIFY2(text.contains(QStringLiteral("Inputs"), Qt::CaseInsensitive) &&
                         text.contains(QStringLiteral("Outputs"), Qt::CaseInsensitive),
                     qPrintable(text));

            const QJsonObject before = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                const canvas = document.getElementById('canvas');
                const button = document.getElementById('zoom-in-button');
                const rect = button.getBoundingClientRect();
                const transform = canvas.getScreenCTM();
                const x = rect.left + rect.width / 2;
                const y = rect.top + rect.height / 2;
                const hit = document.elementFromPoint(x, y);
                window.__visionStudioTestClicks = [];
                document.addEventListener('click', event => {
                    window.__visionStudioTestClicks.push({ id: event.target.id,
                        button: event.target.closest('button')?.id || '',
                        x: event.clientX, y: event.clientY });
                }, { capture: true });
                return { x, y, rect: { left: rect.left, top: rect.top,
                        width: rect.width, height: rect.height },
                    hit: hit?.closest('button')?.id || '',
                    viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
                    scale: transform.a, zoom: window.__view__.target.zoom,
                    width: canvas.style.width };
            })()))JS"));
            QVERIFY(before.value("scale").toDouble() > 0);
            QVERIFY(before.value("zoom").toDouble() < 1.4);
            QCOMPARE(before.value("hit").toString(), QStringLiteral("zoom-in-button"));
            clickBrowserPoint(browser, before);
            // Netron scales the SVG through its CSS dimensions, so screen CTM
            // verifies the real rendering transform rather than a style label.
            QJsonObject after;
            const bool zoomed = QTest::qWaitFor(
                [&]
                {
                    after = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                        const canvas = document.getElementById('canvas');
                        return { scale: canvas.getScreenCTM().a,
                            zoom: window.__view__.target.zoom, width: canvas.style.width,
                            clicks: window.__visionStudioTestClicks };
                    })()))JS"));
                    return after.value("zoom").toDouble() > before.value("zoom").toDouble();
                },
                3000);
            const QString diagnostic = QStringLiteral("before=%1 after=%2 browser=%3x%4 zoomFactor=%5")
                                           .arg(QString::fromUtf8(QJsonDocument(before).toJson(QJsonDocument::Compact)),
                                                QString::fromUtf8(QJsonDocument(after).toJson(QJsonDocument::Compact)))
                                           .arg(browser->width()).arg(browser->height()).arg(browser->zoomFactor());
            QVERIFY2(zoomed, qPrintable(diagnostic));
            QVERIFY(after.value("scale").toDouble() > before.value("scale").toDouble());
            QVERIFY(after.value("width").toString() != before.value("width").toString());
        }
        else
        {
            const QJsonObject colors = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                const rgb = value => {
                    const match = value.match(/^rgba?\(([^)]+)\)$/);
                    if (!match) return null;
                    const parts = match[1].split(',').map(Number);
                    return parts.length < 4 || parts[3] > 0 ? parts.slice(0, 3) : null;
                };
                const luminance = color => color.map(v => v / 255)
                    .map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4)
                    .reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
                const backgrounds = Array.from(document.querySelectorAll(
                    '#target .node-argument-list > path, #target .node-block-background, ' +
                    '#target .node-item-input > path, #target .node-item-function > path, ' +
                    '#target .node-item-constant > path')).map(path => ({
                        class: path.parentElement.getAttribute('class'),
                        fill: getComputedStyle(path).fill, rgb: rgb(getComputedStyle(path).fill)
                    })).filter(value => value.rgb);
                const white = backgrounds.filter(value => value.rgb.every(channel => channel > 235));
                const ratios = [];
                for (const text of document.querySelectorAll('#target text')) {
                    const group = text.closest('.node-item, .node-argument, .node-argument-list');
                    const path = group && Array.from(group.children).find(child => child.tagName === 'path');
                    if (!path) continue;
                    const foreground = rgb(getComputedStyle(text).fill);
                    const background = rgb(getComputedStyle(path).fill);
                    if (!foreground || !background) continue;
                    const a = luminance(foreground), b = luminance(background);
                    ratios.push({ text: text.textContent.slice(0, 60),
                        ratio: (Math.max(a, b) + .05) / (Math.min(a, b) + .05) });
                }
                const categories = {};
                for (const name of ['Conv2d', 'BatchNorm2d']) {
                    const header = Array.from(document.querySelectorAll('#target .node-item-type'))
                        .find(header => !header.matches(':hover') && header.querySelector('text')?.textContent === name);
                    const path = header && Array.from(header.children).find(child => child.tagName === 'path');
                    categories[name] = path ? rgb(getComputedStyle(path).fill) : null;
                }
                return { backgrounds: backgrounds.length, white: white.slice(0, 8),
                    compared: ratios.length,
                    lowContrast: ratios.filter(value => value.ratio < 4.5).slice(0, 8), categories };
            })()))JS"));
            const QByteArray diagnostic = QJsonDocument(colors).toJson(QJsonDocument::Compact);
            QVERIFY2(colors.value("backgrounds").toInt() > 0, diagnostic.constData());
            QVERIFY2(colors.value("white").toArray().isEmpty(), diagnostic.constData());
            QVERIFY2(colors.value("compared").toInt() > 0, diagnostic.constData());
            QVERIFY2(colors.value("lowContrast").toArray().isEmpty(), diagnostic.constData());
            const QJsonObject categories = colors.value("categories").toObject();
            const QJsonArray convolution = categories.value("Conv2d").toArray();
            const QJsonArray normalization = categories.value("BatchNorm2d").toArray();
            QVERIFY2(convolution.size() == 3 && normalization.size() == 3, diagnostic.constData());
            QVERIFY2(convolution[2].toInt() > convolution[1].toInt() &&
                         convolution[2].toInt() > convolution[0].toInt(), diagnostic.constData());
            QVERIFY2(normalization[1].toInt() > normalization[0].toInt() &&
                         normalization[1].toInt() > normalization[2].toInt(), diagnostic.constData());
        }
    }

    void threeModesShowRealMetadataWithoutReload_data()
    {
        QTest::addColumn<QString>("path");
        QTest::addColumn<QString>("weight");
        QTest::addColumn<QString>("shape");
        QTest::addColumn<QString>("dtype");
        QTest::addColumn<QString>("convolution");
        QTest::addColumn<bool>("weightsOnly");
        QTest::newRow("onnx") << m_onnx << QStringLiteral("model.0.conv.weight")
                              << QStringLiteral("[16,3,6,6]") << QStringLiteral("float32")
                              << QStringLiteral("Conv") << false;
        QTest::newRow("pytorch") << m_pt << QStringLiteral("model.0.conv.weight")
                                 << QStringLiteral("[16,3,3,3]") << QStringLiteral("float16")
                                 << QStringLiteral("Conv2d") << false;
        QTest::newRow("state-dict") << m_weights << QStringLiteral("backbone.block.conv.weight")
                                    << QStringLiteral("[4,3,3,3]") << QStringLiteral("float32")
                                    << QString() << true;
        QTest::newRow("torchscript-forward") << m_script << QStringLiteral("weight")
                                             << QStringLiteral("[3,4]") << QStringLiteral("float32")
                                             << QStringLiteral("linear") << false;
    }

    void threeModesShowRealMetadataWithoutReload()
    {
        QFETCH(QString, path);
        QFETCH(QString, weight);
        QFETCH(QString, shape);
        QFETCH(QString, dtype);
        QFETCH(QString, convolution);
        QFETCH(bool, weightsOnly);
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(path);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
        QVERIFY(viewer.hierarchyItemCount() > 0);
        QVERIFY(viewer.parameterCount() > 0);
        auto *tree = viewer.findChild<QTreeView *>("modelHierarchyTree");
        auto *table = viewer.findChild<QTableView *>("modelParameterTable");
        auto *search = viewer.findChild<QLineEdit *>("modelStructureSearch");
        auto *details = viewer.findChild<QLabel *>("modelSelectionDetails");
        auto *summary = viewer.findChild<QLabel *>("modelStructureSummary");
        auto *hint = viewer.findChild<QLabel *>("modelHierarchyHint");
        QVERIFY(tree && table && search && details && summary && hint);
        QVERIFY(tree->model() && table->model());
        QCOMPARE(itemCount(tree->model()), viewer.hierarchyItemCount());
        QCOMPARE(table->model()->rowCount(), viewer.parameterCount());
        if (weightsOnly)
            QCOMPARE(viewer.parameterCount(), 3);
        QPointer<QProcess> service = viewer.findChild<QProcess *>();
        QPointer<QWebEngineView> browser = viewer.findChild<QWebEngineView *>("netronWebView");
        QVERIFY(service && browser);
        const qint64 processId = service->processId();
        QSignalSpy pageLoads(browser->page(), &QWebEnginePage::loadStarted);

        viewer.setDisplayMode(ModelViewer::DisplayMode::Hierarchy);
        QCOMPARE(viewer.displayMode(), ModelViewer::DisplayMode::Hierarchy);
        QVERIFY(tree->isVisible());
        const QModelIndex module = findItem(tree->model(), [&](const QModelIndex &index)
        {
            return weightsOnly ? index.data(pathRole).toString().contains("backbone.block")
                               : index.data(typeRole).toString().contains(convolution, Qt::CaseInsensitive);
        });
        QVERIFY2(module.isValid(), qPrintable(convolution));
        QVERIFY(module.parent().isValid());
        QVERIFY(!module.data(pathRole).toString().isEmpty());
        if (weightsOnly)
        {
            const QString explanation = hint->text();
            QVERIFY2(explanation.contains(QStringLiteral("名称分组")) &&
                         explanation.contains(QStringLiteral("不代表前向")), qPrintable(explanation));
            const QModelIndex dictionary = findItem(tree->model(), [](const QModelIndex &index)
            {
                return index.data(typeRole).toString() == QStringLiteral("权重字典");
            });
            QVERIFY(dictionary.isValid());
        }
        else if (path == m_script)
        {
            QVERIFY(viewer.graphNodeCount() > 0);
            const QString explanation = hint->text();
            QVERIFY2(!explanation.contains(QStringLiteral("名称分组")), qPrintable(explanation));
        }

        viewer.setDisplayMode(ModelViewer::DisplayMode::Parameters);
        QCOMPARE(viewer.displayMode(), ModelViewer::DisplayMode::Parameters);
        QVERIFY(table->isVisible());
        if (weightsOnly)
        {
            table->sortByColumn(3, Qt::AscendingOrder);
            QTRY_COMPARE_WITH_TIMEOUT(table->model()->index(0, 3).data().toString(), QStringLiteral("4"), 3000);
            QCOMPARE(table->model()->index(1, 3).data().toString(), QStringLiteral("8"));
            QCOMPARE(table->model()->index(2, 3).data().toString(), QStringLiteral("108"));
            table->sortByColumn(0, Qt::AscendingOrder);
        }
        const QModelIndex parameter = findItem(table->model(), [&](const QModelIndex &index)
        {
            return index.data(pathRole).toString().contains(weight);
        });
        QVERIFY2(parameter.isValid(), qPrintable(weight));
        QCOMPARE(parameter.data(shapeRole).toString(), shape);
        QCOMPARE(table->model()->index(parameter.row(), 1).data().toString(), dtype);
        QCOMPARE(table->model()->index(parameter.row(), 2).data().toString(), shape);
        const int allParameters = table->model()->rowCount();
        search->setFocus();
        QTest::keyClicks(search, weight);
        QTRY_COMPARE_WITH_TIMEOUT(table->model()->rowCount(), 1, 3000);
        const QModelIndex filtered = table->model()->index(0, 0);
        QVERIFY(filtered.data(pathRole).toString().contains(weight));
        QCOMPARE(filtered.data(shapeRole).toString(), shape);
        table->scrollTo(filtered);
        QTest::mouseClick(table->viewport(), Qt::LeftButton, Qt::NoModifier,
                          table->visualRect(filtered).center());
        QTRY_VERIFY_WITH_TIMEOUT(!details->text().isEmpty(), 3000);
        QVERIFY2(details->text().contains(weight), qPrintable(details->text()));
        QVERIFY2(details->text().contains(shape), qPrintable(details->text()));
        QVERIFY2(details->text().contains(dtype), qPrintable(details->text()));
        search->clear();
        QTRY_COMPARE_WITH_TIMEOUT(table->model()->rowCount(), allParameters, 3000);

        for (const auto &choice : {qMakePair(ModelViewer::DisplayMode::Graph, "modelGraphModeButton"),
                                   qMakePair(ModelViewer::DisplayMode::Hierarchy, "modelHierarchyModeButton"),
                                   qMakePair(ModelViewer::DisplayMode::Parameters, "modelParametersModeButton"),
                                   qMakePair(ModelViewer::DisplayMode::Graph, "modelGraphModeButton")})
        {
            auto *button = viewer.findChild<QAbstractButton *>(choice.second);
            QVERIFY(button);
            QTest::mouseClick(button, Qt::LeftButton);
            QCOMPARE(viewer.displayMode(), choice.first);
            QCOMPARE(viewer.state(), ModelViewer::State::Ready);
            QCOMPARE(viewer.findChild<QProcess *>(), service.data());
            QCOMPARE(viewer.findChild<QWebEngineView *>("netronWebView"), browser.data());
            QCOMPARE(service->processId(), processId);
        }
        QTest::qWait(200);
        QCOMPARE(pageLoads.count(), 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
    }

    void changingOrStoppingModelClearsOldMetadata()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        viewer.setDisplayMode(ModelViewer::DisplayMode::Parameters);
        viewer.openModel(m_pt);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        auto *search = viewer.findChild<QLineEdit *>("modelStructureSearch");
        auto *details = viewer.findChild<QLabel *>("modelSelectionDetails");
        auto *tree = viewer.findChild<QTreeView *>("modelHierarchyTree");
        auto *table = viewer.findChild<QTableView *>("modelParameterTable");
        QVERIFY(search && details && tree && table);
        QVERIFY(viewer.parameterCount() > 0);
        search->setText(QStringLiteral("model.0.conv.weight"));
        QTRY_COMPARE_WITH_TIMEOUT(table->model()->rowCount(), 1, 3000);
        const QModelIndex selected = table->model()->index(0, 0);
        QTest::mouseClick(table->viewport(), Qt::LeftButton, Qt::NoModifier,
                          table->visualRect(selected).center());
        QTRY_VERIFY_WITH_TIMEOUT(details->text().contains(QStringLiteral("model.0.conv.weight")), 3000);

        viewer.openModel(m_corrupt);
        QCOMPARE(viewer.displayMode(), ModelViewer::DisplayMode::Parameters);
        QCOMPARE(viewer.parameterCount(), 0);
        QCOMPARE(viewer.hierarchyItemCount(), 0);
        QCOMPARE(table->model()->rowCount(), 0);
        QCOMPARE(tree->model()->rowCount(), 0);
        QVERIFY(search->text().isEmpty());
        QVERIFY(!details->text().contains(QStringLiteral("model.0.conv.weight")));
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Error);
        QCOMPARE(viewer.parameterCount(), 0);
        QCOMPARE(viewer.hierarchyItemCount(), 0);
        QVERIFY(!details->text().contains(QStringLiteral("model.0.conv.weight")));

        viewer.openModel(m_onnx);
        QCOMPARE(viewer.state(), ModelViewer::State::Loading);
        viewer.stop();
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QCOMPARE(viewer.parameterCount(), 0);
        QCOMPARE(viewer.hierarchyItemCount(), 0);
        QVERIFY(search->text().isEmpty());
        QVERIFY(!details->text().contains(QStringLiteral("model.0.conv.weight")));
        viewer.openModel(QString());
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QVERIFY(viewer.modelPath().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChildren<QProcess *>().isEmpty(), 3000);
    }

    void nestedOnnxGraphsKeepSameNamedParametersSeparate()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        viewer.setDisplayMode(ModelViewer::DisplayMode::Parameters);
        viewer.openModel(m_scopedOnnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        auto *table = viewer.findChild<QTableView *>("modelParameterTable");
        QVERIFY(table && table->model());
        QCOMPARE(viewer.parameterCount(), 2);
        QCOMPARE(table->model()->rowCount(), 2);
        QSet<QString> shapes;
        QSet<QString> paths;
        for (int row = 0; row < table->model()->rowCount(); ++row)
        {
            const QModelIndex parameter = table->model()->index(row, 0);
            paths.insert(parameter.data(pathRole).toString());
            const QString shape = parameter.data(shapeRole).toString();
            shapes.insert(shape);
            QCOMPARE(table->model()->index(row, 1).data().toString(), QStringLiteral("float32"));
            QCOMPARE(table->model()->index(row, 2).data().toString(), shape);
            const QString elements = shape == "[2,2]" ? QStringLiteral("4") : QStringLiteral("6");
            QCOMPARE(table->model()->index(row, 3).data().toString(), elements);
        }
        QCOMPARE(shapes, (QSet<QString>{"[2,2]", "[3,2]"}));
        QCOMPARE(paths.size(), 2);
    }

    void browserParsingFailuresAreChinese_data()
    {
        QTest::addColumn<QString>("path");
        QTest::newRow("unknown-format") << m_unknown;
        QTest::newRow("corrupt-onnx") << m_corrupt;
    }

    void browserParsingFailuresAreChinese()
    {
        QFETCH(QString, path);
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(path);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Error);
        QCOMPARE(loaded.count(), 0);
        QCOMPARE(failed.count(), 1);
        QVERIFY2(containsChinese(viewer.errorString()), qPrintable(viewer.errorString()));
        QVERIFY(viewer.errorString().contains(QStringLiteral("解析")) ||
                viewer.errorString().contains(QStringLiteral("格式")));
        QCOMPARE(viewer.graphNodeCount(), 0);
        QCOMPARE(viewer.hierarchyItemCount(), 0);
        QCOMPARE(viewer.parameterCount(), 0);
        QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChildren<QProcess *>().isEmpty(), 3000);
    }

    void rapidSwitchOnlyCompletesLatestModel()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChild<QWebEngineView *>("netronWebView"), deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Loading);
        QPointer<QWebEngineView> oldBrowser = viewer.findChild<QWebEngineView *>("netronWebView");
        QPointer<QProcess> oldService = viewer.findChild<QProcess *>();
        QVERIFY(oldService && oldService->state() == QProcess::Running);

        viewer.openModel(m_pt);
        viewer.openModel(m_corrupt);
        viewer.openModel(m_final);
        QVERIFY(oldBrowser.isNull());
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        QCOMPARE(viewer.modelPath(), QFileInfo(m_final).absoluteFilePath());
        QVERIFY(viewer.graphNodeCount() > 0);
        QVERIFY(viewer.hierarchyItemCount() > 0);
        QVERIFY(viewer.parameterCount() > 0);
        auto *parameters = viewer.findChild<QTableView *>("modelParameterTable");
        QVERIFY(parameters && parameters->model());
        const QModelIndex firstWeight = findItem(parameters->model(), [](const QModelIndex &index)
        {
            return index.data(pathRole).toString().contains(QStringLiteral("model.0.conv.weight"));
        });
        QVERIFY(firstWeight.isValid());
        QCOMPARE(firstWeight.data(shapeRole).toString(), QStringLiteral("[16,3,6,6]"));
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(loaded.first().first().toString(), QFileInfo(m_final).absoluteFilePath());
        QCOMPARE(failed.count(), 0);
        QTRY_VERIFY_WITH_TIMEOUT(oldService.isNull(), 3000);
        // Give retired browser/process callbacks time to arrive; they cannot
        // overwrite the latest model's state or emit stale completion signals.
        QTest::qWait(600);
        QCOMPARE(viewer.state(), ModelViewer::State::Ready);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
    }

    void hiddenStackedPagePreloadsAndKeepsCachedGraph()
    {
        QStackedWidget pages;
        pages.resize(1100, 760);
        auto *workspace = new QWidget(&pages);
        auto *modelPage = new QWidget(&pages);
        auto *layout = new QVBoxLayout(modelPage);
        auto *viewer = new ModelViewer(modelPage);
        layout->addWidget(viewer);
        pages.addWidget(workspace);
        pages.addWidget(modelPage);
        pages.setCurrentWidget(workspace);
        pages.show();
        QSignalSpy loaded(viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(viewer, &ModelViewer::loadFailed);
        QVERIFY(!viewer->isVisible());
        viewer->openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
        QCOMPARE(pages.currentWidget(), workspace);
        QVERIFY(!viewer->isVisible());
        QVERIFY(viewer->graphNodeCount() > 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
        QPointer<QProcess> service = viewer->findChild<QProcess *>();
        QPointer<QWebEngineView> browser = viewer->findChild<QWebEngineView *>("netronWebView");
        QVERIFY(service && service->state() == QProcess::Running);
        QVERIFY(browser);
        const qint64 processId = service->processId();
        const int nodes = viewer->graphNodeCount();
        QSignalSpy pageLoads(browser->page(), &QWebEnginePage::loadStarted);
        const QJsonObject graph = graphSnapshot(browser);
        QVERIFY(graph.value("rendered").toBool());
        QVERIFY(graph.value("nodes").toInt() > 0);

        pages.setCurrentWidget(modelPage);
        QTest::qWait(300);
        QVERIFY(viewer->isVisible());
        pages.setCurrentWidget(workspace);
        QTest::qWait(100);
        pages.setCurrentWidget(modelPage);
        QTest::qWait(300);
        QCOMPARE(viewer->state(), ModelViewer::State::Ready);
        QCOMPARE(viewer->graphNodeCount(), nodes);
        QCOMPARE(viewer->findChild<QProcess *>(), service.data());
        QCOMPARE(viewer->findChild<QWebEngineView *>("netronWebView"), browser.data());
        QCOMPARE(service->processId(), processId);
        QCOMPARE(pageLoads.count(), 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
    }

    void stopDuringLoadingReleasesStartedBrowserAndService()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChild<QWebEngineView *>("netronWebView"), deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Loading);
        QPointer<QWebEngineView> browser = viewer.findChild<QWebEngineView *>("netronWebView");
        QPointer<QWebEngineProfile> profile = browser->page()->profile();
        QPointer<QProcess> service = viewer.findChild<QProcess *>();
        QVERIFY(service && service->state() == QProcess::Running);
        viewer.stop();
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QCOMPARE(viewer.graphNodeCount(), 0);
        QCOMPARE(viewer.hierarchyItemCount(), 0);
        QCOMPARE(viewer.parameterCount(), 0);
        QVERIFY(viewer.errorString().isEmpty());
        QVERIFY(browser.isNull());
        QVERIFY(profile.isNull());
        QTRY_VERIFY_WITH_TIMEOUT(service.isNull(), 3000);
        QTest::qWait(400);
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QCOMPARE(loaded.count(), 0);
        QCOMPARE(failed.count(), 0);
    }

    void missingHelperOrRuntimeHasActionableChineseError()
    {
        ModelViewer viewer;
        {
            EnvironmentOverride missing("VISION_STUDIO_NETRON_WORKER",
                                        (m_temporary->path() + "/missing netron_server.py").toUtf8());
            viewer.openModel(m_onnx);
            QCOMPARE(viewer.state(), ModelViewer::State::Error);
            QVERIFY(viewer.errorString().contains("netron_server.py"));
            QVERIFY(containsChinese(viewer.errorString()));
            QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        }
        {
            EnvironmentOverride missing("VISION_STUDIO_PYTHON",
                                        (m_temporary->path() + "/missing python").toUtf8());
            viewer.openModel(m_onnx);
            QCOMPARE(viewer.state(), ModelViewer::State::Error);
            QVERIFY(viewer.errorString().contains("VISION_STUDIO_PYTHON"));
            QVERIFY(viewer.errorString().contains(QStringLiteral("运行环境")));
            QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
            QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        }
    }

    void cleanupTestCase()
    {
        m_helper.reset();
        m_python.reset();
        m_temporary.reset();
    }

  private:
    std::unique_ptr<QTemporaryDir> m_temporary;
    std::unique_ptr<EnvironmentOverride> m_python;
    std::unique_ptr<EnvironmentOverride> m_helper;
    QString m_onnx;
    QString m_pt;
    QString m_final;
    QString m_weights;
    QString m_script;
    QString m_scopedOnnx;
    QString m_corrupt;
    QString m_unknown;
};

int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication application(argc, argv);
    ModelViewerTests tests;
    return QTest::qExec(&tests, argc, argv);
}

#include "modelviewer_tests.moc"
