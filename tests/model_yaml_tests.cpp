#include "ui/modelyaml.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTest>
#include <atomic>
#include <thread>
#include <vector>

namespace
{
QJsonObject blueprint()
{
    return {{"nc", 2}, {"depth_multiple", .33}, {"width_multiple", .25}, {"ch", 3},
        {"backbone", QJsonArray{QJsonArray{-1, 1, "Conv", QJsonArray{64, 3, 2}},
                                  QJsonArray{-1, 3, "C2f", QJsonArray{128, true}}}},
        {"head", QJsonArray{QJsonArray{QJsonArray{0, 1}, 1, "Concat", QJsonArray{1}},
                              QJsonArray{QJsonArray{1, 2}, 1, "Detect", QJsonArray{"nc"}}}}};
}

QByteArray bytes(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
}

class ModelYamlTests : public QObject
{
    Q_OBJECT
  private slots:
    void blueprintRowsPreserveAllOriginalTypesAndOrder()
    {
        auto object = blueprint();
        object.insert("activation", QStringLiteral("nn.ReLU()\n!!python/object/apply:os.system # data only"));
        object.insert("yaml_file", "/unavailable/original/file.yaml");
        auto backbone = object.value("backbone").toArray();
        auto first = backbone[0].toArray(); first[2] = "null";
        first[3] = QJsonArray{"true", "None", "a: b", "x\ny", QJsonValue::Null, false,
            QJsonObject{{"!!python/object", "never executed"}}};
        backbone[0] = first; object.insert("backbone", backbone);
        const auto document = vision::modelYamlDocument(object);
        QVERIFY2(document.available, qPrintable(document.notice));
        QCOMPARE(document.layers, 4);
        QVERIFY(document.notice.contains(QStringLiteral("不是原始")));
        QVERIFY(document.text.contains("# [from, repeats, module, args]"));
        QVERIFY(!document.text.contains("/unavailable/original/file.yaml"));
        QVERIFY(document.text.contains("\\n!!python/object"));
        QJsonArray expected = backbone;
        for (const auto &row : object.value("head").toArray()) expected.append(row);
        int row = 0;
        for (const auto &line : document.text.split('\n'))
            if (line.startsWith("  - ["))
            {
                QJsonParseError error;
                const auto parsed = QJsonDocument::fromJson(line.mid(4).toUtf8(), &error);
                QCOMPARE(error.error, QJsonParseError::NoError);
                QCOMPARE(parsed.array(), expected[row++].toArray());
            }
        QCOMPARE(row, 4);
        QCOMPARE(document.text, vision::modelYamlDocument(object).text);
    }

    void modernScalesAndLegacyAnchorsArePreserved()
    {
        auto object = blueprint();
        object.insert("scales", QJsonObject{{"n", QJsonArray{.33, .25, 1024}}, {"s", QJsonArray{.33, .5, 1024}}});
        object.insert("scale", "n");
        object.insert("anchors", QJsonArray{QJsonArray{10, 13, 16, 30, 33, 23}, QJsonArray{30, 61, 62, 45, 59, 119}});
        const auto document = vision::modelYamlDocument(object);
        QVERIFY2(document.available, qPrintable(document.notice));
        QVERIFY(document.text.contains("scales: {\"n\": [0.33, 0.25, 1024]"));
        QVERIFY(document.text.contains("scale: \"n\""));
        QVERIFY(document.text.contains("anchors: [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119]]"));
    }

    void malformedOrExcessiveArchitecturesAreRejected_data()
    {
        QTest::addColumn<QJsonValue>("architecture");
        QTest::newRow("missing") << QJsonValue();
        QTest::newRow("array-not-dictionary") << QJsonValue(QJsonArray{1, 2});
        auto object = blueprint(); object.insert("nc", true);
        QTest::newRow("boolean-class-count") << QJsonValue(object);
        object = blueprint(); object.insert("unknown-field", 1);
        QTest::newRow("unknown-global-field") << QJsonValue(object);
        object = blueprint(); object.insert("scales", QJsonObject{{"n", QJsonArray{1, 1, false}}});
        QTest::newRow("invalid-scale") << QJsonValue(object);
        object = blueprint(); object.insert("anchors", QJsonArray{QJsonArray{10, 13, 2}});
        QTest::newRow("incomplete-anchor-pair") << QJsonValue(object);
        for (const auto &kind : {"self-reference", "bad-negative", "fractional-repeat", "expression-module", "non-list-args"})
        {
            object = blueprint(); auto backbone = object.value("backbone").toArray(); auto first = backbone[0].toArray();
            if (QString(kind) == "self-reference") first[0] = 0;
            else if (QString(kind) == "bad-negative") first[0] = -2;
            else if (QString(kind) == "fractional-repeat") first[1] = 1.5;
            else if (QString(kind) == "expression-module") first[2] = "__import__('os').system('bad')";
            else first[3] = QJsonObject{{"arg", 1}};
            backbone[0] = first; object.insert("backbone", backbone);
            QTest::newRow(kind) << QJsonValue(object);
        }
        object = blueprint(); object.insert("activation", QString(1025, 'x'));
        QTest::newRow("oversized-string") << QJsonValue(object);
        object = blueprint(); QJsonValue nested(1);
        for (int depth = 0; depth < 16; ++depth) nested = QJsonArray{nested};
        auto backbone = object.value("backbone").toArray(); auto first = backbone[0].toArray();
        first[3] = QJsonArray{nested}; backbone[0] = first; object.insert("backbone", backbone);
        QTest::newRow("excessive-depth") << QJsonValue(object);
        object = blueprint(); backbone = {};
        for (int index = 0; index < 2001; ++index) backbone.append(QJsonArray{-1, 1, "Conv", QJsonArray{8}});
        object.insert("backbone", backbone);
        QTest::newRow("excessive-layer-count") << QJsonValue(object);
        object = blueprint(); backbone = {};
        for (int index = 0; index < 500; ++index) backbone.append(QJsonArray{-1, 1, "Conv", QJsonArray{QString(1000, 'x')}});
        object.insert("backbone", backbone);
        QTest::newRow("excessive-document-bytes") << QJsonValue(object);
    }

    void malformedOrExcessiveArchitecturesAreRejected()
    {
        QFETCH(QJsonValue, architecture);
        const auto document = vision::modelYamlDocument(architecture);
        QVERIFY(!document.available);
        QVERIFY(document.text.isEmpty());
        QVERIFY(!document.notice.isEmpty());
    }

    void newFileExportNeverOverwritesSourceExistingOrSymlink()
    {
        QTemporaryDir area; QVERIFY(area.isValid());
        const auto document = vision::modelYamlDocument(blueprint()); QVERIFY(document.available);
        const QString source = area.path() + "/source.pt", output = area.path() + "/结构.yaml";
        QFile model(source); QVERIFY(model.open(QIODevice::WriteOnly)); QCOMPARE(model.write("original-model"), 14); model.close();
        QString error;
        QVERIFY2(vision::saveModelYaml(output, source, document, &error), qPrintable(error));
        QCOMPARE(bytes(output), document.text.toUtf8());
        QVERIFY(!vision::saveModelYaml(output, source, document, &error)); QVERIFY(!error.isEmpty());
        QVERIFY(!vision::saveModelYaml(source, source, document, &error));
        QVERIFY(!vision::saveModelYaml(area.path() + "/unsupported.pt", source, document, &error));
        QVERIFY(!vision::saveModelYaml(area.path() + "/missing/out.yaml", source, document, &error));
        QVERIFY(!vision::saveModelYaml(area.path() + "/empty.yaml", source, {}, &error));
#ifdef Q_OS_UNIX
        const QString link = area.path() + "/linked.yaml";
        QVERIFY(QFile::link(source, link));
        QVERIFY(!vision::saveModelYaml(link, source, document, &error));
#endif
        QCOMPARE(bytes(source), QByteArray("original-model"));
        QCOMPARE(bytes(output), document.text.toUtf8());
        QVERIFY(QDir(area.path()).entryList({".vision-studio-model-yaml-*"}, QDir::Files | QDir::Hidden).isEmpty());
    }

    void concurrentExportsOnlyOneWriterCreatesDestination()
    {
        QTemporaryDir area; QVERIFY(area.isValid());
        const QString output = area.path() + "/race.yaml";
        std::atomic_int winners{0}, failures{0};
        std::vector<std::thread> writers;
        QStringList candidates;
        for (int index = 0; index < 8; ++index)
        {
            auto object = blueprint(); object.insert("nc", index + 1);
            const auto document = vision::modelYamlDocument(object); QVERIFY(document.available);
            candidates.append(document.text);
            writers.emplace_back([output, document, &winners, &failures] {
                QString error;
                if (vision::saveModelYaml(output, {}, document, &error)) ++winners;
                else if (!error.isEmpty()) ++failures;
            });
        }
        for (auto &writer : writers) writer.join();
        QCOMPARE(winners.load(), 1); QCOMPARE(failures.load(), 7);
        QVERIFY(candidates.contains(QString::fromUtf8(bytes(output))));
        QVERIFY(QDir(area.path()).entryList({".vision-studio-model-yaml-*"}, QDir::Files | QDir::Hidden).isEmpty());
        qInfo() << "model_yaml_atomic_export_writers" << 8 << "created" << winners.load();
    }
};

QTEST_GUILESS_MAIN(ModelYamlTests)
#include "model_yaml_tests.moc"
