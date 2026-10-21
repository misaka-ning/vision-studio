// SPDX-License-Identifier: AGPL-3.0-only
#include "core/datasetsessionstore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QThread>
#include <QtTest>
#include <atomic>
#include <memory>

namespace
{
QByteArray contents(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}
bool writeFile(const QString &path, const QByteArray &data)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(data) == data.size();
}
vision::DatasetSession session(const QString &root, const QString &split = QStringLiteral("val"))
{
    return {root + QStringLiteral("/数据集 with spaces/data.yaml"), split,
            root + QStringLiteral("/人工审核/review.csv")};
}
}

class DatasetSessionTests final : public QObject
{
    Q_OBJECT
private slots:
    void missingFileDoesNotCreateAnything()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/not-created/dataset-session.json";
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::Missing);
        QVERIFY(!QFileInfo::exists(area.path() + "/not-created"));
    }

    void roundTripPreservesOriginalSourceAndSplit_data()
    {
        QTest::addColumn<QString>("split");
        for (const auto &split : {"train", "val", "test"})
            QTest::newRow(split) << QString::fromLatin1(split);
    }
    void roundTripPreservesOriginalSourceAndSplit()
    {
        QFETCH(QString, split);
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        auto expected = session(area.path(), split);
        QString error;
        QVERIFY2(vision::DatasetSessionStore::save(path, expected, &error), qPrintable(error));
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::Loaded);
        QCOMPARE(loaded.session.sourcePath, expected.sourcePath);
        QCOMPARE(loaded.session.split, expected.split);
        QCOMPARE(loaded.session.metadataPath, expected.metadataPath);
        const auto json = QJsonDocument::fromJson(contents(path)).object();
        QCOMPARE(json.value("schema_version").toInt(), 1);
        QCOMPARE(json.value("source_path").toString(), expected.sourcePath);
        QCOMPARE(json.value("split").toString(), split);
#ifdef Q_OS_UNIX
        const auto permissions = QFile::permissions(path);
        QVERIFY(permissions.testFlag(QFileDevice::ReadOwner));
        QVERIFY(permissions.testFlag(QFileDevice::WriteOwner));
        QVERIFY(!(permissions & (QFileDevice::ReadGroup | QFileDevice::WriteGroup | QFileDevice::ExeGroup |
                                 QFileDevice::ReadOther | QFileDevice::WriteOther | QFileDevice::ExeOther)));
#endif
    }

    void optionalMetadataAndUnavailableSourceAreNotFilesystemValidation()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        const QString source = area.path() + "/currently-unmounted dataset";
        QVERIFY(!QFileInfo::exists(source));
        const QJsonObject json{{"schema_version", 1}, {"source_path", source}, {"split", "test"}};
        QVERIFY(writeFile(path, QJsonDocument(json).toJson()));
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::Loaded);
        QCOMPARE(loaded.session.sourcePath, source);
        QCOMPARE(loaded.session.split, QString("test"));
        QVERIFY(loaded.session.metadataPath.isEmpty());
    }

    void nonFileRecordReportsIoErrorWithoutDeletingIt()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        QVERIFY(QDir().mkpath(path));
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::IoError);
        QVERIFY(!loaded.error.isEmpty());
        QVERIFY(QFileInfo(path).isDir());
    }

    void invalidJsonIsPreserved_data()
    {
        QTest::addColumn<QByteArray>("data");
        QTest::newRow("truncated") << QByteArray("{\"schema_version\":1,");
        QTest::newRow("array") << QByteArray("[]");
        QTest::newRow("unknown-schema") << QByteArray("{\"schema_version\":2,\"source_path\":\"/data\",\"split\":\"val\"}");
        QTest::newRow("wrong-field-type") << QByteArray("{\"schema_version\":1,\"source_path\":42,\"split\":\"val\"}");
        QTest::newRow("relative-source") << QByteArray("{\"schema_version\":1,\"source_path\":\"relative/data.yaml\",\"split\":\"val\"}");
        QTest::newRow("bad-split") << QByteArray("{\"schema_version\":1,\"source_path\":\"/data\",\"split\":\"validation\"}");
        QTest::newRow("relative-metadata") << QByteArray("{\"schema_version\":1,\"source_path\":\"/data\",\"split\":\"val\",\"metadata_path\":\"review.csv\"}");
    }
    void invalidJsonIsPreserved()
    {
        QFETCH(QByteArray, data);
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        QVERIFY(writeFile(path, data));
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::Invalid);
        QVERIFY(!loaded.error.isEmpty());
        QCOMPARE(contents(path), data);
    }

    void oversizedInputIsRejectedWithoutChangingIt()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        const QByteArray data(64 * 1024 + 1, ' ');
        QVERIFY(writeFile(path, data));
        const auto loaded = vision::DatasetSessionStore::load(path);
        QCOMPARE(loaded.status, vision::DatasetSessionStatus::Invalid);
        QVERIFY(!loaded.error.isEmpty());
        QCOMPARE(contents(path), data);
    }

    void invalidSaveDoesNotReplaceValidRecord()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        const auto valid = session(area.path());
        QString error;
        QVERIFY2(vision::DatasetSessionStore::save(path, valid, &error), qPrintable(error));
        const QByteArray original = contents(path);
        for (int invalid = 0; invalid < 3; ++invalid) {
            auto value = valid;
            if (invalid == 0) value.sourcePath = "relative/data.yaml";
            else if (invalid == 1) value.split = "validation";
            else value.metadataPath = "relative.csv";
            error.clear();
            QVERIFY(!vision::DatasetSessionStore::save(path, value, &error));
            QVERIFY(!error.isEmpty());
            QCOMPARE(contents(path), original);
        }
    }

    void unwritableParentDoesNotFallBackToTruncatingExistingFile()
    {
#ifndef Q_OS_UNIX
        QSKIP("Unix permission failure injection; Windows file-sharing acceptance requires a Windows runner.");
#else
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString directory = area.path() + "/locked";
        QVERIFY(QDir().mkpath(directory));
        const QString path = directory + "/dataset-session.json";
        QString error;
        QVERIFY2(vision::DatasetSessionStore::save(path, session(area.path()), &error), qPrintable(error));
        const QByteArray original = contents(path);
        struct RestorePermissions {
            QString path;
            ~RestorePermissions() { QFile::setPermissions(path, QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner); }
        } restore{directory};
        QVERIFY(QFile::setPermissions(directory, QFileDevice::ReadOwner | QFileDevice::ExeOwner));
        QFile probe(directory + "/write-probe");
        if (probe.open(QIODevice::WriteOnly)) {
            probe.close(); probe.remove();
            QSKIP("Runner bypasses directory write permissions; this failure injection cannot be exercised here.");
        }
        auto replacement = session(area.path(), "train");
        QVERIFY(!vision::DatasetSessionStore::save(path, replacement, &error));
        QVERIFY(!error.isEmpty());
        QCOMPARE(contents(path), original);
        QCOMPARE(vision::DatasetSessionStore::load(path).status, vision::DatasetSessionStatus::Loaded);
#endif
    }

    void concurrentReadersObserveCompleteOldOrNewRecords()
    {
        QTemporaryDir area;
        QVERIFY(area.isValid());
        const QString path = area.path() + "/dataset-session.json";
        auto first = session(area.path(), "train");
        auto second = session(area.path(), "test");
        first.sourcePath = area.path() + "/first/data.yaml";
        second.sourcePath = area.path() + "/second/data.yaml";
        QString error;
        QVERIFY2(vision::DatasetSessionStore::save(path, first, &error), qPrintable(error));
        std::atomic_bool stop{false}, started{false}, invalid{false};
        std::atomic_int reads{0};
        std::unique_ptr<QThread> reader(QThread::create([&] {
            started = true;
            while (!stop.load()) {
                const auto loaded = vision::DatasetSessionStore::load(path);
                const bool oldRecord = loaded.session.sourcePath == first.sourcePath && loaded.session.split == first.split;
                const bool newRecord = loaded.session.sourcePath == second.sourcePath && loaded.session.split == second.split;
                if (loaded.status != vision::DatasetSessionStatus::Loaded || (!oldRecord && !newRecord) ||
                    loaded.session.metadataPath != first.metadataPath) invalid = true;
                ++reads;
            }
        }));
        reader->start();
        while (!started.load()) QThread::yieldCurrentThread();
        bool allSaved = true;
        for (int index = 0; index < 120; ++index)
            if (!vision::DatasetSessionStore::save(path, index % 2 ? first : second, &error)) { allSaved = false; break; }
        stop = true;
        reader->wait();
        QVERIFY2(allSaved, qPrintable(error));
        QVERIFY(reads.load() > 0);
        QVERIFY(!invalid.load());
        qInfo() << "dataset_session_atomic_read_observations" << reads.load() << "committed_replacements" << 120;
    }
};

QTEST_GUILESS_MAIN(DatasetSessionTests)
#include "dataset_session_tests.moc"
