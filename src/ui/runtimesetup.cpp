// SPDX-License-Identifier: AGPL-3.0-only
#include "runtimesetup.h"
#include "core/cpuruntime.h"
#include <QApplication>
#include <QNetworkRequest>
#include <QUrl>
#include <QWidget>
#include <utility>
#include <QCryptographicHash>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLabel>
#include <QLockFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QPointer>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSaveFile>
#include <QStorageInfo>
#include <QNetworkProxyFactory>
#include <QNetworkProxyQuery>
#include <QTimer>
#include <QUuid>
#include <QVBoxLayout>
#include <memory>
#ifdef Q_OS_WIN
#include <qt_windows.h>
#endif

namespace ui {
namespace {
class RuntimeDialog;
QPointer<QDialog> active;
QList<std::function<void()>> retries;

class RuntimeDialog final : public QDialog
{
    QJsonObject descriptor = vision::cpuRuntimeDescriptor();
    QString destination = vision::cpuRuntimeDirectory(descriptor);
    QString archivePath, stage;
    QLabel *message;
    QProgressBar *progress;
    QPushButton *prepare;
    QNetworkAccessManager network{this};
    QPointer<QNetworkReply> reply;
    QProcess process{this};
    QFile archive;
    QCryptographicHash hash{QCryptographicHash::Sha256};
    std::unique_ptr<QLockFile> lock;
    qint64 received = 0;
    bool busy = false, cancelled = false, waiting = false;
    QTimer poll;
#ifdef Q_OS_WIN
    HANDLE childJob = nullptr;
#endif

    void stopChildren()
    {
#ifdef Q_OS_WIN
        if (childJob) { CloseHandle(childJob); childJob = nullptr; }
#endif
        if (process.state() != QProcess::NotRunning) { process.kill(); process.waitForFinished(2000); }
    }
    void removeStage()
    {
        if (!stage.isEmpty() && QFileInfo(stage).absolutePath() == QDir(vision::cpuRuntimeCacheRoot()).absolutePath()
            && QFileInfo(stage).fileName().startsWith(".staging-") && !QFileInfo(stage).isSymLink() && !QFileInfo(stage).isJunction())
            QDir(stage).removeRecursively();
        stage.clear();
    }

    void fail(const QString &error)
    {
        busy = false; waiting = false; poll.stop();
        if (reply) { reply->disconnect(this); reply->abort(); reply->deleteLater(); reply = nullptr; }
        archive.close();
        stopChildren(); removeStage();
        lock.reset();
        message->setText(error + "\n现有运行环境和个人文件保持保留，可点击重试。");
        progress->setRange(0, 100); progress->setValue(0);
        prepare->setEnabled(true); prepare->setText("重试");
    }
    void succeed()
    {
        busy = false; poll.stop(); archive.close(); lock.reset();
        QFile::remove(archivePath);
        accept();
    }
    void extract()
    {
        message->setText("下载校验通过，正在解压并验证运行组件…");
        progress->setRange(0, 0);
        stage = QDir(vision::cpuRuntimeCacheRoot()).filePath(".staging-" + QUuid::createUuid().toString(QUuid::WithoutBraces));
#ifdef Q_OS_WIN
        childJob = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!childJob || !SetInformationJobObject(childJob, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            fail("无法建立解压子进程保护，已停止准备。"); return;
        }
#endif
        process.setProgram(qEnvironmentVariable("SystemRoot", "C:/Windows") + "/System32/WindowsPowerShell/v1.0/powershell.exe");
        process.setArguments({"-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
            vision::packageRoot() + "/scripts/prepare_cpu_runtime.ps1", "-Archive", archivePath,
            "-Destination", stage, "-ExpectedSha256", descriptor.value("sha256").toString(),
            "-ExpectedBytes", QString::number(qint64(descriptor.value("bytes").toDouble()))});
        process.start();
    }
    void consume()
    {
        if (!reply || !busy) return;
        const auto data = reply->readAll();
        received += data.size();
        if (received > qint64(descriptor.value("bytes").toDouble()) || archive.write(data) != data.size()) {
            fail("下载文件超出清单大小或磁盘写入失败。"); return;
        }
        hash.addData(data);
    }
public:
    explicit RuntimeDialog(QWidget *parent) : QDialog(parent)
    {
        setObjectName("cpuRuntimeSetupDialog"); setWindowTitle("准备运行环境");
        setWindowModality(Qt::ApplicationModal); setAttribute(Qt::WA_DeleteOnClose);
        resize(540, 240);
        auto *layout = new QVBoxLayout(this);
        message = new QLabel(QString("PT 推理、模型转换、Netron 视图和数据集评估需要共用的 CPU 运行环境。\n"
            "需下载 %1 MiB，完成后可离线使用。此操作不安装显卡驱动。")
            .arg(descriptor.value("bytes").toDouble() / 1048576., 0, 'f', 1));
        message->setWordWrap(true); message->setTextFormat(Qt::PlainText); layout->addWidget(message);
        progress = new QProgressBar; progress->setRange(0, 100); layout->addWidget(progress);
        prepare = new QPushButton("下载并准备"); prepare->setObjectName("prepareCpuRuntimeButton"); layout->addWidget(prepare);
        auto *cancel = new QPushButton("取消"); cancel->setObjectName("cancelCpuRuntimeButton"); layout->addWidget(cancel);
        connect(prepare, &QPushButton::clicked, this, [this] { start(); });
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
#ifdef Q_OS_WIN
        connect(&process, &QProcess::started, this, [this] {
            const HANDLE child = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, DWORD(process.processId()));
            const bool attached = child && childJob && AssignProcessToJobObject(childJob, child);
            if (child) CloseHandle(child);
            if (!attached) fail("无法保护运行环境解压任务，已停止准备。");
        });
#endif
        poll.setInterval(500);
        connect(&poll, &QTimer::timeout, this, [this] {
            if (!vision::cpuPython().isEmpty()) { succeed(); return; }
            if (waiting && lock && lock->tryLock(0)) { waiting = false; poll.stop(); download(); }
        });
        connect(&process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
            if (!cancelled && busy && e == QProcess::FailedToStart) fail("无法启动 Windows 运行环境解压工具。");
        });
        connect(&process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) {
            if (cancelled || !busy) return;
            stopChildren();
            if (code || status != QProcess::NormalExit) {
                fail("运行环境验证失败：" + QString::fromUtf8(process.readAllStandardError()).right(1600)); return;
            }
            QSaveFile ready(stage + "/ready.json");
            if (!ready.open(QIODevice::WriteOnly) || ready.write(QJsonDocument(descriptor).toJson()) < 0 || !ready.commit()) {
                fail("无法写入运行环境完成标记。"); return;
            }
            // Never replace an existing cache directory, including unknown or modified files.
            if (QFileInfo::exists(destination) || !QDir().rename(stage, destination)) {
                fail("运行环境缓存目录已存在或无法发布；请检查目录权限，现有文件已保留。"); return;
            }
            stage.clear(); succeed();
        });
    }
    ~RuntimeDialog() override
    {
        cancelled = true; busy = false;
        if (reply) { reply->disconnect(this); reply->abort(); }
        process.disconnect(this);
        stopChildren();
        archive.close();
        // Only the randomly named staging folder created by this operation is removable.
        removeStage();
    }
    void start()
    {
        if (busy) return;
        if (!vision::cpuPython().isEmpty()) { succeed(); return; }
        const QUrl url(descriptor.value("url").toString());
        if (destination.isEmpty() || descriptor.value("schema_version").toInt() != 1 ||
            descriptor.value("bytes").toDouble() <= 0 ||
            !(url.scheme() == "https" || (url.scheme() == "http" && url.host() == "127.0.0.1"))) {
            fail("安装包缺少有效的运行环境下载清单，请重新下载安装包。"); return;
        }
        if (!QDir().mkpath(vision::cpuRuntimeCacheRoot()) || QFileInfo(vision::cpuRuntimeCacheRoot()).isSymLink() ||
            QFileInfo(vision::cpuRuntimeCacheRoot()).isJunction()) {
            fail("无法建立安全的用户运行环境缓存目录。"); return;
        }
        const QStorageInfo storage(vision::cpuRuntimeCacheRoot());
        const qint64 required = qint64(descriptor.value("bytes").toDouble()) +
            qint64(descriptor.value("uncompressed_bytes").toDouble(3221225472.));
        if (storage.isValid() && storage.bytesAvailable() < required + 256 * 1024 * 1024) {
            fail("用户数据盘空间不足，请释放空间后重试。"); return;
        }
        busy = true; prepare->setEnabled(false);
        lock = std::make_unique<QLockFile>(destination + ".lock"); lock->setStaleLockTime(0);
        if (!lock->tryLock(0)) {
            waiting = true; message->setText("另一个窗口正在准备同一运行环境，正在等待…");
            progress->setRange(0, 0); poll.start(); return;
        }
        download();
    }
    void download()
    {
        received = 0; hash.reset(); progress->setRange(0, 100); progress->setValue(0);
        archivePath = destination + ".download";
        archive.setFileName(archivePath);
        if (QFileInfo(archivePath).isSymLink() || !archive.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            fail("无法写入运行环境下载缓存。"); return;
        }
        message->setText("正在下载 CPU 运行环境，可随时取消…");
        QNetworkRequest request(QUrl(descriptor.value("url").toString()));
        request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
        request.setTransferTimeout(60000);
        const auto url = request.url();
        network.setProxy(url.host() == "127.0.0.1" ? QNetworkProxy(QNetworkProxy::NoProxy) :
            QNetworkProxyFactory::systemProxyForQuery(QNetworkProxyQuery(url)).value(0, QNetworkProxy(QNetworkProxy::NoProxy)));
        reply = network.get(request);
        connect(reply, &QNetworkReply::readyRead, this, [this] { consume(); });
        connect(reply, &QNetworkReply::downloadProgress, this, [this](qint64 value, qint64) {
            progress->setValue(qMin(100, int(100. * value / descriptor.value("bytes").toDouble())));
        });
        connect(reply, &QNetworkReply::finished, this, [this] {
            if (!busy || !reply) return;
            consume(); if (!busy || !reply) return;
            const auto error = reply->error(); const auto detail = reply->errorString();
            reply->deleteLater(); reply = nullptr; archive.close();
            if (error != QNetworkReply::NoError) { fail("下载失败：" + detail); return; }
            if (received != qint64(descriptor.value("bytes").toDouble()) ||
                QString::fromLatin1(hash.result().toHex()) != descriptor.value("sha256").toString()) {
                fail("下载文件大小或 SHA256 校验失败，未使用该文件。"); return;
            }
            extract();
        });
    }
};
}
bool ensureCpuRuntime(QWidget *parent, std::function<void()> retry)
{
    if (!vision::cpuRuntimeDownloadNeeded()) return true;
    const QPointer<QWidget> owner(parent);
    retries.append([owner, retry = std::move(retry)] { if (owner) retry(); });
    if (active) { active->raise(); active->activateWindow(); return false; }
    auto *dialog = new RuntimeDialog(parent); active = dialog;
    QObject::connect(dialog, &QDialog::finished, dialog, [](int result) {
        active = nullptr; const auto pending = std::exchange(retries, {});
        if (result == QDialog::Accepted) for (const auto &retry : pending) QTimer::singleShot(0, qApp, retry);
    });
    dialog->open(); return false;
}
} // namespace ui
