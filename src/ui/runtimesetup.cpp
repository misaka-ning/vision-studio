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
#include <QThread>
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
quint64 dialogGeneration = 0;
}

// The dialog may disappear while QProcess still owns a live child. This object
// remains alive until Qt has reaped it and the private staging cleanup has ended.
class RuntimeResources final : public QObject
{
    bool stopping_ = false;
    bool retiring_ = false;
    bool cleanupRequested_ = false;
    QThread *cleanup_ = nullptr;
    QList<std::function<void()>> stopped_;
    QTimer childPoll_;

    bool childrenExited()
    {
#ifdef Q_OS_WIN
        if (childJob) {
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
            if (!QueryInformationJobObject(childJob, JobObjectBasicAccountingInformation,
                    &accounting, sizeof(accounting), nullptr) || accounting.ActiveProcesses != 0) {
                childPoll_.start();
                return false;
            }
            CloseHandle(childJob);
            childJob = nullptr;
        }
#endif
        childPoll_.stop();
        return true;
    }

    void finishStop()
    {
        if (!stopping_ || process.state() != QProcess::NotRunning || cleanup_ || !childrenExited()) return;
        if (cleanupRequested_ && !stage.isEmpty()) {
            const auto directory = std::exchange(stage, {});
            const auto cacheRoot = vision::cpuRuntimeCacheRoot();
            auto ownedLock = std::move(lock);
            cleanup_ = QThread::create([directory, cacheRoot, ownedLock = std::move(ownedLock)] {
                const QFileInfo root(cacheRoot), info(directory);
                // Remove only this operation's randomly named staging folder.
                if (!root.isSymLink() && !root.isJunction()
                    && info.absolutePath() == QDir(cacheRoot).absolutePath()
                    && info.fileName().startsWith(".staging-")
                    && !info.isSymLink() && !info.isJunction())
                    QDir(directory).removeRecursively();
            });
            QObject::connect(cleanup_, &QThread::finished, cleanup_, &QObject::deleteLater);
            QObject::connect(cleanup_, &QObject::destroyed, this, [this] {
                cleanup_ = nullptr;
                finishStop();
            });
            cleanup_->start();
            return;
        }
        if (cleanupRequested_) lock.reset();
        cleanupRequested_ = false;
        stopping_ = false;
        const auto callbacks = std::exchange(stopped_, {});
        for (const auto &callback : callbacks) callback();
        if (retiring_) deleteLater();
    }

public:
    QProcess process{this};
    QString stage;
    std::unique_ptr<QLockFile> lock;
#ifdef Q_OS_WIN
    HANDLE childJob = nullptr;
#endif

    RuntimeResources()
    {
        setObjectName("cpuRuntimePreparation");
        runtimeSetup()->trackPreparation(this);
        childPoll_.setInterval(25);
        QObject::connect(&childPoll_, &QTimer::timeout, this, [this] { finishStop(); });
        QObject::connect(&process, &QProcess::started, this, [this] {
            if (stopping_) process.kill();
        });
        QObject::connect(&process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this] { finishStop(); });
        QObject::connect(&process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) finishStop();
        });
    }

    void afterChildrenExit(std::function<void()> stopped)
    {
        if (stopped) stopped_.append(std::move(stopped));
        stopping_ = true;
#ifdef Q_OS_WIN
        if (childJob) TerminateJobObject(childJob, 1);
#endif
        if (process.state() != QProcess::NotRunning) process.kill();
        finishStop();
    }

    void stopChildren(std::function<void()> stopped = {})
    {
        cleanupRequested_ = true;
        afterChildrenExit(std::move(stopped));
    }

    void retire()
    {
        retiring_ = true;
        stopChildren();
    }
};

namespace {

class RuntimeDialog final : public QDialog
{
    QJsonObject descriptor = vision::cpuRuntimeDescriptor();
    QString destination = vision::cpuRuntimeDirectory(descriptor);
    QString archivePath;
    QLabel *message;
    QProgressBar *progress;
    QPushButton *prepare;
    QNetworkAccessManager network{this};
    QPointer<QNetworkReply> reply;
    RuntimeResources *resources = new RuntimeResources;
    QFile archive;
    QCryptographicHash hash{QCryptographicHash::Sha256};
    qint64 received = 0;
    bool busy = false, cancelled = false, waiting = false;
    QTimer poll;

    void fail(const QString &error)
    {
        busy = false; waiting = false; poll.stop();
        if (reply) { reply->disconnect(this); reply->abort(); reply->deleteLater(); reply = nullptr; }
        archive.close();
        message->setText(error + "\n现有运行环境和个人文件保持保留，可点击重试。");
        progress->setRange(0, 100); progress->setValue(0);
        prepare->setEnabled(false); prepare->setText("重试");
        const QPointer<RuntimeDialog> self(this);
        resources->stopChildren([self] { if (self && !self->cancelled) self->prepare->setEnabled(true); });
    }
    void succeed()
    {
        busy = false; poll.stop(); archive.close(); resources->lock.reset();
        if (!archivePath.isEmpty()) QFile::remove(archivePath);
        accept();
    }
    void extract()
    {
        message->setText("下载校验通过，正在解压并验证运行组件…");
        progress->setRange(0, 0);
        resources->stage = QDir(vision::cpuRuntimeCacheRoot()).filePath(".staging-" + QUuid::createUuid().toString(QUuid::WithoutBraces));
#ifdef Q_OS_WIN
        resources->childJob = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!resources->childJob || !SetInformationJobObject(resources->childJob, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            fail("无法建立解压子进程保护，已停止准备。"); return;
        }
#endif
        auto &process = resources->process;
        process.setProgram(qEnvironmentVariable("SystemRoot", "C:/Windows") + "/System32/WindowsPowerShell/v1.0/powershell.exe");
        process.setArguments({"-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
            vision::packageRoot() + "/scripts/prepare_cpu_runtime.ps1", "-Archive", archivePath,
            "-Destination", resources->stage, "-ExpectedSha256", descriptor.value("sha256").toString(),
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
    void finishExtraction(int code, QProcess::ExitStatus status)
    {
        if (cancelled || !busy) return;
        if (code || status != QProcess::NormalExit) {
            fail("运行环境验证失败：" + QString::fromUtf8(resources->process.readAllStandardError()).right(1600)); return;
        }
        QSaveFile ready(resources->stage + "/ready.json");
        if (!ready.open(QIODevice::WriteOnly) || ready.write(QJsonDocument(descriptor).toJson()) < 0 || !ready.commit()) {
            fail("无法写入运行环境完成标记。"); return;
        }
        // Never replace an existing cache directory, including unknown or modified files.
        if (QFileInfo::exists(destination) || !QDir().rename(resources->stage, destination)) {
            fail("运行环境缓存目录已存在或无法发布；请检查目录权限，现有文件已保留。"); return;
        }
        resources->stage.clear(); succeed();
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
        auto &process = resources->process;
#ifdef Q_OS_WIN
        connect(&process, &QProcess::started, this, [this] {
            if (cancelled || !busy) return;
            const HANDLE child = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, DWORD(resources->process.processId()));
            const bool attached = child && resources->childJob && AssignProcessToJobObject(resources->childJob, child);
            if (child) CloseHandle(child);
            if (!attached) fail("无法保护运行环境解压任务，已停止准备。");
        });
#endif
        poll.setInterval(500);
        connect(&poll, &QTimer::timeout, this, [this] {
            if (!vision::cpuPython().isEmpty()) { succeed(); return; }
            if (waiting && resources->lock && resources->lock->tryLock(0)) { waiting = false; poll.stop(); download(); }
        });
        connect(&process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
            if (!cancelled && busy && e == QProcess::FailedToStart) fail("无法启动 Windows 运行环境解压工具。");
        });
        connect(&process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) {
            if (cancelled || !busy) return;
            const QPointer<RuntimeDialog> self(this);
            resources->afterChildrenExit([self, code, status] { if (self) self->finishExtraction(code, status); });
        });
    }
    ~RuntimeDialog() override
    {
        cancelled = true; busy = false;
        if (reply) { reply->disconnect(this); reply->abort(); }
        resources->process.disconnect(this);
        archive.close();
        resources->retire();
    }
    void done(int result) override
    {
        if (result != QDialog::Accepted) {
            cancelled = true; busy = false; waiting = false; poll.stop();
            if (reply) { reply->disconnect(this); reply->abort(); reply->deleteLater(); reply = nullptr; }
            archive.close();
            resources->stopChildren();
        }
        QDialog::done(result);
    }
    void start()
    {
        if (busy || cancelled) return;
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
        resources->lock = std::make_unique<QLockFile>(destination + ".lock"); resources->lock->setStaleLockTime(0);
        if (!resources->lock->tryLock(0)) {
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

RuntimeSetup::RuntimeSetup(QObject *parent) : QObject(parent)
{
    setObjectName("cpuRuntimeSetup");
}

RuntimeSetup::~RuntimeSetup()
{
    requestShutdown();
}

RuntimeSetup *runtimeSetup()
{
    static QPointer<RuntimeSetup> instance;
    if (!instance) instance = new RuntimeSetup(qApp);
    return instance;
}

bool RuntimeSetup::isShutdownReady() const
{
    return preparations_ == 0;
}

void RuntimeSetup::trackPreparation(QObject *preparation)
{
    ++preparations_;
    connect(preparation, &QObject::destroyed, this, [this] {
        --preparations_;
        maybeEmitShutdownReady();
    });
}

void RuntimeSetup::maybeEmitShutdownReady()
{
    if (!shuttingDown_ || shutdownEmitted_ || !isShutdownReady()) return;
    shutdownEmitted_ = true;
    emit shutdownReady();
}

void RuntimeSetup::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    cancelPendingPreparations();
    QTimer::singleShot(0, this, [this] { maybeEmitShutdownReady(); });
}

void RuntimeSetup::cancelPendingPreparations()
{
    // A window can close and be recreated in the same QApplication. Cancelling
    // that window's pending preparation must not permanently disable the runtime.
    ++cancellationGeneration_;
    retries.clear();
    if (active) active->reject();
}

bool ensureCpuRuntime(QWidget *parent, std::function<void()> retry)
{
    auto *setup = runtimeSetup();
    if (setup->shuttingDown_) return false;
    if (!vision::cpuRuntimeDownloadNeeded()) return true;
    const QPointer<QWidget> owner(parent);
    retries.append([owner, retry = std::move(retry)] { if (owner) retry(); });
    if (active) { active->raise(); active->activateWindow(); return false; }
    auto *dialog = new RuntimeDialog(parent); active = dialog;
    const auto generation = ++dialogGeneration;
    QObject::connect(dialog, &QDialog::finished, dialog, [](int result) {
        active = nullptr; const auto pending = std::exchange(retries, {});
        if (result == QDialog::Accepted) {
            const QPointer<RuntimeSetup> setup(runtimeSetup());
            const auto cancellationGeneration = setup->cancellationGeneration_;
            for (const auto &retry : pending) QTimer::singleShot(0, qApp, [setup, retry, cancellationGeneration] {
                if (setup && !setup->shuttingDown_ && setup->cancellationGeneration_ == cancellationGeneration) retry();
            });
        }
    });
    QObject::connect(dialog, &QObject::destroyed, setup, [generation] {
        if (dialogGeneration == generation) { active = nullptr; retries.clear(); }
    });
    dialog->open(); return false;
}
} // namespace ui
