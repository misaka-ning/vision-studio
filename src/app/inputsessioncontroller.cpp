// SPDX-License-Identifier: AGPL-3.0-only
#include "inputsessioncontroller.h"
#include "asyncthreadretirement.h"
#include "core/evaluationworker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QImageReader>
#include <QSet>
#include <QThread>
#include <QTimer>
#include <utility>

namespace vision::app
{
namespace
{
struct ImageCheck
{
    QStringList files;
    QString error;
};
QString canonicalFile(const QString &path)
{
    const QFileInfo info(path);
    const QString canonical = info.canonicalFilePath();
    return canonical.isEmpty() ? info.absoluteFilePath() : canonical;
}
}

InputSessionController::InputSessionController(QString projectRoot, QString sessionFilePath, QObject *parent)
    : QObject(parent), projectRoot_(std::move(projectRoot)), sessionFilePath_(std::move(sessionFilePath))
{
    qRegisterMetaType<InputSessionSnapshot>();
    populationTimer_ = new QTimer(this);
    populationTimer_->setSingleShot(true);
    populationTimer_->setInterval(0);
    connect(populationTimer_, &QTimer::timeout, this, &InputSessionController::appendBatch);
}

InputSessionController::~InputSessionController()
{
    stopWorkers();
    // Each thread captures only copied input and a shared cancellation token.
    // Normal window closure first waits asynchronously for shutdownReady.
}

void InputSessionController::updatePhase(InputPhase phase)
{
    state_.phase = phase;
    state_.importing = phase != InputPhase::Idle;
    state_.queuePopulating = phase == InputPhase::Projecting || phase == InputPhase::WaitingForProjection;
    emit stateChanged();
}

void InputSessionController::noteUserInteraction()
{
    state_.userInteracted = true;
    ++state_.restoreGeneration;
}

void InputSessionController::setRestoreEnabled(bool enabled)
{
    restoreEnabled_ = enabled;
    if (!enabled) ++state_.restoreGeneration;
}

void InputSessionController::setRestoreGuard(std::function<bool()> guard)
{
    restoreGuard_ = std::move(guard);
}

bool InputSessionController::beginOperation(InputPhase phase)
{
    if (shuttingDown_) return false;
    cancelOperation(true);
    // Cancelling an old transaction publishes state synchronously. Its owner
    // may close while handling that notification, before a new worker exists.
    if (shuttingDown_) return false;
    previous_ = state_;
    transactionActive_ = true;
    cancellation_ = std::make_shared<std::atomic_bool>(false);
    ++state_.importGeneration;
    updatePhase(phase);
    // The initial phase notification is another reentrant close boundary.
    return !shuttingDown_;
}

bool InputSessionController::loadDataset(const QString &source, const QString &split, const QString &metadata)
{
    noteUserInteraction();
    return beginDataset(source, split, metadata);
}

bool InputSessionController::beginDataset(const QString &source, const QString &split, const QString &metadata)
{
    if (shuttingDown_) return false;
    if (source.trimmed().isEmpty() || (split != "train" && split != "val" && split != "test"))
    {
        emit failed(QStringLiteral("请选择数据集目录或 YAML，以及有效的数据集合。"));
        return false;
    }
    if (!beginOperation(InputPhase::CheckingDataset)) return false;
    const quint64 generation = state_.importGeneration;
    const auto cancelled = cancellation_;
    const DatasetSession session{QFileInfo(source).absoluteFilePath(), split,
        metadata.isEmpty() ? QString() : QFileInfo(metadata).absoluteFilePath()};
    auto *worker = new EvaluationWorker;
    datasetWorker_ = worker;
    workers_.insert(worker, cancelled);
    worker->configureDataset(session.sourcePath, split, session.metadataPath, projectRoot_);
    connect(worker, &EvaluationWorker::progress, this,
        [this, generation, cancelled](int completed, int total, const QString &stage) {
            if (!shuttingDown_ && generation == state_.importGeneration && !cancelled->load())
                emit progress(completed, total, stage);
        });
    connect(worker, &EvaluationWorker::datasetReady, this,
        [this, generation, cancelled, session](const QJsonObject &manifest) {
            if (shuttingDown_ || generation != state_.importGeneration || cancelled->load()) return;
            if (!manifest.value("samples").isArray() || manifest.value("samples").toArray().isEmpty())
            {
                failOperation(QStringLiteral("数据集中没有有效图片。"));
                return;
            }
            auto remembered = session;
            remembered.split = manifest.value("split").toString(session.split);
            remembered.metadataPath = manifest.value("metadata_path").toString(session.metadataPath);
            beginProjection({}, manifest, false, remembered);
        });
    connect(worker, &EvaluationWorker::failed, this,
        [this, generation, cancelled](const QString &message) {
            if (!shuttingDown_ && generation == state_.importGeneration && !cancelled->load())
                failOperation(message);
        });
    AsyncThreadRetirement::watch(worker, this, [this, worker, generation, cancelled] {
        workers_.remove(worker);
        if (datasetWorker_ == worker) datasetWorker_ = nullptr;
        if (!shuttingDown_ && generation == state_.importGeneration && !cancelled->load() &&
            state_.phase == InputPhase::CheckingDataset)
            failOperation(QStringLiteral("数据集检查结束，但未返回有效清单。"));
        notifyShutdownIfReady();
    });
    worker->start();
    return true;
}

bool InputSessionController::requestImages(const QStringList &paths, bool append)
{
    noteUserInteraction();
    if (shuttingDown_) return false;
    if (paths.isEmpty())
    {
        emit failed(QStringLiteral("没有可导入的图片。"));
        return false;
    }
    if (!beginOperation(InputPhase::CheckingImages)) return false;
    const quint64 generation = state_.importGeneration;
    const auto cancelled = cancellation_;
    const auto result = std::make_shared<ImageCheck>();
    const QStringList existing = append ? state_.files : QStringList();
    auto *thread = QThread::create([paths, existing, cancelled, result] {
        QSet<QString> seen;
        for (const auto &path : existing) {
            if (cancelled->load()) return;
            seen.insert(canonicalFile(path));
        }
        for (const auto &path : paths)
        {
            if (cancelled->load()) return;
            const QFileInfo info(path);
            if (!info.isFile() || !info.isReadable())
            {
                result->error = QStringLiteral("图片不存在或不可读取：%1").arg(info.fileName());
                return;
            }
            const QString canonical = canonicalFile(path);
            if (seen.contains(canonical)) continue;
            QImageReader reader(canonical);
            if (!reader.canRead())
            {
                result->error = QStringLiteral("无法读取图片格式：%1").arg(info.fileName());
                return;
            }
            seen.insert(canonical);
            result->files.append(canonical);
        }
    });
    workers_.insert(thread, cancelled);
    AsyncThreadRetirement::watch(thread, this, [this, thread, generation, cancelled, result, append] {
        workers_.remove(thread);
        if (!shuttingDown_ && generation == state_.importGeneration && !cancelled->load())
        {
            if (!result->error.isEmpty()) failOperation(result->error);
            else if (result->files.isEmpty())
            {
                transactionActive_ = false;
                resetProjection();
                updatePhase(InputPhase::Idle);
                emit notice(QStringLiteral("图片已在当前队列中，没有重复添加。"));
                emit finished(false);
            }
            else beginProjection(result->files, {}, append);
        }
        notifyShutdownIfReady();
    });
    emit progress(0, paths.size(), QStringLiteral("正在后台检查图片…"));
    thread->start();
    return true;
}

bool InputSessionController::openDatasetSnapshot(const QJsonObject &manifest)
{
    noteUserInteraction();
    if (shuttingDown_) return false;
    if (!manifest.value("samples").isArray() || manifest.value("samples").toArray().isEmpty())
    {
        emit failed(QStringLiteral("历史数据集快照中没有有效图片。"));
        return false;
    }
    if (!beginOperation(InputPhase::Projecting)) return false;
    beginProjection({}, manifest, false);
    return true;
}

void InputSessionController::beginProjection(const QStringList &images, const QJsonObject &manifest,
                                           bool append, const DatasetSession &session)
{
    datasetProjection_ = !manifest.isEmpty();
    projectionStarted_ = true;
    appendProjection_ = append;
    pendingImages_ = images;
    pendingSamples_ = manifest.value("samples").toArray();
    pendingSession_ = session;
    pendingIndex_ = 0;
    pendingTotal_ = datasetProjection_ ? pendingSamples_.size() : images.size();
    ++state_.projectionGeneration;
    state_.sourceKind = SourceKind::Images;
    state_.streamPath.clear();
    if (!append)
    {
        state_.files.clear();
        state_.datasetSamples.clear();
        state_.datasetManifest = manifest;
        state_.selectedRow = -1;
        state_.hasSelection = false;
    }
    firstNewRow_ = state_.files.size();
    updatePhase(InputPhase::Projecting);
    if (!append)
    {
        emit queueReset({}, state_.projectionGeneration);
        emit selectionChanged(-1, false);
    }
    if (datasetProjection_) emit datasetReady(manifest);
    populationTimer_->start();
}

void InputSessionController::appendBatch()
{
    if (shuttingDown_ || !transactionActive_ || !projectionStarted_ || awaitingProjection_) return;
    QElapsedTimer budget;
    budget.start();
    QStringList batch;
    const int firstRow = state_.files.size();
    while (pendingIndex_ < pendingTotal_ && batch.size() < MaximumBatchItems &&
           (batch.isEmpty() || budget.elapsed() < BatchBudgetMilliseconds))
    {
        QString path;
        if (datasetProjection_)
        {
            const QJsonObject sample = pendingSamples_.at(pendingIndex_).toObject();
            path = sample.value("image").toString();
            if (path.isEmpty() || state_.datasetSamples.contains(path))
            {
                failOperation(QStringLiteral("数据集清单包含空路径或重复图片。"));
                return;
            }
            state_.datasetSamples.insert(path, sample);
        }
        else path = pendingImages_.at(pendingIndex_);
        state_.files.append(path);
        batch.append(path);
        ++pendingIndex_;
    }
    awaitingProjection_ = true;
    expectedProjectionCount_ = state_.files.size();
    updatePhase(InputPhase::WaitingForProjection);
    emit progress(pendingIndex_, pendingTotal_, QStringLiteral("正在建立输入队列…"));
    emit queueAppend(batch, firstRow, state_.projectionGeneration);
}

void InputSessionController::acknowledgeQueueProjection(quint64 generation, int projectedCount)
{
    if (shuttingDown_ || !transactionActive_ || !awaitingProjection_ ||
        generation != state_.projectionGeneration || projectedCount != expectedProjectionCount_)
        return;
    awaitingProjection_ = false;
    expectedProjectionCount_ = -1;
    if (pendingIndex_ == pendingTotal_) completeProjection();
    else
    {
        updatePhase(InputPhase::Projecting);
        populationTimer_->start();
    }
}

void InputSessionController::completeProjection()
{
    const bool remember = datasetProjection_;
    const DatasetSession session = pendingSession_;
    const int selected = firstNewRow_ < state_.files.size() ? firstNewRow_ : -1;
    transactionActive_ = false;
    resetProjection();
    state_.selectedRow = selected;
    state_.hasSelection = selected >= 0;
    updatePhase(InputPhase::Idle);
    emit selectionChanged(state_.selectedRow, state_.hasSelection);
    if (remember && !session.sourcePath.isEmpty() && !sessionFilePath_.isEmpty())
    {
        QString error;
        if (!DatasetSessionStore::save(sessionFilePath_, session, &error))
            emit notice(QStringLiteral("数据集已载入，但无法记住启动恢复路径：%1").arg(error));
    }
    emit finished(false);
}

void InputSessionController::resetProjection()
{
    populationTimer_->stop();
    pendingImages_.clear();
    pendingSamples_ = {};
    pendingSession_ = {};
    pendingIndex_ = pendingTotal_ = firstNewRow_ = 0;
    expectedProjectionCount_ = -1;
    datasetProjection_ = awaitingProjection_ = projectionStarted_ = appendProjection_ = false;
}

void InputSessionController::restorePreviousState()
{
    const bool changedQueue = projectionStarted_;
    const auto importGeneration = state_.importGeneration;
    const auto projectionGeneration = state_.projectionGeneration + (changedQueue ? 1 : 0);
    const auto restoreGeneration = state_.restoreGeneration;
    const bool restoreRequested = state_.restoreRequested, userInteracted = state_.userInteracted;
    state_ = previous_;
    state_.importGeneration = importGeneration;
    state_.projectionGeneration = projectionGeneration;
    state_.restoreGeneration = restoreGeneration;
    state_.restoreRequested = restoreRequested;
    state_.userInteracted = userInteracted;
    resetProjection();
    updatePhase(InputPhase::Idle);
    if (changedQueue)
    {
        emit queueReset(state_.files, state_.projectionGeneration);
        emit selectionChanged(state_.selectedRow, state_.hasSelection);
    }
}

void InputSessionController::failOperation(const QString &message)
{
    if (!transactionActive_) return;
    if (cancellation_) cancellation_->store(true);
    ++state_.importGeneration;
    transactionActive_ = false;
    restorePreviousState();
    emit failed(message);
    emit finished(false);
}

void InputSessionController::cancelOperation(bool notify)
{
    if (!transactionActive_) return;
    if (cancellation_) cancellation_->store(true);
    if (datasetWorker_) datasetWorker_->requestStop();
    ++state_.importGeneration;
    transactionActive_ = false;
    // Match the established UI contract: validation has not changed the old
    // session, and an append can be undone. Once a replacement has projected
    // rows, cancellation leaves an empty image source instead of resurrecting
    // the replaced source. The last successfully saved startup record remains.
    if (projectionStarted_ && !appendProjection_) previous_ = {};
    restorePreviousState();
    if (notify) emit finished(true);
}

void InputSessionController::cancel()
{
    noteUserInteraction();
    cancelOperation(true);
}

void InputSessionController::setSelection(int row, bool selected)
{
    if (shuttingDown_) return;
    if (row < 0 || row >= state_.files.size()) { row = -1; selected = false; }
    if (state_.selectedRow == row && state_.hasSelection == selected) return;
    state_.selectedRow = row;
    state_.hasSelection = selected;
    emit selectionChanged(row, selected);
    emit stateChanged();
}

bool InputSessionController::selectVideo(const QString &path)
{
    noteUserInteraction();
    if (shuttingDown_) return false;
    if (!QFileInfo(path).isFile() || !QFileInfo(path).isReadable())
    {
        emit failed(QStringLiteral("视频不存在或不可读取。"));
        return false;
    }
    cancelOperation(true);
    clear();
    state_.sourceKind = SourceKind::Video;
    state_.streamPath = QFileInfo(path).absoluteFilePath();
    emit stateChanged();
    return true;
}

bool InputSessionController::selectCamera()
{
    if (shuttingDown_) return false;
    clear();
    state_.sourceKind = SourceKind::Camera;
    emit stateChanged();
    return true;
}

void InputSessionController::clear()
{
    noteUserInteraction();
    if (shuttingDown_) return;
    cancelOperation(true);
    state_.sourceKind = SourceKind::Images;
    state_.files.clear();
    state_.streamPath.clear();
    state_.datasetManifest = {};
    state_.datasetSamples.clear();
    state_.selectedRow = -1;
    state_.hasSelection = false;
    ++state_.importGeneration;
    ++state_.projectionGeneration;
    emit queueReset({}, state_.projectionGeneration);
    emit selectionChanged(-1, false);
    emit stateChanged();
}

QStringList InputSessionController::range(ImageRange mode, bool forEvaluation, QString *error) const
{
    if (error) error->clear();
    const auto reject = [error](const QString &message) {
        if (error) *error = message;
        return QStringList();
    };
    if (state_.importing) return reject(QStringLiteral("输入队列仍在载入，请稍候。"));
    if (state_.sourceKind != SourceKind::Images || state_.files.isEmpty())
        return reject(QStringLiteral("输入队列中没有图片。"));
    if (forEvaluation && (state_.datasetManifest.isEmpty() || state_.datasetSamples.isEmpty()))
        return reject(QStringLiteral("请先导入带标注的 YOLO 数据集。"));
    if (mode != ImageRange::All && (!state_.hasSelection || state_.selectedRow < 0 ||
                                    state_.selectedRow >= state_.files.size()))
        return reject(QStringLiteral("请先在输入队列中选中当前图片。"));
    if (forEvaluation && mode != ImageRange::All &&
        !state_.datasetSamples.contains(state_.files.at(state_.selectedRow)))
        return reject(QStringLiteral("临时图片没有数据集标注，不能作为评估范围起点。"));
    const int first = mode == ImageRange::All ? 0 : state_.selectedRow;
    const int end = mode == ImageRange::Current ? first + 1 : state_.files.size();
    QStringList result;
    for (int row = first; row < end; ++row)
        if (!forEvaluation || state_.datasetSamples.contains(state_.files.at(row)))
            result.append(state_.files.at(row));
    if (result.isEmpty()) return reject(QStringLiteral("当前范围中没有可评估的图片。"));
    return result;
}

void InputSessionController::restore(const QString &sessionPath)
{
    if (shuttingDown_ || state_.restoreRequested) return;
    state_.restoreRequested = true;
    const quint64 generation = state_.restoreGeneration;
    const QString path = sessionPath.isEmpty() ? sessionFilePath_ : sessionPath;
    QTimer::singleShot(0, this, [this, generation, path] {
        if (shuttingDown_ || !restoreEnabled_ || state_.userInteracted ||
            generation != state_.restoreGeneration || state_.importing ||
            state_.sourceKind != SourceKind::Images || !state_.files.isEmpty() ||
            !state_.datasetManifest.isEmpty() || (restoreGuard_ && !restoreGuard_())) return;
        const auto loaded = DatasetSessionStore::load(path);
        if (loaded.status == DatasetSessionStatus::Missing) return;
        if (loaded.status != DatasetSessionStatus::Loaded)
        {
            emit notice(QStringLiteral("未恢复上次数据集：%1").arg(loaded.error));
            return;
        }
        beginDataset(loaded.session.sourcePath, loaded.session.split, loaded.session.metadataPath);
    });
}

void InputSessionController::stopWorkers()
{
    for (auto it = workers_.cbegin(); it != workers_.cend(); ++it)
    {
        it.value()->store(true);
        if (auto *dataset = qobject_cast<EvaluationWorker *>(it.key())) dataset->requestStop();
    }
}

void InputSessionController::requestShutdown()
{
    if (shuttingDown_) return;
    shuttingDown_ = true;
    cancelOperation(false);
    ++state_.restoreGeneration;
    stopWorkers();
    notifyShutdownIfReady();
}

void InputSessionController::notifyShutdownIfReady()
{
    if (isShutdownReady() && !shutdownReadyEmitted_)
    {
        shutdownReadyEmitted_ = true;
        emit shutdownReady();
    }
}
} // namespace vision::app
