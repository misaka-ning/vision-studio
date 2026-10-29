// SPDX-License-Identifier: AGPL-3.0-only
#include "recordingspage.h"
#include "pagehelpers.h"
#include "ui/imagecanvas.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSplitter>
#include <QTableWidget>
#include <QVBoxLayout>
using namespace vision::ui::pages;

RecordingsPage::RecordingsPage(const QString &recordingsDirectory, QWidget *parent) : QWidget(parent)
{
    setObjectName("recordingsPage");
    auto *layout = new QVBoxLayout(this); layout->setContentsMargins(0, 0, 0, 0); layout->setSpacing(14);
    auto *actions = new QHBoxLayout;
    controls_.count = text("本地录像 · 0 个文件", "sectionTitle"); actions->addWidget(controls_.count, 1);
    controls_.refresh = button("刷新", "history"); controls_.refresh->setObjectName("refreshRecordingsButton");
    controls_.folder = button("打开文件夹", "folder"); controls_.folder->setObjectName("openRecordingsFolderButton");
    actions->addWidget(controls_.refresh); actions->addWidget(controls_.folder); layout->addLayout(actions);
    connect(controls_.refresh, &QPushButton::clicked, this, &RecordingsPage::refreshRequested);
    connect(controls_.folder, &QPushButton::clicked, this, &RecordingsPage::openFolderRequested);
    auto *split = new QSplitter(Qt::Horizontal);
    auto *list = card(); auto *listLayout = new QVBoxLayout(list); listLayout->setContentsMargins(12, 12, 12, 12);
    controls_.table = new QTableWidget(0, 4); controls_.table->setObjectName("recordingsTable");
    controls_.table->setHorizontalHeaderLabels({"录像 / 时间", "时长", "分辨率", "大小"}); tableStyle(controls_.table);
    controls_.table->setWordWrap(false); controls_.table->verticalHeader()->setDefaultSectionSize(58);
    controls_.table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    controls_.table->setColumnWidth(1, 72); controls_.table->setColumnWidth(2, 96); controls_.table->setColumnWidth(3, 72);
    listLayout->addWidget(controls_.table); split->addWidget(list);
    connect(controls_.table, &QTableWidget::itemSelectionChanged, this, [this] {
        const int row = controls_.table->currentRow(); const auto *item = row < 0 ? nullptr : controls_.table->item(row, 0);
        emit selectionRequested(item ? item->data(Qt::UserRole).toString() : QString());
    });
    auto *viewer = card(); auto *viewerLayout = new QVBoxLayout(viewer);
    viewerLayout->setContentsMargins(12, 12, 12, 12); viewerLayout->setSpacing(12);
    viewerLayout->addWidget(text("录像预览", "sectionTitle"));
    controls_.canvas = new ImageCanvas; controls_.canvas->setObjectName("recordingCanvas");
    controls_.canvas->setAcceptDrops(false); controls_.canvas->setBoxesVisible(false); controls_.canvas->setLabelsVisible(false);
    viewerLayout->addWidget(controls_.canvas, 1);
    controls_.details = text("在工作台开始视频或摄像头检测，再点击「开始录制」。", "muted");
    controls_.details->setWordWrap(true); viewerLayout->addWidget(controls_.details);
    auto *playActions = new QHBoxLayout;
    controls_.play = button("播放录像", "play"); controls_.play->setObjectName("recordingPlayButton"); controls_.play->setEnabled(false);
    controls_.exportVideo = button("导出录像", "export"); controls_.exportVideo->setObjectName("exportRecordingButton"); controls_.exportVideo->setEnabled(false);
    connect(controls_.play, &QPushButton::clicked, this, &RecordingsPage::playRequested);
    connect(controls_.exportVideo, &QPushButton::clicked, this, &RecordingsPage::exportRequested);
    playActions->addWidget(controls_.play); playActions->addWidget(controls_.exportVideo); playActions->addStretch();
    viewerLayout->addLayout(playActions); split->addWidget(viewer);
    split->setStretchFactor(0, 1); split->setStretchFactor(1, 1); layout->addWidget(split, 1);
    auto *hint = text("录像保存为 MJPEG AVI，包含所选画面和检测标注。停止录制或结束检测后自动保存。", "tiny");
    hint->setWordWrap(true); hint->setToolTip(recordingsDirectory); layout->addWidget(hint);
}

void RecordingsPage::setRecords(const QVector<vision::app::RecordingEntry> &entries, const QString &selectedPath)
{
    const QSignalBlocker blocker(controls_.table);
    controls_.table->setRowCount(entries.size());
    int selection = entries.isEmpty() ? -1 : 0;
    for (int row = 0; row < entries.size(); ++row) {
        const auto &entry = entries[row];
        const QStringList values{entry.caption, entry.durationText, entry.resolutionText, entry.sizeText};
        for (int column = 0; column < values.size(); ++column) {
            auto *item = new QTableWidgetItem(values[column]); item->setData(Qt::UserRole, entry.path); item->setToolTip(entry.path);
            controls_.table->setItem(row, column, item);
        }
        if (entry.path == selectedPath) selection = row;
    }
    controls_.count->setText(QString("本地录像 · %1 个文件").arg(entries.size()));
    if (selection >= 0) controls_.table->setCurrentCell(selection, 0);
}

void RecordingsPage::setPlayback(const vision::app::RecordingPlaybackState &state)
{
    controls_.details->setText(state.details); controls_.play->setText(state.playing ? "暂停播放" : "播放录像");
    controls_.play->setEnabled(state.canPlay); controls_.exportVideo->setEnabled(state.canExport && !state.exporting);
    if (!state.canPlay) controls_.canvas->clear();
}

void RecordingsPage::showFrame(const QImage &image, const QString &path)
{
    vision::InferenceResult preview; preview.source = path; preview.image = image; controls_.canvas->setResult(preview);
}
