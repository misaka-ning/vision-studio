// SPDX-License-Identifier: AGPL-3.0-only
#include "morepage.h"
#include "pagehelpers.h"
#include <QCheckBox>
#include <QHBoxLayout>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QVBoxLayout>

using namespace vision::ui::pages;

MorePage::MorePage(bool restoreDatasetOnStartup, bool cpuDownloadNeeded, QWidget *parent) : QWidget(parent)
{
    setObjectName("morePage");
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto *scroll = new QScrollArea;
    scroll->setObjectName("gpuMoreScroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->viewport()->setObjectName("moreViewport");
    scroll->viewport()->setAutoFillBackground(false);
    auto *content = new QWidget;
    content->setObjectName("moreContent");
    auto *layout = new QVBoxLayout(content);
    scroll->setWidget(content);
    outer->addWidget(scroll);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(16);
    auto *startup = card();
    auto *startupLayout = new QVBoxLayout(startup);
    startupLayout->setContentsMargins(24, 18, 24, 18);
    startupLayout->setSpacing(10);
    startupLayout->addWidget(text("启动与数据集", "modelName"));
    controls_.restoreDataset = new QCheckBox("启动时恢复上次数据集");
    controls_.restoreDataset->setObjectName("restoreDatasetOnStartup");
    controls_.restoreDataset->setChecked(restoreDatasetOnStartup);
    startupLayout->addWidget(controls_.restoreDataset);
    auto *startupHint = text("默认开启。下次打开 Vision Studio 时，在后台载入上次成功导入的数据集与集合，不自动开始检测或评估。"
                             "关闭后仍保留路径记录，当前已载入的数据集不受影响；重新开启在下次启动生效。", "muted");
    startupHint->setObjectName("restoreDatasetOnStartupHint");
    startupHint->setWordWrap(true);
    startupHint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    startupLayout->addWidget(startupHint);
    connect(controls_.restoreDataset, &QCheckBox::toggled, this, &MorePage::restoreDatasetOnStartupChanged);
    layout->addWidget(startup);
    auto *environment = card();
    auto *gpuLayout = new QVBoxLayout(environment);
    gpuLayout->setContentsMargins(24, 18, 24, 18);
    gpuLayout->setSpacing(10);
    gpuLayout->addWidget(text("NVIDIA GPU 推理环境", "modelName"));
    auto *gpuHint = text("CPU 可直接使用。首次准备 GPU 支持需要联网下载约 4 GB，建议至少预留 20 GB 空间；"
                         "依赖安装在个人目录，保留 CPU 环境，不修改显卡驱动。", "muted");
    gpuHint->setWordWrap(true);
    gpuHint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    gpuLayout->addWidget(gpuHint);
    controls_.gpuEnvironmentStatus = text("正在检查 GPU 环境…", "body");
    controls_.gpuEnvironmentStatus->setObjectName("gpuEnvironmentStatus");
    controls_.gpuEnvironmentStatus->setWordWrap(true);
    controls_.gpuEnvironmentStatus->setTextFormat(Qt::PlainText);
    controls_.gpuEnvironmentStatus->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    controls_.gpuEnvironmentStatus->setTextInteractionFlags(Qt::TextSelectableByMouse);
    gpuLayout->addWidget(controls_.gpuEnvironmentStatus);
    auto *gpuActions = new QHBoxLayout;
    controls_.gpuCheckButton = button("检查 GPU 环境", "history");
    controls_.gpuCheckButton->setObjectName("gpuCheckButton");
    controls_.gpuPrepareButton = button("准备 GPU 支持", "plus", "primary");
    controls_.gpuPrepareButton->setObjectName("gpuPrepareButton");
    controls_.gpuCancelButton = button("取消准备", "stop", "danger");
    controls_.gpuCancelButton->setObjectName("gpuCancelButton");
    controls_.gpuCancelButton->hide();
    gpuActions->addWidget(controls_.gpuCheckButton);
    gpuActions->addWidget(controls_.gpuPrepareButton);
    gpuActions->addWidget(controls_.gpuCancelButton);
    gpuActions->addStretch();
    gpuLayout->addLayout(gpuActions);
    controls_.gpuSetupProgress = new QProgressBar;
    controls_.gpuSetupProgress->setObjectName("gpuSetupProgress");
    controls_.gpuSetupProgress->setRange(0, 0);
    controls_.gpuSetupProgress->setFixedHeight(7);
    controls_.gpuSetupProgress->setTextVisible(false);
    controls_.gpuSetupProgress->hide();
    gpuLayout->addWidget(controls_.gpuSetupProgress);
    controls_.gpuLog = new QPlainTextEdit;
    controls_.gpuLog->setObjectName("gpuSetupLog");
    controls_.gpuLog->setReadOnly(true);
    controls_.gpuLog->setMaximumBlockCount(600);
    controls_.gpuLog->setMaximumHeight(135);
    controls_.gpuLog->setMinimumHeight(90);
    controls_.gpuLog->setStyleSheet("QPlainTextEdit { background:#202020; border:1px solid #454545; "
                                    "border-radius:4px; color:#D2D2D2; padding:8px; font-size:12px; }");
    controls_.gpuLog->hide();
    gpuLayout->addWidget(controls_.gpuLog);
    connect(controls_.gpuCheckButton, &QPushButton::clicked, this, &MorePage::gpuCheckRequested);
#ifdef Q_OS_WIN
    controls_.cpuPrepareButton = button("准备 CPU 运行环境", "plus");
    controls_.cpuPrepareButton->setObjectName("cpuPrepareButton");
    controls_.cpuPrepareButton->setVisible(cpuDownloadNeeded);
    gpuLayout->insertWidget(0, controls_.cpuPrepareButton);
    connect(controls_.cpuPrepareButton, &QPushButton::clicked, this, &MorePage::cpuPrepareRequested);
#else
    Q_UNUSED(cpuDownloadNeeded);
#endif
    connect(controls_.gpuPrepareButton, &QPushButton::clicked, this, &MorePage::gpuPrepareRequested);
    connect(controls_.gpuCancelButton, &QPushButton::clicked, this, &MorePage::gpuCancelRequested);
    layout->addWidget(environment);
    auto *exports = card();
    auto *exportLayout = new QVBoxLayout(exports);
    exportLayout->setContentsMargins(24, 22, 24, 22);
    exportLayout->setSpacing(12);
    exportLayout->addWidget(text("导出检测结果", "modelName"));
    auto *exportHint = text("将当前检测结果保存为标注 PNG、JSON 和 CSV。请先在工作台完成检测；也可使用 Ctrl + E。", "muted");
    exportHint->setWordWrap(true);
    exportLayout->addWidget(exportHint);
    controls_.exportButton = button("导出结果", "export");
    controls_.exportButton->setEnabled(false);
    controls_.exportButton->setObjectName("exportResultButton");
    connect(controls_.exportButton, &QPushButton::clicked, this, &MorePage::exportRequested);
    exportLayout->addWidget(controls_.exportButton, 0, Qt::AlignLeft);
    layout->addWidget(exports);
    layout->addStretch();
}

void MorePage::setRestoreDatasetOnStartup(bool enabled)
{
    const QSignalBlocker blocker(controls_.restoreDataset);
    controls_.restoreDataset->setChecked(enabled);
}
