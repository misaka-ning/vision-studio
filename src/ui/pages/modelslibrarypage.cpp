// SPDX-License-Identifier: AGPL-3.0-only
#include "modelslibrarypage.h"
#include "pagehelpers.h"
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>

using namespace vision::ui::pages;
namespace
{
class LibraryList final : public QListWidget
{
  public:
    bool busy = false;
  protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (busy || event->button() == Qt::RightButton) event->accept();
        else QListWidget::mousePressEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (busy || event->button() == Qt::RightButton) event->accept();
        else QListWidget::mouseReleaseEvent(event);
    }
    void mouseDoubleClickEvent(QMouseEvent *event) override
    {
        if (busy || event->button() == Qt::RightButton) event->accept();
        else QListWidget::mouseDoubleClickEvent(event);
    }
    void keyPressEvent(QKeyEvent *event) override
    {
        if (busy && event->key() != Qt::Key_Menu &&
            !(event->key() == Qt::Key_F10 && event->modifiers() == Qt::ShiftModifier)) event->accept();
        else QListWidget::keyPressEvent(event);
    }
};
}

ModelsLibraryPage::ModelsLibraryPage(QWidget *parent) : QWidget(parent)
{
    setObjectName("modelLibraryPage");
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0); layout->setSpacing(16);
    auto *row = new QHBoxLayout;
    controls_.count = text("本地模型库", "sectionTitle"); row->addWidget(controls_.count, 1);
    controls_.import = button("导入 ONNX / PT 模型", "plus", "primary"); row->addWidget(controls_.import);
    connect(controls_.import, &QPushButton::clicked, this, &ModelsLibraryPage::importRequested);
    layout->addLayout(row);
    auto *frame = card(); auto *body = new QVBoxLayout(frame);
    body->setContentsMargins(20, 18, 20, 18); body->setSpacing(12);
    body->addWidget(text("集中管理你的模型", "modelName"));
    body->addWidget(text("点击模型即全局选用：检测工作台与模型显示同步，模型结构会在后台预加载。", "muted"));
    controls_.dropHint = text("将 ONNX / PT / TorchScript 文件拖到此处，或点击导入；支持多选，原文件保留。", "muted");
    controls_.dropHint->setObjectName("modelLibraryDropHint"); controls_.dropHint->setWordWrap(true);
    body->addWidget(controls_.dropHint);
    auto *list = new LibraryList; controls_.list = list;
    list->setObjectName("globalModelList"); list->setIconSize(QSize(34, 34));
    list->setDragEnabled(true); list->setAcceptDrops(true); list->setDropIndicatorShown(true);
    list->setDragDropMode(QAbstractItemView::InternalMove); list->setDefaultDropAction(Qt::MoveAction);
    list->setDragDropOverwriteMode(false); list->setSelectionMode(QAbstractItemView::SingleSelection);
    list->setEditTriggers(QAbstractItemView::NoEditTriggers);
    list->setToolTip("拖入本地模型文件以添加；库内拖动调整顺序；点击全局选用，右键管理模型。");
    list->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(list, &QListWidget::customContextMenuRequested, this, &ModelsLibraryPage::contextMenuRequested);
    connect(list->model(), &QAbstractItemModel::rowsAboutToBeMoved, this, [this] { reordering_ = true; });
    connect(list->model(), &QAbstractItemModel::rowsMoved, this, [this] {
        QStringList paths;
        for (int row = 0; row < controls_.list->count(); ++row)
            paths.append(controls_.list->item(row)->data(Qt::UserRole).toString());
        reordering_ = false;
        emit orderChanged(paths);
    });
    connect(list, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *item) {
        if (item && !busy_ && !reordering_) emit selectionRequested(item->data(Qt::UserRole).toString());
    });
    connect(list, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *) { emit detectRequested(); });
    body->addWidget(list, 1);
    auto *actions = new QHBoxLayout;
    controls_.activate = button("前往检测工作台", "arrow", "primary");
    controls_.structure = button("查看模型结构", "graph"); controls_.structure->setObjectName("showModelStructureButton");
    controls_.remove = button("从列表移除", "cross");
    controls_.evaluationReport = button("查看评估 / 检测配置", "history");
    controls_.evaluationReport->setObjectName("modelEvaluationReportButton");
    for (auto *action : {controls_.activate, controls_.structure, controls_.remove, controls_.evaluationReport}) actions->addWidget(action);
    actions->addStretch(); body->addLayout(actions);
    connect(controls_.activate, &QPushButton::clicked, this, &ModelsLibraryPage::detectRequested);
    connect(controls_.structure, &QPushButton::clicked, this, &ModelsLibraryPage::structureRequested);
    connect(controls_.evaluationReport, &QPushButton::clicked, this, &ModelsLibraryPage::evaluationRequested);
    connect(controls_.remove, &QPushButton::clicked, this, [this] {
        if (const auto *item = controls_.list->currentItem()) emit removeRequested(item->data(Qt::UserRole).toString());
    });
    layout->addWidget(frame, 1);
    auto *notes = card(); auto *noteLayout = new QVBoxLayout(notes); noteLayout->setContentsMargins(18, 14, 18, 14);
    noteLayout->addWidget(text("推荐模型导出配置", "sectionTitle"));
    auto *info = text("PT · 本机 PyTorch 直接推理，自动读取模型内置类别。\nONNX · batch=1、固定正方形输入、FP32、不包含 "
                      "NMS。YOLOv5 使用 [1, N, 5+C]，YOLOv8 / 11 使用 [1, 4+C, N]。", "muted");
    info->setWordWrap(true); noteLayout->addWidget(info); layout->addWidget(notes);
    setAcceptDrops(true);
    for (auto *child : findChildren<QWidget *>()) child->setAcceptDrops(true);
}

void ModelsLibraryPage::setBusy(bool busy)
{
    busy_ = busy;
    static_cast<LibraryList *>(controls_.list)->busy = busy;
    controls_.list->setDragEnabled(!busy); controls_.list->setAcceptDrops(!busy);
    for (auto *action : {controls_.import, controls_.activate, controls_.remove}) action->setEnabled(!busy);
}

void ModelsLibraryPage::setRows(const QVector<ModelLibraryRow> &rows, const QString &selectedPath)
{
    const QSignalBlocker blocker(controls_.list);
    controls_.list->clear();
    for (const auto &row : rows) {
        QString caption = row.displayName + "\n" + row.detail + "\n" + row.path;
        if (!row.note.isEmpty()) {
            const auto compact = row.note.simplified();
            caption += "\n备注 · " + compact.left(140) + (compact.size() > 140 ? QStringLiteral("…") : QString());
        }
        if (!row.evaluationSummary.isEmpty()) caption += "\n" + row.evaluationSummary;
        auto *item = new QListWidgetItem(::ui::icon("model", QColor("#60CDFF"), 34), caption);
        item->setData(Qt::UserRole, row.path); item->setData(Qt::UserRole + 1, row.displayName);
        item->setData(Qt::UserRole + 2, row.note); item->setToolTip(row.tooltip);
        item->setSizeHint(QSize(100, row.height)); controls_.list->addItem(item);
        if (row.path == selectedPath) controls_.list->setCurrentItem(item);
    }
    controls_.count->setText(QString("本地模型库 · %1 个模型").arg(rows.size()));
}

void ModelsLibraryPage::setDropHint(const QString &message, bool error)
{
    controls_.dropHint->setText(message);
    controls_.dropHint->setStyleSheet(error ? "color:#FFB4BB;" : "color:#B8B8B8;");
}
