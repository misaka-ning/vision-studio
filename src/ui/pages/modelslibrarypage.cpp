// SPDX-License-Identifier: AGPL-3.0-only
#include "modelslibrarypage.h"
#include "pagehelpers.h"
#include "app/modelsessioncontroller.h"
#include "app/operationcoordinator.h"
#include <QAction>
#include <QDesktopServices>
#include <QFileInfo>
#include <QInputDialog>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QMenu>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <QUrl>
#include <functional>

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
    connect(list, &QListWidget::customContextMenuRequested, this, [this](const QPoint &position) {
        emit contextMenuRequested(position);
        showContextMenu(position);
    });
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

void ModelsLibraryPage::setApplicationContext(const vision::app::ModelSessionController *models,
    const vision::app::OperationCoordinator *operations)
{
    models_ = models; operations_ = operations;
}

bool ModelsLibraryPage::canMutate(const QString &path) const
{
    return models_ && operations_ && !busy_ && models_->snapshot().models.contains(path) &&
        vision::app::OperationCoordinator::availability(operations_->activity()).importModels;
}

void ModelsLibraryPage::showContextMenu(const QPoint &position)
{
    auto *list = controls_.list;
    auto *item = list->itemAt(position);
    if (!item && position.x() < 0 && position.y() < 0) item = list->currentItem();
    if (!item || !models_) return;
    const QString path = item->data(Qt::UserRole).toString();
    auto *menu = new QMenu(list);
    menu->setObjectName("modelContextMenu");
    menu->setAttribute(Qt::WA_DeleteOnClose);
    auto addAction = [this, menu, path](const QString &title, const QString &icon,
        const char *name, std::function<void()> handler, bool mutation) {
        auto *action = menu->addAction(::ui::icon(icon, QColor("#D2D2D2")), title);
        action->setObjectName(name); action->setData(path);
        action->setEnabled(!mutation || canMutate(path));
        connect(action, &QAction::triggered, this, std::move(handler));
    };
    addAction("打开模型位置", "folder", "modelOpenLocationAction", [this, path] { openModelLocation(path); }, false);
    addAction(models_->snapshot().notes.value(path).toString().isEmpty() ? "添加备注" : "修改备注",
        "note", "modelNoteAction", [this, path] { editModelNote(path); }, true);
    addAction("重命名", "edit", "modelRenameAction", [this, path] { renameModel(path); }, true);
    addAction("从列表移除", "cross", "modelRemoveAction", [this, path] {
        if (canMutate(path)) emit removeRequested(path);
    }, true);
    const QPoint anchor = position.x() < 0 && position.y() < 0 ? list->visualItemRect(item).center() : position;
    menu->popup(list->viewport()->mapToGlobal(anchor));
}

void ModelsLibraryPage::openModelLocation(const QString &path)
{
    const QString directory = QFileInfo(path).absolutePath();
    if (!QFileInfo(directory).isDir()) {
        emit noticeRequested("模型所在文件夹不存在：" + directory, true); return;
    }
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(directory)))
        emit noticeRequested("无法打开模型文件夹：" + directory, true);
}

void ModelsLibraryPage::editModelNote(const QString &path)
{
    if (!canMutate(path)) return;
    QInputDialog dialog(this);
    dialog.setObjectName("modelNoteDialog"); dialog.setWindowTitle("模型备注");
    dialog.setInputMode(QInputDialog::TextInput);
    dialog.setOption(QInputDialog::UsePlainTextEditForTextInput);
    dialog.setLabelText(models_->displayName(path) + "\n添加实验说明、数据集或用途；留空可清除备注。");
    for (auto *label : dialog.findChildren<QLabel *>()) label->setTextFormat(Qt::PlainText);
    dialog.setTextValue(models_->snapshot().notes.value(path).toString());
    dialog.setOkButtonText("保存"); dialog.setCancelButtonText("取消"); dialog.resize(520, 260);
    if (dialog.exec() != QDialog::Accepted || !canMutate(path)) return;
    emit noteRequested(path, dialog.textValue().trimmed());
}

void ModelsLibraryPage::renameModel(const QString &path)
{
    if (!canMutate(path)) return;
    QInputDialog dialog(this);
    dialog.setObjectName("modelRenameDialog"); dialog.setWindowTitle("重命名模型");
    dialog.setInputMode(QInputDialog::TextInput);
    dialog.setLabelText("模型库中的显示名称（原始文件名和路径不变）\n留空可恢复原始文件名：" + QFileInfo(path).fileName());
    for (auto *label : dialog.findChildren<QLabel *>()) label->setTextFormat(Qt::PlainText);
    dialog.setTextValue(models_->snapshot().displayNames.value(path).toString());
    dialog.setOkButtonText("保存"); dialog.setCancelButtonText("取消"); dialog.resize(520, 180);
    if (dialog.exec() != QDialog::Accepted || !canMutate(path)) return;
    emit renameRequested(path, dialog.textValue().simplified());
}
