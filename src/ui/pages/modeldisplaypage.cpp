// SPDX-License-Identifier: AGPL-3.0-only
#include "modeldisplaypage.h"
#include "pagehelpers.h"
#include "ui/modelviewer.h"
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>
using namespace vision::ui::pages;

ModelDisplayPage::ModelDisplayPage(QWidget *parent) : QWidget(parent)
{
    setObjectName("modelDisplayPage");
    auto *layout = new QVBoxLayout(this); layout->setContentsMargins(0, 0, 0, 0); layout->setSpacing(14);
    auto *cardWidget = card(); auto *body = new QVBoxLayout(cardWidget);
    body->setContentsMargins(18, 16, 18, 16); body->setSpacing(10);
    auto *heading = new QHBoxLayout;
    controls_.modelName = text("在模型库中选择模型", "structureModelName");
    controls_.modelName->setStyleSheet("font-size:16px; font-weight:600; color:#F5F5F5;");
    heading->addWidget(controls_.modelName, 1);
    controls_.status = text("等待模型", "chip"); controls_.status->setObjectName("structureLoadStatus");
    heading->addWidget(controls_.status); heading->addWidget(text("Netron 9.3.1 · 本地查看", "chip")); body->addLayout(heading);
    controls_.metadata = text("支持 ONNX、PyTorch、TorchScript 等模型格式。", "structureModelMeta");
    controls_.metadata->setStyleSheet("font-size:12px; color:#B8B8B8;"); body->addWidget(controls_.metadata);
    controls_.hint = text("想看完整结构，建使用导出的 ONNX。", "structureOnnxHint");
    controls_.hint->setStyleSheet("font-size:11px; color:#A5A5A5;"); controls_.hint->hide(); body->addWidget(controls_.hint);
    layout->addWidget(cardWidget);
    controls_.viewer = new ModelViewer; controls_.viewer->setObjectName("modelStructureViewer");
    layout->addWidget(controls_.viewer, 1);
    connect(controls_.viewer, &ModelViewer::stateChanged, this, [this](ModelViewer::State state) {
        controls_.status->setText(state == ModelViewer::State::Loading ? "后台加载中…" : state == ModelViewer::State::Ready
            ? "可用视图已缓存" : state == ModelViewer::State::Error ? "无法显示" : "等待模型");
    });
    connect(controls_.viewer, &ModelViewer::modelLoaded, this, &ModelDisplayPage::modelLoaded);
    connect(controls_.viewer, &ModelViewer::loadFailed, this, &ModelDisplayPage::loadFailed);
    auto *note = text("使用模型库中全局选中的模型，各视图共用后台解析缓存。结构图可缩放与平移；层级树和参数"
                      "表支持搜索与详情查看。包含受支持架构信息的 PT 模型还可查看 YOLO YAML。", "tiny");
    note->setWordWrap(true); layout->addWidget(note);
}

void ModelDisplayPage::setSelectedModelHeader(const QString &path, const QString &displayName, bool needsRuntime)
{
    controls_.modelName->setText(path.isEmpty() ? "在模型库中选择模型" : displayName);
    controls_.modelName->setToolTip(path); controls_.metadata->setToolTip(path); controls_.metadata->setWordWrap(true);
    controls_.metadata->setText(path.isEmpty() ? "检测工作台与模型显示使用同一个全局模型。" : needsRuntime
        ? "首次查看模型结构需要准备运行环境，可打开模型显示页或在更多页点击“准备 CPU 运行环境”。"
        : QFileInfo(path).suffix().toUpper() + " · " + path);
    controls_.hint->setVisible(!path.isEmpty() && QFileInfo(path).suffix().compare("onnx", Qt::CaseInsensitive) != 0);
}
