// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "pages/workbenchpage.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QSpinBox>

void MainWindow::projectConfigurationToView()
{
    if (!configuration_ || !workbenchPage_) return;
    const auto draft = configuration_->snapshot();
    workbenchPage_->showConfiguration({draft.parameters, draft.confidence, draft.iou,
        draft.classFilterText, draft.cameraIndex, draft.stereoView, draft.automaticExport});
    workbenchPage_->setExportDirectory(draft.exportDirectory);
}

void MainWindow::connectConfiguration()
{
    // Page emits the typed edit before its operation-specific action. Patch
    // only that field, then project the authoritative draft with blocked signals.
    // A rounded or temporarily stale value in another editor is never sampled.
    const auto bind = [this](auto signal, auto command) {
        connect(workbenchPage_, signal, this, [this, command](auto... values) {
            if (!isClosing()) (configuration_->*command)(values...);
        });
    };
    using Configuration = vision::app::WorkbenchConfiguration;
    bind(&WorkbenchPage::taskEdited, &Configuration::setTask);
    bind(&WorkbenchPage::inputSizeEdited, &Configuration::setInputSize);
    bind(&WorkbenchPage::confidenceEdited, &Configuration::setConfidence);
    bind(&WorkbenchPage::iouEdited, &Configuration::setIou);
    bind(&WorkbenchPage::scaleEdited, &Configuration::setScale);
    bind(&WorkbenchPage::meanREdited, &Configuration::setMeanR);
    bind(&WorkbenchPage::meanGEdited, &Configuration::setMeanG);
    bind(&WorkbenchPage::meanBEdited, &Configuration::setMeanB);
    bind(&WorkbenchPage::inputColorEdited, &Configuration::setInputColor);
    bind(&WorkbenchPage::computeDeviceEdited, &Configuration::setComputeDevice);
    bind(&WorkbenchPage::deviceIndexEdited, &Configuration::setDeviceIndex);
    bind(&WorkbenchPage::classFilterEdited, &Configuration::setClassFilterText);
    bind(&WorkbenchPage::cameraIndexEdited, &Configuration::setCameraIndex);
    bind(&WorkbenchPage::stereoViewEdited, &Configuration::setStereoView);
    bind(&WorkbenchPage::automaticExportEdited, &Configuration::setAutomaticExport);
    connect(configuration_, &Configuration::changed, this, &MainWindow::projectConfigurationToView);
    projectConfigurationToView();
}

vision::ModelConfig MainWindow::currentConfig() const
{
    const auto model = modelState();
    return configuration_->compose(model, !convertedModelProfile(model.activeModel).isEmpty());
}

vision::app::ClassFilterStatus MainWindow::classFilterStatus() const
{
    return configuration_->classFilterStatus(modelState(), modelFileIdentity());
}
