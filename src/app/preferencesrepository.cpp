#include "preferencesrepository.h"
#include "workbenchconfiguration.h"
#include "core/gpuruntime.h"

namespace vision::app
{
PreferencesRepository::PreferencesRepository(const QString &filePath, QObject *parent)
    : QObject(parent), settings_(filePath, QSettings::IniFormat)
{
    settings_.setAtomicSyncRequired(true);
}

QVariant PreferencesRepository::value(const QString &key, const QVariant &defaultValue) const
{
    return settings_.value(key, defaultValue);
}

bool PreferencesRepository::contains(const QString &key) const { return settings_.contains(key); }
void PreferencesRepository::setValue(const QString &key, const QVariant &value) { settings_.setValue(key, value); }
void PreferencesRepository::remove(const QString &key) { settings_.remove(key); }
void PreferencesRepository::sync() { settings_.sync(); }
QSettings::Status PreferencesRepository::status() const { return settings_.status(); }
QString PreferencesRepository::fileName() const { return settings_.fileName(); }

QVariantMap PreferencesRepository::snapshot() const
{
    QVariantMap values;
    for (const QString &key : settings_.allKeys()) values.insert(key, settings_.value(key));
    return values;
}

void PreferencesRepository::apply(const QVariantMap &values)
{
    for (auto it = values.cbegin(); it != values.cend(); ++it) settings_.setValue(it.key(), it.value());
}

void PreferencesRepository::saveWorkspace(const ModelSessionSnapshot &model, const WorkbenchDraft &draft,
                                         bool includeParameters)
{
    setValue("models", model.models);
    setValue("modelDisplayNames", model.displayNames);
    setValue("modelNotes", model.notes);
    setValue("convertedModelProfiles", model.convertedProfiles);
    setValue("evaluationProfiles", model.evaluationProfiles);
    setValue("activeModel", model.activeModel);
    setValue("exportDirectory", draft.exportDirectory);
    const auto &parameters = draft.parameters;
    setValue("labels", parameters.labels);
    setValue("labelsPath", draft.labelsPath);
    if (includeParameters) {
        setValue("task", int(parameters.task));
        setValue("inputSize", parameters.inputSize);
        setValue("confidence", draft.confidence);
        setValue("iou", draft.iou);
        setValue("scale", parameters.scale);
        setValue("meanR", parameters.meanR);
        setValue("meanG", parameters.meanG);
        setValue("meanB", parameters.meanB);
        setValue("colorMode", parameters.colorMode == InputColorMode::Grayscale ? "grayscale" : parameters.swapRB ? "rgb" : "bgr");
        setValue("stereoView", draft.stereoView == StereoView::Left ? "left" : draft.stereoView == StereoView::Right ? "right" : "full");
        setValue("cameraIndex", draft.cameraIndex);
        setValue("computeDevice", computeDeviceKey(parameters.device));
        setValue("deviceIndex", parameters.deviceIndex);
        setValue("swapRB", parameters.swapRB);
        setValue("autoExport", draft.automaticExport);
        setValue("classFilter", draft.classFilterText);
    }
    sync();
}
} // namespace vision::app
