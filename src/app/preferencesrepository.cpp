#include "preferencesrepository.h"

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
} // namespace vision::app
