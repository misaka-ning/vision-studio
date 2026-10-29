#pragma once

#include <QObject>
#include <QSettings>
#include <QVariant>
#include <QVariantMap>

namespace vision::app
{
// Preserves the existing INI file, keys, QVariant encoding and default semantics.
// It intentionally does not interpret or duplicate domain configuration state.
class PreferencesRepository final : public QObject
{
    Q_OBJECT
  public:
    explicit PreferencesRepository(const QString &filePath, QObject *parent = nullptr);
    QVariant value(const QString &key, const QVariant &defaultValue = {}) const;
    bool contains(const QString &key) const;
    void setValue(const QString &key, const QVariant &value);
    void remove(const QString &key);
    void sync();
    QSettings::Status status() const;
    QString fileName() const;
    QVariantMap snapshot() const;
    void apply(const QVariantMap &values);

  private:
    QSettings settings_;
};
} // namespace vision::app
