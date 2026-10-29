#pragma once

#include "core/visiontypes.h"
#include <QColor>

class QPainter;

namespace vision::app
{
// QImage/QPainter-only drawing shared by on-screen preview and background export.
class AnnotationRenderer final
{
  public:
    static QImage render(const InferenceResult &result, bool boxesVisible = true,
                         bool labelsVisible = true);
    static void paint(QPainter &painter, const InferenceResult &result, bool boxesVisible,
                      bool labelsVisible, int selected, qreal scale, bool exporting);
    static QColor classColor(int classId);
};
} // namespace vision::app
