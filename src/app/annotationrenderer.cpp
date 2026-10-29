#include "annotationrenderer.h"

#include <QFont>
#include <QPainter>
#include <algorithm>
#include <limits>

namespace vision::app
{
namespace
{
QFont canvasFont(qreal size, QFont::Weight weight = QFont::Normal)
{
    QFont font(QStringLiteral("Noto Sans CJK SC"));
    font.setPointSizeF(size);
    font.setWeight(weight);
    return font;
}

} // namespace

QColor AnnotationRenderer::classColor(int classId)
{
    static const QColor colors[] = {QColor("#2dd4bf"), QColor("#60a5fa"), QColor("#fbbf24"),
                                    QColor("#c084fc"), QColor("#fb7185"), QColor("#a3e635"),
                                    QColor("#38bdf8"), QColor("#f97316")};
    return colors[static_cast<unsigned int>(std::max(classId, 0)) % 8];
}

void AnnotationRenderer::paint(QPainter &painter, const vision::InferenceResult &result,
                                          bool boxesVisible, bool labelsVisible, int selectedIndex,
                                          qreal scale, bool exporting)
{
    if (!boxesVisible && !labelsVisible)
        return;
    const qreal unit = exporting ? std::max(1.0, result.image.width() / 1000.0) : 1.0 / scale;
    const QRectF bounds(QPointF(0, 0), result.image.size());
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing);
    if (result.task == vision::ModelTask::Classification)
    {
        if (labelsVisible && !result.predictions.isEmpty())
        {
            const auto &prediction = result.predictions.first();
            QFont font = canvasFont(10, QFont::DemiBold);
            font.setPixelSize(qRound(13 * unit));
            painter.setFont(font);
            const QString label = QStringLiteral("分类  %1  ·  %2%")
                                      .arg(prediction.label)
                                      .arg(prediction.confidence * 100, 0, 'f', 1);
            const qreal width = std::min(qreal(painter.fontMetrics().horizontalAdvance(label)) + 28 * unit,
                                         std::max(0.0, bounds.width() - 24 * unit));
            const QRectF badge(12 * unit, 12 * unit, width, 34 * unit);
            painter.setPen(QPen(classColor(prediction.classId), 1.5 * unit));
            painter.setBrush(QColor(10, 27, 40, 235));
            painter.drawRoundedRect(badge, 5 * unit, 5 * unit);
            painter.setPen(QColor("#e0f4ed"));
            const QString fitted =
                painter.fontMetrics().elidedText(label, Qt::ElideRight, qRound(width - 18 * unit));
            painter.drawText(badge.adjusted(12 * unit, 0, -6 * unit, 0), Qt::AlignVCenter, fitted);
        }
        painter.restore();
        return;
    }
    struct AnnotationLabel
    {
        QRectF rectangle;
        QRectF box;
        QPointF preferred;
        QColor color;
        QString text;
    };
    QVector<AnnotationLabel> labels;
    const qreal labelHeight = 26 * unit;
    const qreal labelGap = 3 * unit;
    QFont labelFont = canvasFont(9, QFont::DemiBold);
    labelFont.setPixelSize(qRound(12 * unit));
    painter.setFont(labelFont);

    // Lay labels out in image coordinates, reserving their full screen-size rectangles.
    // Painting them after the boxes keeps later box outlines from crossing readable text.
    auto placeLabel = [&](const QRectF &box, qreal labelWidth)
    {
        const qreal maxX = std::max(bounds.left(), bounds.right() - labelWidth);
        const qreal maxY = std::max(bounds.top(), bounds.bottom() - labelHeight);
        const QPointF preferred(std::clamp(box.left(), bounds.left(), maxX),
                                std::clamp(box.top() - labelHeight - labelGap, bounds.top(), maxY));
        QVector<QPointF> candidates;
        const qreal columns[] = {preferred.x(), box.right() - labelWidth, box.center().x() - labelWidth / 2};
        const qreal rows[] = {preferred.y(), box.top() + labelGap, box.bottom() - labelHeight - labelGap,
                              box.bottom() + labelGap};
        auto append = [&](qreal x, qreal y)
        {
            candidates.append(QPointF(std::clamp(x, bounds.left(), maxX), std::clamp(y, bounds.top(), maxY)));
        };
        for (qreal y : rows)
            for (qreal x : columns)
                append(x, y);
        for (int step = 1; step <= 5; ++step)
        {
            for (qreal x : columns)
            {
                append(x, preferred.y() - step * (labelHeight + labelGap));
                append(x, preferred.y() + step * (labelHeight + labelGap));
            }
        }

        QRectF best(preferred, QSizeF(labelWidth, labelHeight));
        qreal bestScore = std::numeric_limits<qreal>::max();
        auto consider = [&](const QPointF &position)
        {
            const QRectF rectangle(position, QSizeF(labelWidth, labelHeight));
            qreal overlap = 0;
            for (const auto &label : labels)
            {
                const QRectF intersection =
                    rectangle.intersected(label.rectangle.adjusted(-labelGap, -labelGap, labelGap, labelGap));
                overlap += intersection.width() * intersection.height();
            }
            const QPointF offset = position - preferred;
            const qreal score = overlap * 1000000 + offset.x() * offset.x() + offset.y() * offset.y();
            if (score < bestScore)
            {
                best = rectangle;
                bestScore = score;
            }
            return overlap <= 0;
        };
        for (const auto &position : candidates)
            if (consider(position))
                return qMakePair(best, preferred);

        // If a dense cluster fills all nearby rows, search the remaining image area.
        for (qreal y = bounds.top(); y <= maxY; y += labelHeight + labelGap)
        {
            for (qreal x = bounds.left(); x <= maxX; x += labelWidth + labelGap)
            {
                consider(QPointF(x, y));
            }
            consider(QPointF(maxX, y));
        }
        return qMakePair(best, preferred);
    };

    for (int i = 0; i < result.predictions.size(); ++i)
    {
        const auto &prediction = result.predictions[i];
        const QRectF box = prediction.box.intersected(bounds);
        if (box.isEmpty())
            continue;
        const bool selected = i == selectedIndex && !exporting;
        const QColor color = result.hasEvaluationMatches && !result.ignoredPredictions.contains(i)
                                 ? QColor(result.matchedPredictions.contains(i) ? "#6FE7B6" : "#FFB08F")
                                 : classColor(prediction.classId);
        QColor fill = color;
        fill.setAlpha(selected ? 43 : 14);
        if (boxesVisible)
        {
            painter.setPen(QPen(color, (selected ? 2.5 : 1.7) * unit));
            painter.setBrush(fill);
            painter.drawRect(box);
            painter.setPen(QPen(selected ? QColor("#ffffff") : color, 3 * unit));
            const qreal length = std::min({14 * unit, box.width() / 4, box.height() / 4});
            for (const auto &corner : {box.topLeft(), box.topRight(), box.bottomLeft(), box.bottomRight()})
            {
                const qreal dx = corner.x() == box.left() ? length : -length;
                const qreal dy = corner.y() == box.top() ? length : -length;
                painter.drawLine(corner, corner + QPointF(dx, 0));
                painter.drawLine(corner, corner + QPointF(0, dy));
            }
        }
        if (labelsVisible)
        {
            const QString label =
                QStringLiteral("%1  %2%")
                    .arg((result.hasEvaluationMatches && !result.ignoredPredictions.contains(i)
                              ? (result.matchedPredictions.contains(i) ? QStringLiteral("TP · ") : QStringLiteral("FP · "))
                              : QString()) +
                         (prediction.label.isEmpty() ? QStringLiteral("class %1").arg(prediction.classId)
                                                     : prediction.label))
                    .arg(qRound(prediction.confidence * 100));
            const qreal labelWidth =
                std::min(painter.fontMetrics().horizontalAdvance(label) + 16 * unit, bounds.width());
            const auto placement = placeLabel(box, labelWidth);
            const QString fitted =
                painter.fontMetrics().elidedText(label, Qt::ElideRight, qRound(labelWidth - 13 * unit));
            labels.append({placement.first, box, placement.second, color, fitted});
        }
    }

    if (result.hasGroundTruth)
        for (int i = 0; i < result.groundTruth.size(); ++i)
        {
            const auto &gt = result.groundTruth[i];
            const QRectF box = gt.box.intersected(bounds);
            const bool matched = !result.hasEvaluationMatches || result.matchedGroundTruth.contains(i);
            const QColor color(matched ? "#87C8FF" : "#E6AAFF");
            if (boxesVisible)
            {
                painter.setBrush(Qt::NoBrush);
                painter.setPen(QPen(color, 2 * unit, Qt::DashLine));
                painter.drawRect(box);
            }
            if (labelsVisible && !box.isEmpty())
            {
                const QString label = (matched ? QStringLiteral("GT · ") : QStringLiteral("FN · ")) + gt.label;
                const qreal width = std::min(painter.fontMetrics().horizontalAdvance(label) + 16 * unit, bounds.width());
                const auto position = placeLabel(box, width);
                labels.append({position.first, box, position.second, color, label});
            }
        }

    for (const auto &label : labels)
    {
        if ((label.rectangle.topLeft() - label.preferred).manhattanLength() < unit)
            continue;
        const QPointF center = label.rectangle.center();
        QPointF anchor(std::clamp(center.x(), label.box.left(), label.box.right()),
                       std::clamp(center.y(), label.box.top(), label.box.bottom()));
        if (label.box.contains(center))
        {
            const qreal distances[] = {center.y() - label.box.top(), label.box.bottom() - center.y(),
                                       center.x() - label.box.left(), label.box.right() - center.x()};
            const int edge =
                std::min_element(std::begin(distances), std::end(distances)) - std::begin(distances);
            if (edge == 0)
                anchor.setY(label.box.top());
            if (edge == 1)
                anchor.setY(label.box.bottom());
            if (edge == 2)
                anchor.setX(label.box.left());
            if (edge == 3)
                anchor.setX(label.box.right());
        }
        const QPointF endpoint(std::clamp(anchor.x(), label.rectangle.left(), label.rectangle.right()),
                               std::clamp(anchor.y(), label.rectangle.top(), label.rectangle.bottom()));
        painter.setPen(QPen(label.color, 1.2 * unit));
        painter.drawLine(anchor, endpoint);
        painter.setPen(Qt::NoPen);
        painter.setBrush(label.color);
        painter.drawEllipse(anchor, 2.2 * unit, 2.2 * unit);
    }
    painter.setFont(labelFont);
    for (const auto &label : labels)
    {
        painter.setPen(Qt::NoPen);
        painter.setBrush(label.color);
        painter.drawRoundedRect(label.rectangle, 3 * unit, 3 * unit);
        painter.setPen(QColor("#081422"));
        painter.drawText(label.rectangle.adjusted(8 * unit, 0, -5 * unit, 0),
                         Qt::AlignVCenter | Qt::AlignLeft, label.text);
    }
    painter.restore();
}

QImage AnnotationRenderer::render(const vision::InferenceResult &result,
                                         bool boxesVisible, bool labelsVisible)
{
    if (result.image.isNull())
        return {};
    QImage image = result.image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&image);
    paint(painter, result, boxesVisible, labelsVisible, -1, 1, true);
    if (result.demonstration)
    {
        const qreal scale = std::max(1.0, image.width() / 1000.0);
        QFont font = canvasFont(10, QFont::DemiBold);
        font.setPixelSize(qRound(14 * scale));
        painter.setFont(font);
        const QString text = QStringLiteral("示范场景 · 非模型推理结果");
        QRectF badge(16 * scale, 16 * scale, painter.fontMetrics().horizontalAdvance(text) + 24 * scale,
                     32 * scale);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(12, 24, 37, 225));
        painter.drawRoundedRect(badge, 5 * scale, 5 * scale);
        painter.setPen(QColor("#d5e3ee"));
        painter.drawText(badge, Qt::AlignCenter, text);
    }
    return image;
}

} // namespace vision::app
