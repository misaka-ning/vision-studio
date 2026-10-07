#include "icons.h"
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
namespace ui
{
QIcon icon(const QString &name, QColor color, int size)
{
    QPixmap p(size, size);
    p.fill(Qt::transparent);
    QPainter a(&p);
    a.setRenderHint(QPainter::Antialiasing);
    a.scale(size / 24.0, size / 24.0);
    a.setPen(QPen(color, 1.7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    a.setBrush(Qt::NoBrush);
    if (name == "work")
    {
        a.drawRoundedRect(QRectF(3, 3, 18, 18), 4, 4);
        a.drawLine(3, 9, 21, 9);
        a.drawLine(9, 9, 9, 21);
    }
    else if (name == "model")
    {
        a.drawPolygon(QPolygonF{QPointF(12, 2), {21, 7}, {12, 12}, {3, 7}});
        a.drawPolyline(QPolygonF{{3, 12}, {12, 17}, {21, 12}});
        a.drawPolyline(QPolygonF{{3, 17}, {12, 22}, {21, 17}});
    }
    else if (name == "graph")
    {
        a.drawLine(7, 7, 17, 12);
        a.drawLine(7, 17, 17, 12);
        a.setBrush(color.darker(240));
        for (const auto &point : {QPointF(5, 6), QPointF(5, 18), QPointF(19, 12)})
            a.drawRoundedRect(QRectF(point.x() - 3, point.y() - 3, 6, 6), 1, 1);
    }
    else if (name == "more")
    {
        a.setBrush(color);
        for (int x : {5, 12, 19})
            a.drawEllipse(QPointF(x, 12), 1.5, 1.5);
    }
    else if (name == "record")
    {
        a.drawEllipse(QPointF(12, 12), 9, 9);
        a.setBrush(color);
        a.drawEllipse(QPointF(12, 12), 4, 4);
    }
    else if (name == "history")
    {
        a.drawArc(QRectF(4, 4, 16, 16), -45 * 16, 290 * 16);
        a.drawPolyline(QPolygonF{{2, 5}, {3, 10}, {8, 9}});
        a.drawLine(12, 7, 12, 12);
        a.drawLine(12, 12, 16, 14);
    }
    else if (name == "help")
    {
        a.drawRoundedRect(QRectF(4, 3, 16, 18), 2, 2);
        a.drawLine(8, 7, 16, 7);
        a.drawLine(8, 11, 16, 11);
        a.drawLine(8, 15, 13, 15);
    }
    else if (name == "image")
    {
        a.drawRoundedRect(QRectF(3, 4, 18, 16), 2, 2);
        a.drawEllipse(QPointF(8, 9), 1.5, 1.5);
        a.drawPolyline(QPolygonF{{4, 18}, {10, 12}, {14, 16}, {17, 12}, {21, 17}});
    }
    else if (name == "folder")
    {
        a.drawPolyline(QPolygonF{{3, 7}, {3, 4}, {9, 4}, {12, 7}, {21, 7}, {21, 20}, {3, 20}, {3, 7}});
    }
    else if (name == "video")
    {
        a.drawRoundedRect(QRectF(3, 5, 13, 14), 2, 2);
        a.drawPolygon(QPolygonF{{16, 9}, {22, 6}, {22, 18}, {16, 15}});
    }
    else if (name == "camera")
    {
        a.drawPolyline(
            QPolygonF{{3, 7}, {7, 7}, {9, 4}, {15, 4}, {17, 7}, {21, 7}, {21, 20}, {3, 20}, {3, 7}});
        a.drawEllipse(QPointF(12, 13), 4, 4);
    }
    else if (name == "play")
    {
        a.setBrush(color);
        a.drawPolygon(QPolygonF{{7, 4}, {20, 12}, {7, 20}});
    }
    else if (name == "stop")
    {
        a.setBrush(color);
        a.drawRoundedRect(QRectF(6, 6, 12, 12), 1, 1);
    }
    else if (name == "import")
    {
        a.drawLine(12, 3, 12, 16);
        a.drawPolyline(QPolygonF{{7, 8}, {12, 3}, {17, 8}});
        a.drawPolyline(QPolygonF{{4, 15}, {4, 21}, {20, 21}, {20, 15}});
    }
    else if (name == "export")
    {
        a.drawLine(12, 2, 12, 15);
        a.drawPolyline(QPolygonF{{7, 10}, {12, 15}, {17, 10}});
        a.drawPolyline(QPolygonF{{4, 16}, {4, 21}, {20, 21}, {20, 16}});
    }
    else if (name == "fit")
    {
        a.drawPolyline(QPolygonF{{3, 8}, {3, 3}, {8, 3}});
        a.drawPolyline(QPolygonF{{16, 3}, {21, 3}, {21, 8}});
        a.drawPolyline(QPolygonF{{21, 16}, {21, 21}, {16, 21}});
        a.drawPolyline(QPolygonF{{8, 21}, {3, 21}, {3, 16}});
    }
    else if (name == "eye")
    {
        QPainterPath path;
        path.moveTo(2, 12);
        path.cubicTo(7, 3, 17, 3, 22, 12);
        path.cubicTo(17, 21, 7, 21, 2, 12);
        a.drawPath(path);
        a.drawEllipse(QPointF(12, 12), 3, 3);
    }
    else if (name == "plus")
    {
        a.drawLine(12, 5, 12, 19);
        a.drawLine(5, 12, 19, 12);
    }
    else if (name == "cross")
    {
        a.drawLine(6, 6, 18, 18);
        a.drawLine(18, 6, 6, 18);
    }
    else if (name == "check")
    {
        a.drawPolyline(QPolygonF{{4, 12}, {9, 17}, {20, 6}});
    }
    else if (name == "arrow")
    {
        a.drawLine(4, 12, 20, 12);
        a.drawPolyline(QPolygonF{{14, 6}, {20, 12}, {14, 18}});
    }
    else if (name == "cpu")
    {
        a.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2);
        a.drawRect(QRectF(9, 9, 6, 6));
        for (int v = 8; v <= 16; v += 4)
        {
            a.drawLine(v, 2, v, 6);
            a.drawLine(v, 18, v, 22);
            a.drawLine(2, v, 6, v);
            a.drawLine(18, v, 22, v);
        }
    }
    return QIcon(p);
}
} // namespace ui
