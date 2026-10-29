#include "imagecanvas.h"
#include "app/annotationrenderer.h"

#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QImageReader>
#include <QKeyEvent>
#include <QLinearGradient>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QUrl>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr qreal kMinimumScale = 0.025;
constexpr qreal kMaximumScale = 12.0;

QFont canvasFont(qreal size, QFont::Weight weight = QFont::Normal)
{
    QFont font(QStringLiteral("Noto Sans CJK SC"));
    font.setPointSizeF(size);
    font.setWeight(weight);
    return font;
}

void drawCorner(QPainter &painter, const QPointF &position, qreal x, qreal y)
{
    painter.drawLine(position, position + QPointF(x * 13, 0));
    painter.drawLine(position, position + QPointF(0, y * 13));
}
} // namespace

ImageCanvas::ImageCanvas(QWidget *parent) : QWidget(parent)
{
    setAcceptDrops(true);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumSize(300, 260);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setAccessibleName(QStringLiteral("视觉图像画布"));
    setAccessibleDescription(
        QStringLiteral("可拖入图片、视频或 YOLO 数据集目录 / YAML。临时图片仅参与检测。滚轮缩放，拖动平移，双击适应视图，F 适应视图，1 原始尺寸。"));
}

void ImageCanvas::setResult(const vision::InferenceResult &result)
{
    const bool resetView = m_result.image.isNull() || m_result.source != result.source ||
                           m_result.image.size() != result.image.size();
    m_result = result;
    if (resetView || m_selected >= m_result.predictions.size())
        m_selected = -1;
    if (resetView || m_fit)
        fitToView();
    else
        update();
}

void ImageCanvas::clear()
{
    m_result = {};
    m_selected = -1;
    m_pan = {};
    m_fit = true;
    m_scale = 1;
    emit zoomChanged(100);
    update();
}

void ImageCanvas::setBoxesVisible(bool visible)
{
    m_boxesVisible = visible;
    update();
}
void ImageCanvas::setLabelsVisible(bool visible)
{
    m_labelsVisible = visible;
    update();
}

void ImageCanvas::setSelectedPrediction(int index)
{
    m_selected = index >= 0 && index < m_result.predictions.size() ? index : -1;
    update();
}

void ImageCanvas::fitToView()
{
    m_fit = true;
    m_pan = {};
    if (!m_result.image.isNull())
    {
        const qreal w = std::max(1, width() - 64);
        const qreal h = std::max(1, height() - 64);
        m_scale = std::clamp(std::min(w / m_result.image.width(), h / m_result.image.height()), kMinimumScale,
                             kMaximumScale);
    }
    emit zoomChanged(qRound(m_scale * 100));
    update();
}

void ImageCanvas::actualSize()
{
    m_fit = false;
    m_scale = 1;
    m_pan = {};
    emit zoomChanged(100);
    update();
}

QRectF ImageCanvas::imageRect() const
{
    const QSizeF size = QSizeF(m_result.image.size()) * m_scale;
    return QRectF(QPointF((width() - size.width()) / 2, (height() - size.height()) / 2) + m_pan, size);
}

QPointF ImageCanvas::toImage(const QPointF &point) const
{
    return (point - imageRect().topLeft()) / m_scale;
}

void ImageCanvas::paintAnnotations(QPainter &painter, qreal scale, bool exporting) const
{
    if (m_groundTruthVisible)
        vision::app::AnnotationRenderer::paint(painter, m_result, m_boxesVisible, m_labelsVisible, m_selected, scale, exporting);
    else
    {
        auto visible = m_result;
        visible.groundTruth.clear();
        vision::app::AnnotationRenderer::paint(painter, visible, m_boxesVisible, m_labelsVisible, m_selected, scale, exporting);
    }
}

void ImageCanvas::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor("#202020"));
    painter.setPen(QPen(QColor("#2b2b2b"), 1));
    for (int x = 20; x < width(); x += 24)
        for (int y = 20; y < height(); y += 24)
            painter.drawPoint(x, y);

    if (m_result.image.isNull())
    {
        const QPointF center(width() / 2.0, height() / 2.0 - 26);
        const QRectF icon(center.x() - 34, center.y() - 48, 68, 68);
        painter.setPen(QPen(QColor("#454545"), 1.5));
        painter.setBrush(QColor("#303030"));
        painter.drawRoundedRect(icon, 8, 8);
        const QRectF frame = icon.adjusted(17, 19, -17, -19);
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(QColor("#60CDFF"), 1.8));
        painter.drawRoundedRect(frame, 3, 3);
        QPainterPath mountains;
        mountains.moveTo(frame.left() + 3, frame.bottom() - 4);
        mountains.lineTo(frame.left() + 11, frame.top() + 12);
        mountains.lineTo(frame.left() + 17, frame.top() + 18);
        mountains.lineTo(frame.left() + 24, frame.top() + 10);
        mountains.lineTo(frame.right() - 3, frame.bottom() - 4);
        painter.drawPath(mountains);
        painter.drawEllipse(frame.topLeft() + QPointF(8, 7), 2, 2);
        painter.setFont(canvasFont(13, QFont::DemiBold));
        painter.setPen(QColor("#F5F5F5"));
        painter.drawText(QRectF(0, center.y() + 38, width(), 30), Qt::AlignCenter,
                         QStringLiteral("拖入图片或数据集，开始视觉检测"));
        painter.setFont(canvasFont(9));
        painter.setPen(QColor("#B8B8B8"));
        painter.drawText(QRectF(0, center.y() + 77, width(), 22), Qt::AlignCenter,
                         QStringLiteral("图片 / YOLO 数据集 / 视频 · 也可连接摄像头"));
    }
    else
    {
        const QRectF target = imageRect();
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 75));
        painter.drawRoundedRect(target.adjusted(-5, -5, 5, 5).translated(0, 4), 3, 3);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, m_scale < 2);
        painter.drawImage(target, m_result.image);
        painter.save();
        painter.translate(target.topLeft());
        painter.scale(m_scale, m_scale);
        painter.setClipRect(QRectF(QPointF(0, 0), m_result.image.size()));
        paintAnnotations(painter, m_scale, false);
        painter.restore();

        painter.setPen(QPen(QColor("#626262"), 1));
        painter.setBrush(Qt::NoBrush);
        const QRectF corners = target.adjusted(-5, -5, 5, 5);
        drawCorner(painter, corners.topLeft(), 1, 1);
        drawCorner(painter, corners.topRight(), -1, 1);
        drawCorner(painter, corners.bottomLeft(), 1, -1);
        drawCorner(painter, corners.bottomRight(), -1, -1);

        if (m_result.demonstration)
        {
            const QString text = QStringLiteral("示范场景 · 非模型推理结果");
            painter.setFont(canvasFont(9, QFont::DemiBold));
            const qreal badgeWidth = painter.fontMetrics().horizontalAdvance(text) + 24;
            const QRectF badge(18, 18, badgeWidth, 30);
            painter.setPen(QPen(QColor("#454545"), 1));
            painter.setBrush(QColor("#272727"));
            painter.drawRoundedRect(badge, 6, 6);
            painter.setPen(QColor("#B8B8B8"));
            painter.drawText(badge, Qt::AlignCenter, text);
        }
    }

    if (m_dropActive)
    {
        painter.fillRect(rect(), QColor(96, 205, 255, 18));
        painter.setPen(QPen(QColor("#60CDFF"), 2, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(QRectF(rect()).adjusted(10, 10, -10, -10), 12, 12);
    }
    if (hasFocus())
    {
        painter.setPen(QPen(QColor("#60CDFF"), 1));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect().adjusted(0, 0, -1, -1));
    }
}

void ImageCanvas::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    if (m_fit)
        fitToView();
}

void ImageCanvas::zoomAt(const QPointF &point, double factor)
{
    if (m_result.image.isNull())
        return;
    const QPointF imagePoint = toImage(point);
    m_scale = std::clamp(m_scale * factor, kMinimumScale, kMaximumScale);
    m_fit = false;
    const QSizeF size = QSizeF(m_result.image.size()) * m_scale;
    m_pan =
        point - imagePoint * m_scale - QPointF((width() - size.width()) / 2, (height() - size.height()) / 2);
    emit zoomChanged(qRound(m_scale * 100));
    update();
}

void ImageCanvas::wheelEvent(QWheelEvent *event)
{
    zoomAt(event->position(), std::pow(1.15, event->angleDelta().y() / 120.0));
    event->accept();
}

int ImageCanvas::predictionAt(const QPointF &point) const
{
    if (!m_boxesVisible || m_result.image.isNull())
        return -1;
    const QPointF imagePoint = toImage(point);
    int index = -1;
    qreal smallest = std::numeric_limits<qreal>::max();
    for (int i = 0; i < m_result.predictions.size(); ++i)
    {
        const QRectF &box = m_result.predictions[i].box;
        const qreal area = box.width() * box.height();
        if (box.contains(imagePoint) && area < smallest)
        {
            index = i;
            smallest = area;
        }
    }
    return index;
}

void ImageCanvas::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && !m_result.image.isNull())
    {
        setFocus(Qt::MouseFocusReason);
        m_dragging = true;
        m_moved = false;
        m_pressPosition = event->position();
        m_pressPan = m_pan;
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    }
    else
        QWidget::mousePressEvent(event);
}

void ImageCanvas::mouseMoveEvent(QMouseEvent *event)
{
    if (m_dragging)
    {
        const QPointF movement = event->position() - m_pressPosition;
        if (movement.manhattanLength() > 4)
            m_moved = true;
        if (m_moved)
        {
            m_fit = false;
            m_pan = m_pressPan + movement;
            update();
        }
        event->accept();
    }
    else
    {
        setCursor(!m_result.image.isNull() ? Qt::OpenHandCursor : Qt::ArrowCursor);
        const int index = predictionAt(event->position());
        setToolTip(index >= 0 ? QStringLiteral("%1 · %2% · 点击定位到结果")
                                    .arg(m_result.predictions[index].label)
                                    .arg(qRound(m_result.predictions[index].confidence * 100))
                              : QString());
        QWidget::mouseMoveEvent(event);
    }
}

void ImageCanvas::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && m_dragging)
    {
        m_dragging = false;
        setCursor(Qt::OpenHandCursor);
        if (!m_moved)
        {
            m_selected = predictionAt(event->position());
            emit predictionSelected(m_selected);
            update();
        }
        event->accept();
    }
    else
        QWidget::mouseReleaseEvent(event);
}

void ImageCanvas::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        m_dragging = false;
        fitToView();
        event->accept();
    }
    else
        QWidget::mouseDoubleClickEvent(event);
}

void ImageCanvas::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_F:
    case Qt::Key_0:
        fitToView();
        break;
    case Qt::Key_1:
        actualSize();
        break;
    case Qt::Key_Plus:
    case Qt::Key_Equal:
        zoomAt(rect().center(), 1.2);
        break;
    case Qt::Key_Minus:
        zoomAt(rect().center(), 1 / 1.2);
        break;
    case Qt::Key_Left:
        m_fit = false;
        m_pan += QPointF(32, 0);
        update();
        break;
    case Qt::Key_Right:
        m_fit = false;
        m_pan += QPointF(-32, 0);
        update();
        break;
    case Qt::Key_Up:
        m_fit = false;
        m_pan += QPointF(0, 32);
        update();
        break;
    case Qt::Key_Down:
        m_fit = false;
        m_pan += QPointF(0, -32);
        update();
        break;
    case Qt::Key_Escape:
        m_selected = -1;
        emit predictionSelected(-1);
        update();
        event->ignore(); // Let the main window's task-stop shortcut also handle Escape.
        return;
    default:
        QWidget::keyPressEvent(event);
        return;
    }
    event->accept();
}

QStringList ImageCanvas::localFilesFromDrop(const QMimeData *mime)
{
    QStringList files;
    for (const QUrl &url : mime->urls())
    {
        if (!url.isLocalFile() || url.toLocalFile().isEmpty()) return {};
        files.append(url.toLocalFile());
    }
    return files;
}

void ImageCanvas::setDropActive(bool active)
{
    if (m_dropActive == active) return;
    m_dropActive = active;
    update();
}

void ImageCanvas::dragEnterEvent(QDragEnterEvent *event)
{
    m_dropActive = false;
    event->ignore();
    if (!localFilesFromDrop(event->mimeData()).isEmpty())
    {
        m_dropActive = true;
        event->acceptProposedAction();
        update();
    }
}

void ImageCanvas::dragLeaveEvent(QDragLeaveEvent *event)
{
    m_dropActive = false;
    event->accept();
    update();
}

void ImageCanvas::dropEvent(QDropEvent *event)
{
    m_dropActive = false;
    event->ignore();
    const QStringList files = localFilesFromDrop(event->mimeData());
    if (!files.isEmpty())
    {
        emit filesDropped(files);
        if (files.size() == 1) emit fileDropped(files.first());
        event->acceptProposedAction();
    }
    update();
}

QImage ImageCanvas::annotatedImage() const
{
    return annotatedResultImage(m_result, m_boxesVisible, m_labelsVisible);
}

QImage ImageCanvas::annotatedResultImage(const vision::InferenceResult &result,
                                         bool boxesVisible, bool labelsVisible)
{
    return vision::app::AnnotationRenderer::render(result, boxesVisible, labelsVisible);
}

vision::InferenceResult ImageCanvas::createDemoResult()
{
    vision::InferenceResult result;
    result.image = QImage(1280, 800, QImage::Format_RGB32);
    result.source = QStringLiteral("示范场景");
    result.modelName = QStringLiteral("UI 示范数据");
    result.demonstration = true;
    QPainter painter(&result.image);
    painter.setRenderHint(QPainter::Antialiasing);
    QLinearGradient backdrop(0, 0, 0, 800);
    backdrop.setColorAt(0, QColor("#243a4a"));
    backdrop.setColorAt(0.58, QColor("#1c2d3d"));
    backdrop.setColorAt(1, QColor("#0d1b2b"));
    painter.fillRect(result.image.rect(), backdrop);

    // A deliberately illustrated laboratory scene provides an honest first-run preview.
    painter.setPen(QPen(QColor("#385260"), 2));
    for (int x = 80; x < 1280; x += 160)
        painter.drawLine(x, 0, x, 460);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#162936"));
    painter.drawRect(0, 80, 1280, 15);
    painter.drawRect(0, 260, 1280, 14);
    painter.setBrush(QColor("#405862"));
    painter.drawRoundedRect(92, 52, 255, 6, 3, 3);
    painter.drawRoundedRect(530, 52, 255, 6, 3, 3);
    painter.drawRoundedRect(961, 52, 230, 6, 3, 3);
    painter.setFont(canvasFont(17, QFont::DemiBold));
    painter.setPen(QColor("#93abb7"));
    painter.drawText(QRectF(76, 117, 450, 38), QStringLiteral("VISION / INSPECTION BAY"));
    painter.setFont(canvasFont(10));
    painter.setPen(QColor("#6d8797"));
    painter.drawText(QRectF(78, 163, 420, 25), QStringLiteral("ILLUSTRATED SAMPLE  •  1280 × 800"));
    painter.setPen(QPen(QColor("#2d4655"), 1));
    for (int x = -400; x < 1600; x += 160)
        painter.drawLine(QPointF(640, 400), QPointF(x, 800));
    for (int y : {485, 555, 640, 747})
        painter.drawLine(0, y, 1280, y);

    // Conveyor platform.
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#081422"));
    painter.drawEllipse(QRectF(118, 626, 1040, 90));
    QPainterPath platform;
    platform.moveTo(133, 526);
    platform.lineTo(1084, 526);
    platform.lineTo(1174, 662);
    platform.lineTo(207, 662);
    platform.closeSubpath();
    painter.setBrush(QColor("#425a68"));
    painter.drawPath(platform);
    QPainterPath front;
    front.moveTo(207, 662);
    front.lineTo(1174, 662);
    front.lineTo(1174, 691);
    front.lineTo(207, 691);
    front.closeSubpath();
    painter.setBrush(QColor("#1d3447"));
    painter.drawPath(front);
    painter.setPen(QPen(QColor("#728692"), 7));
    for (int x = 168; x < 1100; x += 43)
        painter.drawLine(QPointF(x, 539), QPointF(x + 78, 648));
    painter.setPen(QPen(QColor("#182d3d"), 4));
    for (int x = 180; x < 1120; x += 43)
        painter.drawLine(QPointF(x, 539), QPointF(x + 78, 648));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#152a3a"));
    painter.drawRect(236, 691, 16, 69);
    painter.drawRect(1069, 691, 16, 69);
    painter.setBrush(QColor("#3a8292"));
    painter.drawRoundedRect(878, 670, 246, 6, 3, 3);

    auto package =
        [&painter](qreal x, qreal y, qreal w, qreal h, qreal depth, const QColor &base, const QString &id)
    {
        QPainterPath top;
        top.moveTo(x, y);
        top.lineTo(x + depth, y - depth * .64);
        top.lineTo(x + w + depth, y - depth * .64);
        top.lineTo(x + w, y);
        top.closeSubpath();
        QPainterPath side;
        side.moveTo(x + w, y);
        side.lineTo(x + w + depth, y - depth * .64);
        side.lineTo(x + w + depth, y + h - depth * .64);
        side.lineTo(x + w, y + h);
        side.closeSubpath();
        painter.setPen(QPen(base.darker(165), 1));
        painter.setBrush(base.lighter(114));
        painter.drawPath(top);
        painter.setBrush(base.darker(128));
        painter.drawPath(side);
        QLinearGradient surface(x, y, x + w, y + h);
        surface.setColorAt(0, base);
        surface.setColorAt(1, base.darker(111));
        painter.setBrush(surface);
        painter.drawRect(QRectF(x, y, w, h));
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor("#d4b485"));
        painter.drawRect(QRectF(x + w * .43, y, w * .12, h));
        QPainterPath tape;
        tape.moveTo(x + w * .43, y);
        tape.lineTo(x + w * .43 + depth, y - depth * .64);
        tape.lineTo(x + w * .55 + depth, y - depth * .64);
        tape.lineTo(x + w * .55, y);
        tape.closeSubpath();
        painter.drawPath(tape);
        painter.setBrush(QColor("#e1ded0"));
        const QRectF label(x + w * .07, y + h * .22, w * .25, h * .32);
        painter.drawRoundedRect(label, 2, 2);
        painter.setPen(QColor("#555955"));
        painter.setFont(canvasFont(6, QFont::DemiBold));
        painter.drawText(label.adjusted(5, 3, -3, -3), Qt::AlignTop | Qt::AlignLeft, id);
        painter.setPen(QPen(QColor("#52554e"), 1));
        for (int i = 0; i < 12; ++i)
        {
            const qreal px = label.left() + 5 + i * 2.5;
            painter.drawLine(QPointF(px, label.bottom() - 12), QPointF(px, label.bottom() - 5));
        }
        painter.setPen(QPen(base.darker(140), 2));
        painter.drawLine(QPointF(x + w * .7, y + h * .79), QPointF(x + w * .7, y + h * .63));
        painter.drawLine(QPointF(x + w * .82, y + h * .79), QPointF(x + w * .82, y + h * .63));
        painter.drawLine(QPointF(x + w * .66, y + h * .67), QPointF(x + w * .7, y + h * .63));
        painter.drawLine(QPointF(x + w * .78, y + h * .67), QPointF(x + w * .82, y + h * .63));
    };
    package(237, 374, 218, 188, 59, QColor("#b49a76"), QStringLiteral("PKG-01"));
    package(555, 333, 184, 251, 67, QColor("#ad8b63"), QStringLiteral("PKG-02"));
    package(869, 413, 185, 147, 53, QColor("#b6a48b"), QStringLiteral("PKG-03"));
    result.predictions = {{0, QStringLiteral("包裹 A"), 0.96f, QRectF(229, 329, 293, 241)},
                          {1, QStringLiteral("包裹 B"), 0.93f, QRectF(547, 282, 267, 309)},
                          {2, QStringLiteral("包裹 C"), 0.88f, QRectF(861, 372, 251, 195)}};
    return result;
}
