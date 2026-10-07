#pragma once

#include "core/visiontypes.h"
#include <QPointF>
#include <QWidget>

class QDragEnterEvent;
class QDragLeaveEvent;
class QDropEvent;
class QKeyEvent;
class QMimeData;
class QMouseEvent;
class QPainter;
class QResizeEvent;
class QWheelEvent;

class ImageCanvas final : public QWidget
{
    Q_OBJECT
  public:
    explicit ImageCanvas(QWidget *parent = nullptr);

    void setResult(const vision::InferenceResult &result);
    const vision::InferenceResult &result() const
    {
        return m_result;
    }
    void clear();
    void setBoxesVisible(bool visible);
    void setLabelsVisible(bool visible);
    void setSelectedPrediction(int index);
    void fitToView();
    void actualSize();
    QImage annotatedImage() const;
    static vision::InferenceResult createDemoResult();

  signals:
    void fileDropped(const QString &file);
    void zoomChanged(int percent);
    void predictionSelected(int index);

  protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dragLeaveEvent(QDragLeaveEvent *event) override;
    void dropEvent(QDropEvent *event) override;

  private:
    QRectF imageRect() const;
    QPointF toImage(const QPointF &point) const;
    void zoomAt(const QPointF &point, double factor);
    int predictionAt(const QPointF &point) const;
    void paintAnnotations(QPainter &painter, qreal scale, bool exporting) const;
    static QColor classColor(int classId);
    static QString imageFileFromDrop(const QMimeData *mime);

    vision::InferenceResult m_result;
    qreal m_scale = 1.0;
    QPointF m_pan;
    QPointF m_pressPosition;
    QPointF m_pressPan;
    int m_selected = -1;
    bool m_fit = true;
    bool m_boxesVisible = true;
    bool m_labelsVisible = true;
    bool m_dragging = false;
    bool m_moved = false;
    bool m_dropActive = false;
};
