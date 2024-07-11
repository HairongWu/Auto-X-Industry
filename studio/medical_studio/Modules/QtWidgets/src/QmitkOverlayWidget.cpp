/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkOverlayWidget.h"
#include <QPainter>
#include <QResizeEvent>

QmitkOverlayWidget::QmitkOverlayWidget(QWidget* parent)
  : QWidget(parent)
{
  this->installEventFilterOnParent();
  this->setTransparentForMouseEvents();
  this->setAttribute(Qt::WA_TranslucentBackground);
}

QmitkOverlayWidget::~QmitkOverlayWidget()
{
  this->removeEventFilterFromParent();
}

bool QmitkOverlayWidget::isTransparentForMouseEvents() const
{
  return this->testAttribute(Qt::WA_TransparentForMouseEvents);
}

void QmitkOverlayWidget::setTransparentForMouseEvents(bool transparent)
{
  this->setAttribute(Qt::WA_TransparentForMouseEvents, transparent);
}

bool QmitkOverlayWidget::event(QEvent* e)
{
  if (e->type() == QEvent::ParentAboutToChange)
  {
    this->removeEventFilterFromParent();
  }
  else if (e->type() == QEvent::ParentChange)
  {
    this->installEventFilterOnParent();
  }

  return QWidget::event(e);
}

bool QmitkOverlayWidget::eventFilter(QObject* watched, QEvent* event)
{
  if (watched == this->parent())
  {
    if (event->type() == QEvent::Resize)
    {
      this->resize(static_cast<QResizeEvent*>(event)->size());
    }
    else if (event->type() == QEvent::ChildAdded)
    {
      this->raise();
    }
  }

  return QWidget::eventFilter(watched, event);
}

void QmitkOverlayWidget::paintEvent(QPaintEvent*)
{
  QPainter painter(this);
  painter.fillRect(this->rect(), QColor(0, 0, 0, 63));
}

void QmitkOverlayWidget::installEventFilterOnParent()
{
  if (this->parent() == nullptr)
    return;

  this->parent()->installEventFilter(this);
  this->raise();
}

void QmitkOverlayWidget::removeEventFilterFromParent()
{
  if (this->parent() == nullptr)
    return;

  this->parent()->removeEventFilter(this);
}
