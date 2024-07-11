/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "QmitkPropertyDelegate.h"

#include "QmitkCustomVariants.h"

#include "mitkRenderingManager.h"

#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <QStringList>
#include <bitset>

QmitkPropertyDelegate::QmitkPropertyDelegate(QObject * /*parent*/)
{
}

void QmitkPropertyDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
  QVariant data = index.data(Qt::DisplayRole);

  QString name = data.value<QString>();

  if (index.column() == 1 && data.typeId() == QMetaType::QColor)
  {
    QColor qcol = data.value<QColor>();

    painter->save();
    painter->fillRect(option.rect, qcol);
    QRect rect = option.rect;
    rect.setWidth(rect.width() - 1);
    rect.setHeight(rect.height() - 1);
    QPen pen;
    pen.setWidth(1);
    painter->setPen(pen);
    painter->drawRect(rect);
    painter->restore();
  }
  else
  {
    QStyledItemDelegate::paint(painter, option, index);
  }
}

QWidget *QmitkPropertyDelegate::createEditor(QWidget *parent,
                                             const QStyleOptionViewItem &option,
                                             const QModelIndex &index) const
{
  QVariant data = index.data(Qt::EditRole);
  QVariant displayData = index.data(Qt::DisplayRole);
  QString name = index.model()->data(index.model()->index(index.row(), index.column() - 1)).value<QString>();

  if (data.isValid())
  {
    QWidget *editorWidget = nullptr;

    if (data.typeId() == QMetaType::QColor)
    {
      auto colorBtn = new QPushButton(parent);
      QColor color = data.value<QColor>();

      QColor result = QColorDialog::getColor(color);
      if (result.isValid())
      {
        QPalette palette = colorBtn->palette();
        palette.setColor(QPalette::Button, result);
        colorBtn->setPalette(palette);
        colorBtn->setStyleSheet(
          QString("background-color: %1;foreground-color: %1; border-style: none;").arg(result.name()));
      }
      // QColorDialog closed by 'Cancel' button, use the old property color
      else
      {
        QPalette palette = colorBtn->palette();
        palette.setColor(QPalette::Button, color);
        colorBtn->setPalette(palette);
        colorBtn->setStyleSheet(
          QString("background-color: %1;foreground-color: %1; border-style: none;").arg(color.name()));
      }

      connect(colorBtn, SIGNAL(pressed()), this, SLOT(commitAndCloseEditor()));

      editorWidget = colorBtn;
    }

    else if (data.typeId() == QMetaType::Int)
    {
      auto spinBox = new QSpinBox(parent);
      spinBox->setSingleStep(1);
      spinBox->setMinimum(std::numeric_limits<int>::min());
      spinBox->setMaximum(std::numeric_limits<int>::max());
      editorWidget = spinBox;
    }

    else if (data.typeId() == QMetaType::Float)
    {
      auto spinBox = new QDoubleSpinBox(parent);
      spinBox->setDecimals(2);
      spinBox->setSingleStep(0.1);
      if (name == "opacity")
      {
        spinBox->setMinimum(0.0);
        spinBox->setMaximum(1.0);
      }
      else
      {
        spinBox->setMinimum(std::numeric_limits<float>::min());
        spinBox->setMaximum(std::numeric_limits<float>::max());
      }

      editorWidget = spinBox;
    }

    else if (data.typeId() == QMetaType::QStringList)
    {
      QStringList entries = data.value<QStringList>();
      auto comboBox = new QComboBox(parent);
      comboBox->setEditable(false);
      comboBox->addItems(entries);

      editorWidget = comboBox;
    }

    else
    {
      editorWidget = QStyledItemDelegate::createEditor(parent, option, index);
    }

    if (editorWidget)
    {
      // install event filter
      editorWidget->installEventFilter(const_cast<QmitkPropertyDelegate *>(this));
    }

    return editorWidget;
  }
  else
    return new QLabel(displayData.toString(), parent);
}

void QmitkPropertyDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
  QVariant data = index.data(Qt::EditRole);
  QVariant displayData = index.data(Qt::DisplayRole);

  if (data.isValid())
  {
    if (data.typeId() == QMetaType::Int)
    {
      QSpinBox *spinBox = qobject_cast<QSpinBox *>(editor);
      spinBox->setValue(data.toInt());
    }
    // see qt documentation. cast is correct, it would be obsolete if we
    // store doubles
    else if (data.typeId() == QMetaType::Float)
    {
      QDoubleSpinBox *spinBox = qobject_cast<QDoubleSpinBox *>(editor);
      spinBox->setValue(data.toDouble());
    }

    else if (data.typeId() == QMetaType::QStringList)
    {
      QComboBox *comboBox = qobject_cast<QComboBox *>(editor);
      QString displayString = displayData.value<QString>();
      comboBox->setCurrentIndex(comboBox->findData(displayString));
    }

    else
      return QStyledItemDelegate::setEditorData(editor, index);
  }
}

void QmitkPropertyDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
  QVariant data = index.data(Qt::EditRole);
  QVariant displayData = index.data(Qt::DisplayRole);

  if (data.isValid())
  {
    if (data.typeId() == QMetaType::QColor)
    {
      QWidget *colorBtn = qobject_cast<QWidget *>(editor);
      QVariant colorVariant;
      colorVariant.setValue(colorBtn->palette().color(QPalette::Button));
      model->setData(index, colorVariant);
    }

    else if (data.typeId() == QMetaType::Int)
    {
      QSpinBox *spinBox = qobject_cast<QSpinBox *>(editor);
      int intValue = spinBox->value();

      QVariant intValueVariant;
      intValueVariant.setValue<float>(static_cast<float>(intValue));
      model->setData(index, intValueVariant);
    }

    else if (data.typeId() == QMetaType::Float)
    {
      QDoubleSpinBox *spinBox = qobject_cast<QDoubleSpinBox *>(editor);
      double doubleValue = spinBox->value();

      QVariant doubleValueVariant;
      doubleValueVariant.setValue<float>(static_cast<float>(doubleValue));
      model->setData(index, doubleValueVariant);
    }

    else if (data.typeId() == QMetaType::QStringList)
    {
      QString displayData = data.value<QString>();

      QComboBox *comboBox = qobject_cast<QComboBox *>(editor);
      QString comboBoxValue = comboBox->currentText();

      QVariant comboBoxValueVariant;
      comboBoxValueVariant.setValue(comboBoxValue);
      model->setData(index, comboBoxValueVariant);
    }

    else
      QStyledItemDelegate::setModelData(editor, model, index);
  }
}

void QmitkPropertyDelegate::commitAndCloseEditor()
{
  QWidget *editor = nullptr;
  if (QPushButton *pushBtn = qobject_cast<QPushButton *>(sender()))
  {
    editor = pushBtn;
  }

  if (editor)
  {
    emit commitData(editor);
    emit closeEditor(editor);
  }
}

void QmitkPropertyDelegate::updateEditorGeometry(QWidget *editor,
                                                 const QStyleOptionViewItem &option,
                                                 const QModelIndex & /*index*/) const
{
  editor->setGeometry(option.rect);
}

void QmitkPropertyDelegate::ComboBoxCurrentIndexChanged(int /*index*/)
{
  if (QComboBox *comboBox = qobject_cast<QComboBox *>(sender()))
  {
    emit commitData(comboBox);
    emit closeEditor(comboBox);
  }
}

void QmitkPropertyDelegate::SpinBoxValueChanged(const QString & /*value*/)
{
  QAbstractSpinBox *spinBox = nullptr;
  if ((spinBox = qobject_cast<QSpinBox *>(sender())) || (spinBox = qobject_cast<QDoubleSpinBox *>(sender())))
  {
    emit commitData(spinBox);
    emit closeEditor(spinBox);
  }
}

void QmitkPropertyDelegate::showColorDialog()
{
}

bool QmitkPropertyDelegate::eventFilter(QObject *o, QEvent *e)
{
  // filter all kind of events on our editor widgets
  // when certain events occur, repaint all render windows, because rendering relevant properties might have changed
  switch (e->type())
  {
    case QEvent::KeyRelease:
    case QEvent::MouseButtonRelease:
    case QEvent::MouseButtonDblClick:
    case QEvent::Wheel:
    case QEvent::FocusIn:
    {
      if (QWidget *editor = dynamic_cast<QWidget *>(o))
      {
        emit commitData(editor);
      }

      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
      break;
    }
    default:
    {
      break;
    }
  }

  return false;
}
