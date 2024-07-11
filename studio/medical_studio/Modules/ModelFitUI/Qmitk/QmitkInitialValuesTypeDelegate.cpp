/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkInitialValuesTypeDelegate.h"

#include <QPainter>
#include <QApplication>
#include <QLabel>
#include <QComboBox>

QmitkInitialValuesTypeDelegate::QmitkInitialValuesTypeDelegate(QObject* /*parent*/)
{
}

QWidget* QmitkInitialValuesTypeDelegate::createEditor(QWidget* parent,
    const QStyleOptionViewItem&
    , const QModelIndex& index) const
{
  QVariant displayData = index.data(Qt::DisplayRole);
  QVariant data = index.data(Qt::EditRole);

  if (data.isValid())
  {
    QComboBox* cmbBox = new QComboBox(parent);
    cmbBox->addItem(QString("scalar"));
    cmbBox->addItem(QString("image"));
    cmbBox->setCurrentIndex(0);

    cmbBox->installEventFilter(const_cast<QmitkInitialValuesTypeDelegate*>(this));

    return cmbBox;
  }
  else
  {
    return new QLabel(displayData.toString(), parent);
  }

}

void QmitkInitialValuesTypeDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
  QVariant data = index.data(Qt::EditRole);

  if (data.isValid())
  {

    QComboBox* cmbBox = qobject_cast<QComboBox*>(editor);

    if (cmbBox)
    {
      cmbBox->setCurrentIndex(data.toInt());
    }
    else
    {
      QStyledItemDelegate::setEditorData(editor, index);
    }
  }
}

void QmitkInitialValuesTypeDelegate::setModelData(QWidget* editor, QAbstractItemModel* model
    , const QModelIndex& index) const
{
  QVariant data = index.data(Qt::EditRole);

  if (data.isValid())
  {
    QComboBox* cmbBox = qobject_cast<QComboBox*>(editor);
    int selection = cmbBox->currentIndex();

    QVariant selectionVariant(selection);
    model->setData(index, selectionVariant);
  }
  else
  {
    QStyledItemDelegate::setModelData(editor, model, index);
  }
}
