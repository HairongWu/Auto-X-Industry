/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkInspectionPositionWidget.h"

QmitkInspectionPositionWidget::QmitkInspectionPositionWidget(QWidget*)
{
  this->m_Controls.setupUi(this);

  this->m_CurrentPosition.Fill(0.0);

  m_Controls.btnAdd->setEnabled(false);

  connect(m_Controls.pointlistWidget, SIGNAL(PointListChanged()), this,
    SLOT(OnPointListChanged()));
  connect(m_Controls.btnAdd, SIGNAL(clicked()), this,
    SLOT(OnAddCurrentPositionClicked()));
}

QmitkInspectionPositionWidget::~QmitkInspectionPositionWidget()
{
}

mitk::Point3D
QmitkInspectionPositionWidget::
GetCurrentPosition() const
{
  return m_CurrentPosition;
};

const mitk::PointSet*
QmitkInspectionPositionWidget::
GetPositionBookmarks() const
{
  return m_Controls.pointlistWidget->GetPointSet();
};

void
QmitkInspectionPositionWidget::
SetCurrentPosition(const mitk::Point3D& currentPos)
{
  m_CurrentPosition = currentPos;

  std::ostringstream strm;
  strm.imbue(std::locale("C"));
  strm << currentPos[0] << " | " << currentPos[1] << " | " << currentPos[2];
  m_Controls.lineCurrentPos->setText(QString::fromStdString(strm.str()));
};

void
QmitkInspectionPositionWidget::
SetPositionBookmarkNode(mitk::DataNode *newNode)
{
  m_Controls.pointlistWidget->SetPointSetNode(newNode);
  m_Controls.btnAdd->setEnabled(newNode != nullptr);
};

mitk::DataNode *
QmitkInspectionPositionWidget::GetPositionBookmarkNode()
{
  return m_Controls.pointlistWidget->GetPointSetNode();
};

void
QmitkInspectionPositionWidget::
OnPointListChanged()
{
  emit PositionBookmarksChanged();
};

void QmitkInspectionPositionWidget::
OnAddCurrentPositionClicked()
{
  if (m_Controls.pointlistWidget->GetPointSet())
  {
    m_Controls.pointlistWidget->GetPointSet()->InsertPoint(m_CurrentPosition);
    emit PositionBookmarksChanged();
  }
}
