/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef QmitkCreateMultiLabelSegmentationAction_h
#define QmitkCreateMultiLabelSegmentationAction_h

#include "mitkIContextMenuAction.h"

#include <org_mitk_gui_qt_segmentation_Export.h>

#include "vector"
#include "mitkDataNode.h"

class MITK_QT_SEGMENTATION QmitkCreateMultiLabelSegmentationAction : public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:

  QmitkCreateMultiLabelSegmentationAction();
  ~QmitkCreateMultiLabelSegmentationAction() override;

  //interface methods
  void Run( const QList<mitk::DataNode::Pointer>& selectedNodes ) override;
  void SetDataStorage(mitk::DataStorage* dataStorage) override;
  void SetFunctionality(berry::QtViewPart* functionality) override;
  void SetSmoothed(bool smoothed) override;
  void SetDecimated(bool decimated) override;

private:

  typedef QList<mitk::DataNode::Pointer> NodeList;

  mitk::DataStorage::Pointer m_DataStorage;
};

#endif
