/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <QmitkDataNodeShowSelectedNodesAction.h>

// mitk core
#include <mitkRenderingManager.h>

#include <QWidget>

QmitkDataNodeShowSelectedNodesAction::QmitkDataNodeShowSelectedNodesAction(QWidget* parent, berry::IWorkbenchPartSite::Pointer workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(workbenchpartSite)
{
  setText(tr("Show only selected nodes"));
  InitializeAction();
}

QmitkDataNodeShowSelectedNodesAction::QmitkDataNodeShowSelectedNodesAction(QWidget* parent, berry::IWorkbenchPartSite* workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(berry::IWorkbenchPartSite::Pointer(workbenchpartSite))
{
  setText(tr("Show only selected nodes"));
  InitializeAction();
}

void QmitkDataNodeShowSelectedNodesAction::InitializeAction()
{
  connect(this, &QmitkDataNodeShowSelectedNodesAction::triggered, this, &QmitkDataNodeShowSelectedNodesAction::OnActionTriggered);
}

void QmitkDataNodeShowSelectedNodesAction::OnActionTriggered(bool /*checked*/)
{
  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  mitk::BaseRenderer::Pointer baseRenderer = GetBaseRenderer();

  auto selectedNodes = GetSelectedNodes();

  auto allNodes = dataStorage->GetAll();

  for (auto& node : *allNodes)
  {
    if (node.IsNotNull() && node->GetData() != nullptr && strcmp(node->GetData()->GetNameOfClass(), "PlaneGeometryData"))
    {
      node->SetVisibility(selectedNodes.contains(node), baseRenderer);
    }
  }

  if (nullptr == baseRenderer)
  {
    mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  }
  else
  {
    mitk::RenderingManager::GetInstance()->RequestUpdate(baseRenderer->GetRenderWindow());
  }
}
