/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <QmitkAbstractDataNodeAction.h>

#include "mitkIRenderWindowPart.h"

// mitk gui common plugin
#include <mitkDataNodeSelection.h>

// berry
#include <berryIWorkbenchPage.h>

QList<mitk::DataNode::Pointer> AbstractDataNodeAction::GetSelectedNodes(berry::IWorkbenchPartSite::Pointer workbenchPartSite)
{
  QList<mitk::DataNode::Pointer> selectedNodes;
  if (workbenchPartSite.IsNull())
  {
    return selectedNodes;
  }

  berry::ISelection::ConstPointer selection = workbenchPartSite->GetWorkbenchWindow()->GetSelectionService()->GetSelection();
  mitk::DataNodeSelection::ConstPointer currentSelection = selection.Cast<const mitk::DataNodeSelection>();
  if (currentSelection.IsNull() || currentSelection->IsEmpty())
  {
    return selectedNodes;
  }

  auto nodes = currentSelection->GetSelectedDataNodes();
  selectedNodes = QList<mitk::DataNode::Pointer>(nodes.begin(), nodes.end());
  return selectedNodes;
}

QmitkAbstractDataNodeAction::QmitkAbstractDataNodeAction(berry::IWorkbenchPartSite::Pointer workbenchPartSite)
{
  m_WorkbenchPartSite = workbenchPartSite;
}

QmitkAbstractDataNodeAction::QmitkAbstractDataNodeAction(berry::IWorkbenchPartSite* workbenchPartSite)
{
  m_WorkbenchPartSite = berry::IWorkbenchPartSite::Pointer(workbenchPartSite);
}

void QmitkAbstractDataNodeAction::SetDataStorage(mitk::DataStorage* dataStorage)
{
  if (m_DataStorage != dataStorage)
  {
    // set the new data storage
    m_DataStorage = dataStorage;
  }
}

void QmitkAbstractDataNodeAction::SetSelectedNodes(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  m_SelectedNodes = selectedNodes;
  // use the first selected node to initialize the data node actions
  InitializeWithDataNode(m_SelectedNodes.front());
}

void QmitkAbstractDataNodeAction::SetBaseRenderer(mitk::BaseRenderer* baseRenderer)
{
  if (m_BaseRenderer != baseRenderer)
  {
    // set the new base renderer
    m_BaseRenderer = baseRenderer;
  }
}

mitk::BaseRenderer::Pointer QmitkAbstractDataNodeAction::GetBaseRenderer()
{
  return m_BaseRenderer.Lock();
}

QList<mitk::DataNode::Pointer> QmitkAbstractDataNodeAction::GetSelectedNodes() const
{
  if (!m_SelectedNodes.isEmpty())
  {
    return m_SelectedNodes;
  }

  auto workbenchPartSite = m_WorkbenchPartSite.Lock();

  if (workbenchPartSite.IsNull())
  {
    // return empty list of selected nodes
    return m_SelectedNodes;
  }

  // retrieve selection from the workbench selection service
  return AbstractDataNodeAction::GetSelectedNodes(workbenchPartSite);
}

mitk::DataNode::Pointer QmitkAbstractDataNodeAction::GetSelectedNode() const
{
  QList<mitk::DataNode::Pointer> selectedNodes = GetSelectedNodes();
  if (selectedNodes.empty())
  {
    return nullptr;
  }

  // no batch action; should only be called with a single node
  mitk::DataNode::Pointer dataNode = selectedNodes.front();
  if (nullptr == dataNode)
  {
    return nullptr;
  }

  return dataNode;
}
