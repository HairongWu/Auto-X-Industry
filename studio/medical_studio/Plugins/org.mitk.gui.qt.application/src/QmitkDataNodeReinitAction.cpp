/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <QmitkDataNodeReinitAction.h>

// mitk core
#include <mitkImage.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkRenderingManager.h>

// mitk gui common plugin
#include <mitkWorkbenchUtil.h>

#include <QWidget>

// namespace that contains the concrete action
namespace ReinitAction
{
  void Run(berry::IWorkbenchPartSite::Pointer workbenchPartSite, mitk::DataStorage::Pointer dataStorage, const QList<mitk::DataNode::Pointer>& selectedNodes /*= QList<mitk::DataNode::Pointer>()*/, mitk::BaseRenderer* baseRenderer /*= nullptr*/)
  {
    if (selectedNodes.empty())
    {
      return;
    }

    if (workbenchPartSite.IsNotNull())
    {
      auto renderWindow = mitk::WorkbenchUtil::GetRenderWindowPart(workbenchPartSite->GetPage(), mitk::WorkbenchUtil::NONE);
      if (nullptr == renderWindow)
      {
        renderWindow = mitk::WorkbenchUtil::OpenRenderWindowPart(workbenchPartSite->GetPage(), false);
        if (nullptr == renderWindow)
        {
          // no render window available
          return;
        }
      }
    }


    auto boundingBoxPredicate = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox", mitk::BoolProperty::New(false), baseRenderer));

    mitk::DataStorage::SetOfObjects::Pointer nodes = mitk::DataStorage::SetOfObjects::New();
    for (const auto& dataNode : selectedNodes)
    {
      if (boundingBoxPredicate->CheckNode(dataNode))
      {
        nodes->InsertElement(nodes->Size(), dataNode);
      }
    }

    if (nodes->empty())
    {
      return;
    }

    if (1 == nodes->Size()) // Special case: If exactly one ...
    {
      auto image = dynamic_cast<mitk::Image*>(nodes->ElementAt(0)->GetData());

      if (nullptr != image) // ... image is selected, reinit is expected to rectify askew images.
      {
        if (nullptr == baseRenderer)
        {
          mitk::RenderingManager::GetInstance()->InitializeViews(image->GetTimeGeometry());
        }
        else
        {
          mitk::RenderingManager::GetInstance()->InitializeView(baseRenderer->GetRenderWindow(), image->GetTimeGeometry());
        }
        return;
      }
    }

    auto boundingGeometry = dataStorage->ComputeBoundingGeometry3D(nodes, "visible", baseRenderer);
    if (nullptr == baseRenderer)
    {
      mitk::RenderingManager::GetInstance()->InitializeViews(boundingGeometry);
    }
    else
    {
      mitk::RenderingManager::GetInstance()->InitializeView(baseRenderer->GetRenderWindow(), boundingGeometry);
    }
  }
}

QmitkDataNodeReinitAction::QmitkDataNodeReinitAction(QWidget* parent, berry::IWorkbenchPartSite::Pointer workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(workbenchpartSite)
{
  setText(tr("Reinit"));
  InitializeAction();
}

QmitkDataNodeReinitAction::QmitkDataNodeReinitAction(QWidget* parent, berry::IWorkbenchPartSite* workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(berry::IWorkbenchPartSite::Pointer(workbenchpartSite))
{
  setText(tr("Reinit"));
  InitializeAction();
}

void QmitkDataNodeReinitAction::InitializeAction()
{
  connect(this, &QmitkDataNodeReinitAction::triggered, this, &QmitkDataNodeReinitAction::OnActionTriggered);
}

void QmitkDataNodeReinitAction::OnActionTriggered(bool /*checked*/)
{
  auto workbenchPartSite = m_WorkbenchPartSite.Lock();

  if (workbenchPartSite.IsNull())
  {
    return;
  }

  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  mitk::BaseRenderer::Pointer baseRenderer = GetBaseRenderer();

  auto selectedNodes = GetSelectedNodes();
  ReinitAction::Run(workbenchPartSite, dataStorage, selectedNodes, baseRenderer);
}
