/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <QmitkDataNodeTextureInterpolationAction.h>

// mitk core
#include <mitkRenderingManager.h>

#include <QWidget>

QmitkDataNodeTextureInterpolationAction::QmitkDataNodeTextureInterpolationAction(QWidget* parent, berry::IWorkbenchPartSite::Pointer workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(workbenchpartSite)
{
  setText(tr("Texture Interpolation"));
  InitializeAction();
}

QmitkDataNodeTextureInterpolationAction::QmitkDataNodeTextureInterpolationAction(QWidget* parent, berry::IWorkbenchPartSite* workbenchpartSite)
  : QAction(parent)
  , QmitkAbstractDataNodeAction(berry::IWorkbenchPartSite::Pointer(workbenchpartSite))
{
  setText(tr("Texture Interpolation"));
  InitializeAction();
}

void QmitkDataNodeTextureInterpolationAction::InitializeAction()
{
  setCheckable(true);

  connect(this, &QmitkDataNodeTextureInterpolationAction::toggled, this, &QmitkDataNodeTextureInterpolationAction::OnActionToggled);
  connect(this, &QmitkDataNodeTextureInterpolationAction::changed, this, &QmitkDataNodeTextureInterpolationAction::OnActionChanged);
}

void QmitkDataNodeTextureInterpolationAction::InitializeWithDataNode(const mitk::DataNode* dataNode)
{
  if (nullptr == dataNode)
  {
    setChecked(false);
    return;
  }

  mitk::BaseRenderer::Pointer baseRenderer = GetBaseRenderer();

  bool textureInterpolation = false;
  dataNode->GetBoolProperty("texture interpolation", textureInterpolation, baseRenderer);
  setChecked(textureInterpolation);
}

void QmitkDataNodeTextureInterpolationAction::OnActionToggled(bool checked)
{
  auto dataNode = GetSelectedNode();
  if (dataNode.IsNull())
  {
    return;
  }

  mitk::BaseRenderer::Pointer baseRenderer = GetBaseRenderer();
  dataNode->SetBoolProperty("texture interpolation", checked, baseRenderer);

  if (nullptr == baseRenderer)
  {
    mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  }
  else
  {
    mitk::RenderingManager::GetInstance()->RequestUpdate(baseRenderer->GetRenderWindow());
  }
}

void QmitkDataNodeTextureInterpolationAction::OnActionChanged()
{
  auto dataNode = GetSelectedNode();
  if (dataNode.IsNull())
  {
    return;
  }

  InitializeWithDataNode(dataNode);
}
