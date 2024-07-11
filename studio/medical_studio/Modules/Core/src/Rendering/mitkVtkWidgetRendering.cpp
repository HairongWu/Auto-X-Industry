/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkVtkWidgetRendering.h"

#include "mitkVtkLayerController.h"

#include <itkMacro.h>
#include <itkObject.h>
#include <itksys/SystemTools.hxx>
#include <mitkConfig.h>

#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>

#include <vtkInteractorObserver.h>

#include <algorithm>

mitk::VtkWidgetRendering::VtkWidgetRendering() : m_RenderWindow(nullptr), m_VtkWidget(nullptr), m_IsEnabled(false)
{
  m_Renderer = vtkRenderer::New();
}

mitk::VtkWidgetRendering::~VtkWidgetRendering()
{
  if (m_RenderWindow != nullptr)
    if (this->IsEnabled())
      this->Disable();

  if (m_Renderer != nullptr)
    m_Renderer->Delete();
}

/**
 * Sets the renderwindow, in which the widget
 * will be shown. Make sure, you have called this function
 * before calling Enable()
 */
void mitk::VtkWidgetRendering::SetRenderWindow(vtkRenderWindow *renderWindow)
{
  m_RenderWindow = renderWindow;
}

/**
 * Returns the vtkRenderWindow, which is used
 * for displaying the widget
 */
vtkRenderWindow *mitk::VtkWidgetRendering::GetRenderWindow()
{
  return m_RenderWindow;
}

/**
 * Returns the renderer responsible for
 * rendering the  widget into the
 * vtkRenderWindow
 */
vtkRenderer *mitk::VtkWidgetRendering::GetVtkRenderer()
{
  return m_Renderer;
}

/**
 * Enables drawing of the widget.
 * If you want to disable it, call the Disable() function.
 */
void mitk::VtkWidgetRendering::Enable()
{
  if (m_IsEnabled)
    return;

  if (m_RenderWindow != nullptr)
  {
    vtkRenderWindowInteractor *interactor = m_RenderWindow->GetInteractor();

    if (m_VtkWidget != nullptr)
    {
      m_VtkWidget->SetInteractor(interactor);

      mitk::VtkLayerController *layerController = mitk::VtkLayerController::GetInstance(m_RenderWindow);

      if (layerController)
      {
        layerController->InsertForegroundRenderer(m_Renderer, false);
      }

      m_IsEnabled = true;
    }
  }
}

/**
 * Disables drawing of the widget.
 * If you want to enable it, call the Enable() function.
 */
void mitk::VtkWidgetRendering::Disable()
{
  if (this->IsEnabled())
  {
    mitk::VtkLayerController *layerController = mitk::VtkLayerController::GetInstance(m_RenderWindow);

    if (layerController)
    {
      layerController->RemoveRenderer(m_Renderer);
    }

    m_IsEnabled = false;
  }
}

/**
 * Checks, if the widget is currently
 * enabled (visible)
 */
bool mitk::VtkWidgetRendering::IsEnabled()
{
  return m_IsEnabled;
}

void mitk::VtkWidgetRendering::SetRequestedRegionToLargestPossibleRegion()
{
  // nothing to do
}

bool mitk::VtkWidgetRendering::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  return false;
}

bool mitk::VtkWidgetRendering::VerifyRequestedRegion()
{
  return true;
}

void mitk::VtkWidgetRendering::SetRequestedRegion(const itk::DataObject *)
{
  // nothing to do
}

void mitk::VtkWidgetRendering::SetVtkWidget(vtkInteractorObserver *widget)
{
  m_VtkWidget = widget;
}

vtkInteractorObserver *mitk::VtkWidgetRendering::GetVtkWidget() const
{
  return m_VtkWidget;
}
