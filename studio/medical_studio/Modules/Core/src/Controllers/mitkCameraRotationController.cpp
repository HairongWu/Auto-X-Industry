/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkCameraRotationController.h"

#include <itkCommand.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include "mitkRenderingManager.h"
#include "mitkVtkPropRenderer.h"

mitk::CameraRotationController::CameraRotationController()
  : BaseController(), m_LastStepperValue(180), m_Camera(nullptr), m_RenderWindow(nullptr)
{
  m_Stepper->SetAutoRepeat(true);
  m_Stepper->SetSteps(360);
  m_Stepper->SetPos(180);

  itk::SimpleMemberCommand<CameraRotationController>::Pointer sliceStepperChangedCommand, timeStepperChangedCommand;
  sliceStepperChangedCommand = itk::SimpleMemberCommand<CameraRotationController>::New();
  sliceStepperChangedCommand->SetCallbackFunction(this, &CameraRotationController::RotateCamera);

  m_Stepper->AddObserver(itk::ModifiedEvent(), sliceStepperChangedCommand);
}

mitk::CameraRotationController::~CameraRotationController()
{
}

void mitk::CameraRotationController::RotateCamera()
{
  if (!m_Camera)
  {
    this->AcquireCamera();
  }

  if (m_Camera)
  {
    int newStepperValue = m_Stepper->GetPos();
    m_Camera->Azimuth(m_LastStepperValue - newStepperValue);
    m_LastStepperValue = newStepperValue;
    // const_cast< RenderWindow* >(m_RenderWindow)->RequestUpdate(); // TODO does not work with movie generator!
    mitk::RenderingManager::GetInstance()->RequestUpdate(m_RenderWindow);
    // m_MultiWidget->RequestUpdate();
  }
}

void mitk::CameraRotationController::AcquireCamera()
{
  BaseRenderer *renderer = mitk::BaseRenderer::GetInstance(m_RenderWindow);

  const auto *propRenderer = dynamic_cast<const mitk::VtkPropRenderer *>(renderer);
  if (propRenderer)
  {
    // get vtk renderer
    vtkRenderer *vtkrenderer = propRenderer->GetVtkRenderer();
    if (vtkrenderer)
    {
      // get vtk camera
      vtkCamera *vtkcam = vtkrenderer->GetActiveCamera();
      if (vtkcam)
      {
        // vtk smart pointer handling
        if (!m_Camera)
        {
          m_Camera = vtkcam;
          m_Camera->Register(nullptr);
        }
        else
        {
          m_Camera->UnRegister(nullptr);
          m_Camera = vtkcam;
          m_Camera->Register(nullptr);
        }
      }
    }
  }
}
