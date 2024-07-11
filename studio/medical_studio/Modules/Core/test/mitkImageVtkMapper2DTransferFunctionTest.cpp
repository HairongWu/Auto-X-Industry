/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK
#include "mitkRenderingTestHelper.h"
#include "mitkTestingMacros.h"

// VTK
#include <mitkRenderingModeProperty.h>
#include <mitkTransferFunction.h>
#include <mitkTransferFunctionProperty.h>
#include <vtkRegressionTestImage.h>

int mitkImageVtkMapper2DTransferFunctionTest(int argc, char *argv[])
{
  try
  {
    mitk::RenderingTestHelper openGlTest(640, 480);
  }
  catch (const mitk::TestNotRunException &e)
  {
    MITK_WARN << "Test not run: " << e.GetDescription();
    return 77;
  }
  // load all arguments into a datastorage, take last argument as reference rendering
  // setup a renderwindow of fixed size X*Y
  // render the datastorage
  // compare rendering to reference image
  MITK_TEST_BEGIN("mitkImageVtkMapper2DTransferFunctionTest")

  mitk::RenderingTestHelper renderingHelper(640, 480, argc, argv);

  // define an arbitrary colortransferfunction
  vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
  colorTransferFunction->SetColorSpaceToRGB();
  colorTransferFunction->AddRGBPoint(0.0, 1, 0, 0);   // black = red
  colorTransferFunction->AddRGBPoint(127.5, 0, 1, 0); // grey = green
  colorTransferFunction->AddRGBPoint(255.0, 0, 0, 1); // white = blue
  mitk::TransferFunction::Pointer transferFucntion = mitk::TransferFunction::New();
  transferFucntion->SetColorTransferFunction(colorTransferFunction);

  // set the rendering mode to use the transfer function
  renderingHelper.SetImageProperty(
    "Image Rendering.Mode", mitk::RenderingModeProperty::New(mitk::RenderingModeProperty::COLORTRANSFERFUNCTION_COLOR));
  // set the property for the image
  renderingHelper.SetImageProperty("Image Rendering.Transfer Function",
                                   mitk::TransferFunctionProperty::New(transferFucntion));

  //### Usage of CompareRenderWindowAgainstReference: See docu of mitkRrenderingTestHelper
  MITK_TEST_CONDITION(renderingHelper.CompareRenderWindowAgainstReference(argc, argv, 20.0) == true,
                      "CompareRenderWindowAgainstReference test result positive?");

  // use this to generate a reference screenshot or save the file:
  if (false)
  {
    renderingHelper.SaveReferenceScreenShot("/media/hdd/thomasHdd/Pictures/tmp/output2.png");
  }

  MITK_TEST_END();
}
