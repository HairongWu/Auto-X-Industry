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
#include <vtkRegressionTestImage.h>

int mitkImageVtkMapper2DOpacityTest(int argc, char *argv[])
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
  MITK_TEST_BEGIN("mitkImageVtkMapper2DTest")

  mitk::RenderingTestHelper renderingHelper(640, 480, argc, argv);
  // Set the opacity for all images
  renderingHelper.SetImageProperty("opacity", mitk::FloatProperty::New(0.5f));
  // for now this test renders in coronal view direction
  renderingHelper.SetViewDirection(mitk::AnatomicalPlane::Coronal);

  //### Usage of CompareRenderWindowAgainstReference: See docu of mitkRrenderingTestHelper
  MITK_TEST_CONDITION(renderingHelper.CompareRenderWindowAgainstReference(argc, argv) == true,
                      "CompareRenderWindowAgainstReference test result positive?");

  // use this to generate a reference screenshot or save the file:
  if (false)
  {
    renderingHelper.SaveReferenceScreenShot("d:/tmp/renderingtest.png");
  }

  MITK_TEST_END();
}
