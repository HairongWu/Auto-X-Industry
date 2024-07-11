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
#include <mitkBaseProperty.h>
#include <mitkNodePredicateDataType.h>
#include <mitkSurface.h>

// VTK
#include <vtkRegressionTestImage.h>

int mitkSurfaceDepthSortingTest(int argc, char *argv[])
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
  MITK_TEST_BEGIN("mitkRenderingDepthSortingTest")

  mitk::RenderingTestHelper renderingHelper(640, 480, argc, argv);

  renderingHelper.SetMapperIDToRender3D();

  mitk::DataNode *dataNode = renderingHelper.GetDataStorage()->GetNode(mitk::NodePredicateDataType::New("Surface"));

  if (dataNode)
  {
    dataNode->SetOpacity(0.8);
    dataNode->SetBoolProperty("Depth Sorting", true);
    dataNode->Update();
  }

  //### Usage of CompareRenderWindowAgainstReference: See docu of mitkRrenderingTestHelper
  MITK_TEST_CONDITION(renderingHelper.CompareRenderWindowAgainstReference(argc, argv) == true,
                      "CompareRenderWindowAgainstReference test result positive?");

  MITK_TEST_END();
}
