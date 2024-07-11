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
#include <mitkEnumerationProperty.h>
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>

// VTK
#include <vtkRegressionTestImage.h>

int mitkPointSetVtkMapper2DTransformedPointsTest(int argc, char *argv[])
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
  MITK_TEST_BEGIN("mitkPointSetVtkMapper2DTransformedPointsTest")

  mitk::RenderingTestHelper renderingHelper(640, 480, argc, argv);

  renderingHelper.SetViewDirection(mitk::AnatomicalPlane::Sagittal);

  mitk::DataNode *dataNode = renderingHelper.GetDataStorage()->GetNode(mitk::NodePredicateDataType::New("PointSet"));

  if (dataNode)
  {
    mitk::PointSet::Pointer pointSet = dynamic_cast<mitk::PointSet *>(dataNode->GetData());

    if (pointSet)
    {
      mitk::Point3D origin = pointSet->GetGeometry()->GetOrigin();

      origin[1] += 10;
      origin[2] += 15;

      pointSet->GetGeometry()->SetOrigin(origin);
      pointSet->Modified();
      dataNode->Update();
    }
  }

  //### Usage of CompareRenderWindowAgainstReference: See docu of mitkRenderingTestHelper
  MITK_TEST_CONDITION(renderingHelper.CompareRenderWindowAgainstReference(argc, argv) == true,
                      "CompareRenderWindowAgainstReference test result positive?");

  // use this to generate a reference screenshot or save the file:
  if (false)
  {
    renderingHelper.SaveReferenceScreenShot("D:/test/output.png");
  }

  MITK_TEST_END();
}
