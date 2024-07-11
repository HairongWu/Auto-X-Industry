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

#include <mitkNodePredicateDataType.h>

#include <mitkPlaneGeometryData.h>

// VTK
#include <vtkRegressionTestImage.h>

mitk::DataNode::Pointer addPlaneToDataStorage(mitk::RenderingTestHelper &renderingHelper,
                                              mitk::Image *image,
                                              mitk::AnatomicalPlane orientation,
                                              mitk::ScalarType zPos)
{
  auto geometry = mitk::PlaneGeometry::New();
  geometry->InitializeStandardPlane(image->GetGeometry(), orientation, zPos);

  auto geometryData = mitk::PlaneGeometryData::New();
  geometryData->SetPlaneGeometry(geometry);

  auto node = mitk::DataNode::New();
  node->SetData(geometryData);

  renderingHelper.AddNodeToStorage(node);

  return node;
}

int mitkPlaneGeometryDataMapper2DTest(int argc, char *argv[])
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
  MITK_TEST_BEGIN("mitkPlaneGeometryDataMapper2DTest")

  mitk::RenderingTestHelper renderingHelper(640, 480, argc, argv);
  auto image = static_cast<mitk::Image *>(
    renderingHelper.GetDataStorage()->GetNode(mitk::TNodePredicateDataType<mitk::Image>::New())->GetData());

  auto zCoord = image->GetGeometry()->GetBoundingBox()->GetCenter()[0];
  addPlaneToDataStorage(renderingHelper, image, mitk::AnatomicalPlane::Sagittal, zCoord);
  addPlaneToDataStorage(renderingHelper, image, mitk::AnatomicalPlane::Coronal, zCoord);

  auto planeNode = addPlaneToDataStorage(renderingHelper, image, mitk::AnatomicalPlane::Sagittal, zCoord);
  auto planeGeometry = static_cast<mitk::PlaneGeometryData *>(planeNode->GetData())->GetPlaneGeometry();

  auto transform = mitk::AffineTransform3D::New();
  mitk::Vector3D rotationAxis;
  rotationAxis.Fill(0.0);
  rotationAxis[2] = 1;
  transform->Rotate3D(rotationAxis, vnl_math::pi_over_4);

  planeGeometry->Compose(transform);

  auto bounds = planeGeometry->GetBounds();
  bounds[1] /= 3;
  planeGeometry->SetBounds(bounds);

  planeGeometry->SetReferenceGeometry(nullptr);

  planeNode->SetIntProperty("Crosshair.Gap Size", 4);

  //### Usage of CompareRenderWindowAgainstReference: See docu of mitkRrenderingTestHelper
  MITK_TEST_CONDITION(renderingHelper.CompareRenderWindowAgainstReference(argc, argv, 1) == true,
                      "CompareRenderWindowAgainstReference test result positive?");

  // use this to generate a reference screenshot or save the file:
  if (false)
  {
    renderingHelper.SaveReferenceScreenShot("output.png");
  }

  MITK_TEST_END();
}
