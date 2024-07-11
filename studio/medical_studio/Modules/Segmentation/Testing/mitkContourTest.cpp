/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkCommon.h"
#include "mitkContour.h"
#include "mitkTestingMacros.h"

#include <fstream>

int mitkContourTest(int /*argc*/, char * /*argv*/ [])
{
  MITK_TEST_BEGIN("Contour");
  mitk::Contour::Pointer contour = mitk::Contour::New();
  MITK_TEST_CONDITION_REQUIRED(contour.IsNotNull(), "Testing initialization!");

  mitk::Point3D p;
  p.Fill(0);
  contour->AddVertex(p);
  p.Fill(1);
  contour->AddVertex(p);
  p.Fill(2);
  contour->AddVertex(p);
  MITK_TEST_CONDITION_REQUIRED(contour->GetNumberOfPoints() == 3, "Testing AddVertex()");

  mitk::Contour::PointsContainerPointer points = contour->GetPoints();
  MITK_TEST_CONDITION_REQUIRED(points.IsNotNull(), "Testing GetPoints()");

  contour->Initialize();
  MITK_TEST_CONDITION_REQUIRED(contour->GetNumberOfPoints() == 0, "Testing Initialize()!");

  contour->SetPoints(points);
  MITK_TEST_CONDITION_REQUIRED(contour->GetNumberOfPoints() == 3, "Testimg SetPoints()!");

  mitk::Contour::PathPointer path = contour->GetContourPath();
  MITK_TEST_CONDITION_REQUIRED(path.IsNotNull(), "Testing GetContourPath()!");

  contour->UpdateOutputInformation();
  contour->SetClosed(false);
  MITK_TEST_CONDITION_REQUIRED(!contour->GetClosed(), "Testing GetClosed()!");

  mitk::Contour::Pointer cloneContour = contour->Clone();
  MITK_TEST_CONDITION_REQUIRED(cloneContour.IsNotNull(), "Testing clone instantiation!");
  MITK_TEST_CONDITION_REQUIRED(cloneContour.GetPointer() != contour.GetPointer(),
                               "Testing cloned object is not original object!");
  MITK_TEST_CONDITION_REQUIRED(cloneContour->GetGeometry()->GetCenter() == contour->GetGeometry()->GetCenter(),
                               "Testing if Geometry is cloned!");
  MITK_TEST_CONDITION_REQUIRED(cloneContour->GetPoints() == points, "Testing cloning of point data!");

  MITK_TEST_END();
}
