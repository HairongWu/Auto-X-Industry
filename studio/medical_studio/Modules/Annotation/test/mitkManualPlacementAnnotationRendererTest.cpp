/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK
#include <mitkIOUtil.h>

#include <mitkManualPlacementAnnotationRenderer.h>
#include <mitkRenderingTestHelper.h>
#include <mitkTestFixture.h>
#include <mitkTestingMacros.h>
#include <mitkTextAnnotation2D.h>
#include <mitkTextAnnotation3D.h>

class mitkManualPlacementAnnotationRendererTestSuite : public mitk::TestFixture
{
  CPPUNIT_TEST_SUITE(mitkManualPlacementAnnotationRendererTestSuite);
  MITK_TEST(Render2DAnnotation);
  MITK_TEST(Render3DAnnotation);

  CPPUNIT_TEST_SUITE_END();

private:
  /** Members used inside the different test methods. All members are initialized via setUp().*/
  mitk::RenderingTestHelper m_RenderingTestHelper;
  std::vector<std::string> m_CommandlineArgs;
  std::string m_PathToBall;
  std::string m_PathToImage;
  std::string m_ReferenceImagePath;

public:
  /**
   * @brief mitkManualPlacementAnnotationRendererTestSuite Because the RenderingTestHelper does not have an
   * empty default constructor, we need this constructor to initialize the helper with a
   * resolution.
   */
  mitkManualPlacementAnnotationRendererTestSuite() : m_RenderingTestHelper(300, 300) {}
  /**
   * @brief Setup Initialize a fresh rendering test helper and a vector of strings
   * to simulate commandline arguments for vtkTesting::Test.
   */
  void setUp()
  {
    m_RenderingTestHelper = mitk::RenderingTestHelper(300, 300);

    m_PathToBall = GetTestDataFilePath("ball.stl");
    m_PathToImage = GetTestDataFilePath("Pic3D.nrrd");
    m_ReferenceImagePath = "RenderingTestData/ReferenceScreenshots/Annotation/";

    // Build a command line for the vtkTesting::Test method.
    // See VTK documentation and RenderingTestHelper for more information.
    // Use the following command line option to save the difference image
    // and the test image in some tmp folder
    // m_CommandlineArgs.push_back("-T");
    // m_CommandlineArgs.push_back("/path/to/save/tmp/difference/images/");
    m_CommandlineArgs.push_back("-V");
  }

  void tearDown() {}
  void Render2DAnnotation()
  {
    mitk::DataNode::Pointer ballnode = mitk::DataNode::New();
    ballnode->SetData(mitk::IOUtil::Load(m_PathToBall)[0]);
    m_RenderingTestHelper.AddNodeToStorage(ballnode);

    mitk::DataNode::Pointer imagenode = mitk::DataNode::New();
    imagenode->SetData(mitk::IOUtil::Load(m_PathToImage)[0]);
    m_RenderingTestHelper.AddNodeToStorage(imagenode);

    std::string refImagePath = GetTestDataFilePath(m_ReferenceImagePath + "ManualPlacementAnnotationRenderer2DAnnotation.png");

    // reference screenshot for this test
    m_CommandlineArgs.push_back(refImagePath);
    // Convert vector of strings to argc/argv
    mitk::RenderingTestHelper::ArgcHelperClass arg(m_CommandlineArgs);
    m_RenderingTestHelper.SetViewDirection(mitk::AnatomicalPlane::Sagittal);

    mitk::TextAnnotation2D::Pointer textAnnotation2D = mitk::TextAnnotation2D::New();

    textAnnotation2D->SetText("TEST ManualPlacementAnnotationRenderer2DAnnotation");
    mitk::Point2D pos;
    pos[0] = 0;
    pos[1] = 0;
    textAnnotation2D->SetPosition2D(pos);

    mitk::BaseRenderer *renderer = mitk::BaseRenderer::GetInstance(m_RenderingTestHelper.GetVtkRenderWindow());
    mitk::ManualPlacementAnnotationRenderer::AddAnnotation(textAnnotation2D.GetPointer(), renderer);

    m_RenderingTestHelper.Render();
//    m_RenderingTestHelper.SaveReferenceScreenShot(refImagePath);
    m_RenderingTestHelper.SetAutomaticallyCloseRenderWindow(true);
    CPPUNIT_ASSERT(m_RenderingTestHelper.CompareRenderWindowAgainstReference(arg.GetArgc(), arg.GetArgv()) == true);
  }

  void Render3DAnnotation()
  {
    mitk::DataNode::Pointer ballnode = mitk::DataNode::New();
    ballnode->SetData(mitk::IOUtil::Load(m_PathToBall)[0]);
    m_RenderingTestHelper.AddNodeToStorage(ballnode);

    mitk::DataNode::Pointer imagenode = mitk::DataNode::New();
    imagenode->SetData(mitk::IOUtil::Load(m_PathToImage)[0]);
    m_RenderingTestHelper.AddNodeToStorage(imagenode);

    std::string refImagePath = GetTestDataFilePath(m_ReferenceImagePath + "ManualPlacementAnnotationRenderer3DAnnotation.png");
    // reference screenshot for this test
    m_CommandlineArgs.push_back(refImagePath);
    // Convert vector of strings to argc/argv
    mitk::RenderingTestHelper::ArgcHelperClass arg(m_CommandlineArgs);
    m_RenderingTestHelper.SetMapperIDToRender3D();

    mitk::TextAnnotation3D::Pointer textAnnotation3D = mitk::TextAnnotation3D::New();

    textAnnotation3D->SetText("TEST ManualPlacementAnnotationRenderer3DAnnotation");
    mitk::Point3D pos;
    pos[0] = 10;
    pos[1] = 10;
    pos[2] = 10;
    textAnnotation3D->SetPosition3D(pos);

    mitk::BaseRenderer *renderer = mitk::BaseRenderer::GetInstance(m_RenderingTestHelper.GetVtkRenderWindow());
    mitk::ManualPlacementAnnotationRenderer::AddAnnotation(textAnnotation3D.GetPointer(), renderer);

    m_RenderingTestHelper.Render();
//    m_RenderingTestHelper.SaveReferenceScreenShot(refImagePath);
    m_RenderingTestHelper.SetAutomaticallyCloseRenderWindow(true);
    CPPUNIT_ASSERT(m_RenderingTestHelper.CompareRenderWindowAgainstReference(arg.GetArgc(), arg.GetArgv()) == true);
  }
};
MITK_TEST_SUITE_REGISTRATION(mitkManualPlacementAnnotationRenderer)
