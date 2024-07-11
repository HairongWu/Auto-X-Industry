/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkTestingMacros.h"
#include <mitkCommon.h>
#include <mitkItkLoggingAdapter.h>
#include <mitkVtkLoggingAdapter.h>

class ItkLoggingTestClass : public itk::Object
{
public:
  mitkClassMacroItkParent(ItkLoggingTestClass, itk::Object);
  itkFactorylessNewMacro(Self) itkCloneMacro(Self)

    void TestItkWarningMessage()
  {
    itkWarningMacro("Test ITK Warning message");
  }
};

/** @brief This test tests all logging adapters of MITK. */
class LoggingAdapterTestClass
{
public:
  static void TestVtkLoggingWithoutAdapter()
  {
    MITK_TEST_OUTPUT(
      << "Testing vtk logging without adapter class: a separate window should open and display the logging messages.")
    vtkOutputWindow::GetInstance()->DisplayText("Test VTK InfoMessage");
    vtkOutputWindow::GetInstance()->DisplayDebugText("Test Vtk Debug Message");
    vtkOutputWindow::GetInstance()->DisplayGenericWarningText("Test Vtk Generic Warning Message");
    vtkOutputWindow::GetInstance()->DisplayWarningText("Test Vtk Warning Message");
    vtkOutputWindow::GetInstance()->DisplayErrorText("Test Vtk Error Message");
    MITK_TEST_CONDITION_REQUIRED(true, "Testing if Vtk logging without adapter runs without errors.");
  }
  static void TestVtkLoggingWithAdapter()
  {
    MITK_TEST_OUTPUT(
      << "Testing vtk logging with adapter class: Vtk logging messages should be logged as MITK logging messages.")
    mitk::VtkLoggingAdapter::Initialize();
    vtkOutputWindow::GetInstance()->DisplayText("Test Vtk Info Message");
    vtkOutputWindow::GetInstance()->DisplayDebugText("Test Vtk Debug Message");
    vtkOutputWindow::GetInstance()->DisplayGenericWarningText("Test Vtk Generic Warning Message");
    vtkOutputWindow::GetInstance()->DisplayWarningText("Test Vtk Warning Message");
    vtkOutputWindow::GetInstance()->DisplayErrorText("Test Vtk Error Message");
    MITK_TEST_CONDITION_REQUIRED(true, "Testing if Vtk logging with MITK logging adapter runs without errors.");
  }

  static void TestItkLoggingWithoutAdapter()
  {
    ItkLoggingTestClass::Pointer myItkLogger = ItkLoggingTestClass::New();
    myItkLogger->TestItkWarningMessage();
  }

  static void TestItkLoggingWithAdapter()
  {
    mitk::ItkLoggingAdapter::Initialize();
    ItkLoggingTestClass::Pointer myItkLogger = ItkLoggingTestClass::New();
    myItkLogger->TestItkWarningMessage();
  }
};

int mitkLoggingAdapterTest(int /*argc*/, char * /*argv*/ [])
{
  MITK_TEST_BEGIN("LoggingAdapters: VTK, ITK");
  LoggingAdapterTestClass::TestVtkLoggingWithoutAdapter();
  LoggingAdapterTestClass::TestVtkLoggingWithAdapter();
  LoggingAdapterTestClass::TestItkLoggingWithoutAdapter();
  LoggingAdapterTestClass::TestItkLoggingWithAdapter();
  MITK_TEST_END();
}
