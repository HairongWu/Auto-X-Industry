/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkStatusBar.h"
#include <itkObjectFactory.h>
#include <itkOutputWindow.h>

namespace mitk
{
  StatusBarImplementation *StatusBar::m_Implementation = nullptr;
  StatusBar *StatusBar::m_Instance = nullptr;

  /**
   * Display the text in the statusbar of the application
   */
  void StatusBar::DisplayText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayText(t);
  }

  /**
   * Display the text in the statusbar of the application for ms seconds
   */
  void StatusBar::DisplayText(const char *t, int ms)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayText(t, ms);
  }

  void StatusBar::DisplayErrorText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayErrorText(t);
  }
  void StatusBar::DisplayWarningText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayWarningText(t);
  }
  void StatusBar::DisplayWarningText(const char *t, int ms)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayWarningText(t, ms);
  }
  void StatusBar::DisplayGenericOutputText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayGenericOutputText(t);
  }
  void StatusBar::DisplayDebugText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayDebugText(t);
  }
  void StatusBar::DisplayGreyValueText(const char *t)
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayGreyValueText(t);
  }

  static void WriteCommonRendererInfo(std::ostringstream& stream, Point3D point, TimePointType time)
  {
    stream << "Position: <" << std::fixed << point[0] << ", "
      << std::fixed << point[1] << ", "
      << std::fixed << point[2] << "> mm; ";

    stream << "Time: " << time << " ms";
  }

  static void WriteCommonImageInfo(std::ostringstream& stream, Point3D point, itk::Index<3> index, TimePointType time)
  {
    stream << "Position: <" << std::fixed << point[0] << ", "
                            << std::fixed << point[1] << ", "
                            << std::fixed << point[2] << "> mm; ";

    stream << "Index: <" << index[0] << ", "
                         << index[1] << ", "
                         << index[2] << "> ; ";

    stream << "Time: " << time << " ms";
  }

  void StatusBar::DisplayRendererInfo(Point3D point, TimePointType time)
  {
    if (m_Implementation == nullptr)
      return;

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.precision(2);

    WriteCommonRendererInfo(stream, point, time);

    m_Implementation->DisplayGreyValueText(stream.str().c_str());
  }

  void StatusBar::DisplayImageInfo(Point3D point, itk::Index<3> index, TimePointType time, ScalarType pixelValue)
  {
    if (m_Implementation == nullptr)
      return;

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.precision(2);

    WriteCommonImageInfo(stream, point, index, time);
    stream << "; Pixel value: ";

    if (fabs(pixelValue) > 1000000 || fabs(pixelValue) < 0.01)
      stream << std::scientific;

    stream << pixelValue;

    m_Implementation->DisplayGreyValueText(stream.str().c_str());
  }

  void StatusBar::DisplayImageInfo(Point3D point, itk::Index<3> index, TimePointType time, const char* pixelValue)
  {
    if (m_Implementation == nullptr)
      return;

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.precision(2);

    WriteCommonImageInfo(stream, point, index, time);
    stream << "; " << pixelValue;

    m_Implementation->DisplayGreyValueText(stream.str().c_str());
  }

  void StatusBar::DisplayImageInfoInvalid()
  {
    if (m_Implementation != nullptr)
      m_Implementation->DisplayGreyValueText("No image information at this position!");
  }

  void StatusBar::Clear()
  {
    if (m_Implementation != nullptr)
      m_Implementation->Clear();
  }

  void StatusBar::SetSizeGripEnabled(bool enable)
  {
    if (m_Implementation != nullptr)
    {
      m_Implementation->SetSizeGripEnabled(enable);
    }
  }

  /**
   * Get the instance of this StatusBar
   */
  StatusBar *StatusBar::GetInstance()
  {
    if (m_Instance == nullptr) // if not set, then send a errormessage on OutputWindow
    {
      m_Instance = new StatusBar();
    }

    return m_Instance;
  }

  /**
   * Set an instance of this; application must do this!See Header!
   */
  void StatusBar::SetImplementation(StatusBarImplementation *implementation)
  {
    if (m_Implementation == implementation)
    {
      return;
    }
    m_Implementation = implementation;
  }

  StatusBar::StatusBar() {}
  StatusBar::~StatusBar() {}
} // end namespace mitk
