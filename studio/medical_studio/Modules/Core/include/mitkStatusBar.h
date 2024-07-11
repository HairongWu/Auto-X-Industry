/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkStatusBar_h
#define mitkStatusBar_h
#include "mitkStatusBarImplementation.h"
#include <MitkCoreExports.h>
#include <itkObject.h>
#include <mitkPoint.h>
#include <mitkTimeGeometry.h>
#include <itkIndex.h>

namespace mitk
{
  //##Documentation
  //## @brief Sending a message to the applications StatusBar
  //##
  //## Holds a GUI dependent StatusBarImplementation and sends the text further.
  //## nearly equal to itk::OutputWindow,
  //## no Window, but one line of text and a delay for clear.
  //## all mitk-classes use this class to display text on GUI-StatusBar.
  //## The mainapplication has to set the internal held StatusBarImplementation with SetInstance(..).
  //## @ingroup Interaction
  class MITKCORE_EXPORT StatusBar : public itk::Object
  {
  public:
    itkTypeMacro(StatusBar, itk::Object);

    //##Documentation
    //## @brief static method to get the GUI dependent StatusBar-instance
    //## so the methods DisplayText, etc. can be called
    //## No reference counting, cause of decentral static use!
    static StatusBar *GetInstance();

    //##Documentation
    //## @brief Supply a GUI- dependent StatusBar. Has to be set by the application
    //## to connect the application dependent subclass of mitkStatusBar
    //## if you create an instance, then call ->Delete() on the supplied
    //## instance after setting it.
    static void SetImplementation(StatusBarImplementation *instance);

    //##Documentation
    //## @brief Send a string to the applications StatusBar
    void DisplayText(const char *t);
    //##Documentation
    //## @brief Send a string with a time delay to the applications StatusBar
    void DisplayText(const char *t, int ms);
    void DisplayErrorText(const char *t);
    void DisplayWarningText(const char *t);
    void DisplayWarningText(const char *t, int ms);
    void DisplayGenericOutputText(const char *t);
    void DisplayDebugText(const char *t);
    void DisplayGreyValueText(const char *t);

    //##Documentation
    void DisplayRendererInfo(Point3D point, TimePointType time);
    //## @brief Display position, index, time and pixel value
    void DisplayImageInfo(Point3D point, itk::Index<3> index, TimePointType time, ScalarType pixelValue);
    //## @brief Display rotation, index, time and custom pixel value
    void DisplayImageInfo(Point3D point, itk::Index<3> index, TimePointType time, const char* pixelValue);
    //## @brief Display placeholder text for invalid information
    void DisplayImageInfoInvalid();

    //##Documentation
    //## @brief removes any temporary message being shown.
    void Clear();

    //##Documentation
    //## @brief Set the SizeGrip of the window
    //## (the triangle in the lower right Windowcorner for changing the size)
    //## to enabled or disabled
    void SetSizeGripEnabled(bool enable);

  protected:
    StatusBar();
    ~StatusBar() override;

    static StatusBarImplementation *m_Implementation;
    static StatusBar *m_Instance;
  };

} // end namespace mitk
#endif
