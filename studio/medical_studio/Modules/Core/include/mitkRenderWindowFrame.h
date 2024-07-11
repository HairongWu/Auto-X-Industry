/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkRenderWindowFrame_h
#define mitkRenderWindowFrame_h

#include <MitkCoreExports.h>
#include <itkObject.h>
#include <mitkCommon.h>
#include <vtkSmartPointer.h>

class vtkRenderer;
class vtkRenderWindow;

namespace mitk
{
  /**
   * This is a simple class for rendering colored rectangles
   * at the boarders of vtkRenderWindows.
   * The rectangle rendering itself is performed by means of a
   * vtkProp (vtkMitkRectangleProp).
   * This class instantiates the vtkProp and a corresponding vtkRenderer instance.
   */
  class MITKCORE_EXPORT RenderWindowFrame : public itk::Object
  {
  public:
    mitkClassMacroItkParent(RenderWindowFrame, itk::Object);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      /**
       * Sets the renderwindow, in which colored rectangle boarders will be shown.
       * Make sure, you have called this function
       * before calling Enable()
       */
      virtual void SetRenderWindow(vtkSmartPointer<vtkRenderWindow> renderWindow);

    /**
     * Enables drawing of the colored rectangle.
     * If you want to disable it, call the Disable() function.
     */
    virtual void Enable(float col1, float col2, float col3);

    /**
     * Disables drawing of the colored rectangle.
     * If you want to enable it, call the Enable() function.
     */
    virtual void Disable();

    /**
     * Checks, if the text is currently
     * enabled (visible)
     */
    virtual bool IsEnabled();

    /**
     * Returns the vtkRenderWindow, which is used
     * for displaying the text
     */
    virtual vtkSmartPointer<vtkRenderWindow> GetRenderWindow();

    /**
     * Returns the renderer responsible for
     * rendering the text into the
     * vtkRenderWindow
     */
    virtual vtkSmartPointer<vtkRenderer> GetVtkRenderer();

  protected:
    /**
     * Constructor
     */
    RenderWindowFrame();

    /**
     * Destructor
     */
    ~RenderWindowFrame() override;

    vtkSmartPointer<vtkRenderWindow> m_RenderWindow;
    vtkSmartPointer<vtkRenderer> m_RectangleRenderer;
    bool m_IsEnabled;
  };
} // end of namespace mitk
#endif
