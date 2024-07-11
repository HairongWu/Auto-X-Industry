/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVtkWidgetRendering_h
#define mitkVtkWidgetRendering_h

#include <mitkBaseData.h>

class vtkRenderer;
class vtkRenderWindow;
class vtkInteractorObserver;

namespace mitk
{
  class RenderWindow;

  /**
   * \brief Mechanism for rendering a vtkWidget in the foreground of a RenderWindow.
   *
   * To use this class, specify the vtkRenderWindow of the window into which the
   * vtkWidget shall be placed, and set the vtkWidget using SetVtkWidget().
   * After enabling the vtkWidget and calling Enable() of this class, the widget
   * should be rendered.
   *
   * Note: this class only provides a basic mechanism for adding widget; all widget
   * configuration such as placement, size, and en-/disabling of interaction
   * mechanisms need to be done in the vtkWidget object.
   */
  class MITKCORE_EXPORT VtkWidgetRendering : public BaseData
  {
  public:
    mitkClassMacro(VtkWidgetRendering, BaseData);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      /**
       * Sets the renderwindow, in which the widget
       * will be shown. Make sure, you have called this function
       * before calling Enable()
       */
      virtual void SetRenderWindow(vtkRenderWindow *renderWindow);

    /**
     * Enables drawing of the widget.
     * If you want to disable it, call the Disable() function.
     */
    virtual void Enable();

    /**
     * Disables drawing of the widget.
     * If you want to enable it, call the Enable() function.
     */
    virtual void Disable();

    /**
     * Checks, if the widget is currently
     * enabled (visible)
     */
    virtual bool IsEnabled();

    /**
     * Empty implementation, since the VtkWidgetRendering doesn't
     * support the requested region concept
     */
    void SetRequestedRegionToLargestPossibleRegion() override;

    /**
     * Empty implementation, since the VtkWidgetRendering doesn't
     * support the requested region concept
     */
    bool RequestedRegionIsOutsideOfTheBufferedRegion() override;

    /**
     * Empty implementation, since the VtkWidgetRendering doesn't
     * support the requested region concept
     */
    bool VerifyRequestedRegion() override;

    /**
     * Empty implementation, since the VtkWidgetRendering doesn't
     * support the requested region concept
     */
    void SetRequestedRegion(const itk::DataObject *) override;

    /**
     * Returns the vtkRenderWindow, which is used
     * for displaying the widget
     */
    virtual vtkRenderWindow *GetRenderWindow();

    /**
     * Returns the renderer responsible for
     * rendering the widget into the
     * vtkRenderWindow
     */
    virtual vtkRenderer *GetVtkRenderer();

    /** Set the vtkWidget to be rendered */
    void SetVtkWidget(vtkInteractorObserver *widget);

    /** Get the vtkWidget to be rendered */
    vtkInteractorObserver *GetVtkWidget() const;

  protected:
    /**
     * Constructor
     */
    VtkWidgetRendering();

    /**
     * Destructor
     */
    ~VtkWidgetRendering() override;

    vtkRenderWindow *m_RenderWindow;
    vtkRenderer *m_Renderer;

    vtkInteractorObserver *m_VtkWidget;

    bool m_IsEnabled;
  };

} // end of namespace mitk
#endif
