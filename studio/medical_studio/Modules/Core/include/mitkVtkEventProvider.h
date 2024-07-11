/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVtkEventProvider_h
#define mitkVtkEventProvider_h

#include "mitkRenderWindow.h"
#include <MitkCoreExports.h>

#include "vtkInteractorObserver.h"

namespace mitk
{
  /**
  * \brief Integrates into the VTK event mechanism to generate MITK specific events.
  * This class is NON-QT dependent pandon to the current MITK event handling code in QmitkRenderWindow.
  * \ingroup Renderer
  */
  class MITKCORE_EXPORT vtkEventProvider : public vtkInteractorObserver
  {
  public:
    static vtkEventProvider *New();
    vtkTypeMacro(vtkEventProvider, vtkInteractorObserver);

    // Satisfy the superclass API. Enable/disable listening for events.
    void SetEnabled(int) override;
    void SetInteractor(vtkRenderWindowInteractor *iren) override;

    // Interface to MITK
    virtual void SetMitkRenderWindow(mitk::RenderWindow *renWin);
    mitk::RenderWindow *GetRenderWindow();

  protected:
    vtkEventProvider();
    ~vtkEventProvider() override;

    // methods for processing events - callback for the observer/command pattern of vtkCommand
    static void ProcessEvents(vtkObject *object, unsigned long event, void *clientdata, void *calldata);

    mitk::RenderWindow *m_RenderWindow;

    // adds the MITK interaction event types to the VTK observer/command pattern
    void AddInteractionEvent(unsigned long ievent);
    // removes the MITK interaction event types
    void RemoveInteractionEvent(unsigned long ievent);
    typedef std::vector<unsigned long> InteractionEventsVectorType;
    InteractionEventsVectorType m_InteractionEventsVector;

  private:
    vtkEventProvider(const vtkEventProvider &); // Not implemented.
    void operator=(const vtkEventProvider &);   // Not implemented.
  };
}
#endif
