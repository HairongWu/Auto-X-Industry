/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYIWORKBENCHLISTENER_H_
#define BERRYIWORKBENCHLISTENER_H_

#include <org_blueberry_ui_qt_Export.h>

#include "berryMessage.h"

namespace berry
{

struct IWorkbench;

/**
 * Interface for listening to workbench lifecycle events.
 * <p>
 * This interface may be implemented by clients.
 * </p>
 *
 * @see IWorkbench#addWorkbenchListener
 * @see IWorkbench#removeWorkbenchListener
 */
struct BERRY_UI_QT IWorkbenchListener
{

  struct Events {
      typedef Message2<IWorkbench*, bool, bool> PreShutdownEvent;
      typedef Message1<IWorkbench*> PostShutdownEvent;

      PreShutdownEvent preShutdown;
      PostShutdownEvent postShutdown;

      void AddListener(IWorkbenchListener* listener);
      void RemoveListener(IWorkbenchListener* listener);

  private:

    typedef MessageDelegate2<IWorkbenchListener, IWorkbench*, bool, bool> Delegate2;
    typedef MessageDelegate1<IWorkbenchListener, IWorkbench*> Delegate1;
  };

  virtual ~IWorkbenchListener();

  /**
   * Notifies that the workbench is about to shut down.
   * <p>
   * This method is called immediately prior to workbench shutdown before any
   * windows have been closed.
   * </p>
   * <p>
   * The listener may veto a regular shutdown by returning <code>false</code>,
   * although this will be ignored if the workbench is being forced to shut down.
   * </p>
   * <p>
   * Since other workbench listeners may veto the shutdown, the listener should
   * not dispose resources or do any other work during this notification that would
   * leave the workbench in an inconsistent state.
   * </p>
   *
   * @return <code>true</code> to allow the workbench to proceed with shutdown,
   *   <code>false</code> to veto a non-forced shutdown
   */
  virtual bool PreShutdown(IWorkbench*  /*workbench*/, bool  /*forced*/) { return true; }

  /**
   * Performs arbitrary finalization after the workbench stops running.
   * <p>
   * This method is called during workbench shutdown after all windows
   * have been closed.
   * </p>
   */
  virtual void PostShutdown(IWorkbench*  /*workbench*/) {}

};

}

#endif /* BERRYIWORKBENCHLISTENER_H_ */
