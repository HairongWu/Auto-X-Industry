/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYIPAGESERVICE_H_
#define BERRYIPAGESERVICE_H_

#include <org_blueberry_ui_qt_Export.h>

#include  "berryIPerspectiveListener.h"

namespace berry {

struct IWorkbenchPage;

/**
 * A page service tracks the page and perspective lifecycle events
 * within a workbench window.
 * <p>
 * This service can be acquired from your service locator:
 * <pre>
 *  IPageService service = (IPageService) getSite().getService(IPageService.class);
 * </pre>
 * <ul>
 * <li>This service is not available globally, only from the workbench window level down.</li>
 * </ul>
 * </p>
 *
 * @see IWorkbenchWindow
 * @see IPageListener
 * @see IPerspectiveListener
 * @see IServiceLocator#GetService
 * @note This interface is not intended to be implemented by clients.
 */
struct BERRY_UI_QT IPageService {

    virtual ~IPageService();

    /**
     * Adds the given listener for a page's perspective lifecycle events.
     * Has no effect if an identical listener is already registered.
   * <p>
   * <b>Note:</b> listeners should be removed when no longer necessary. If
   * not, they will be removed when the IServiceLocator used to acquire this
   * service is disposed.
   * </p>
     *
     * @param listener a perspective listener
     * @see #RemovePerspectiveListener
     */
    virtual void AddPerspectiveListener(IPerspectiveListener* listener) = 0;

    /**
     * Returns the active page.
     *
     * @return the active page, or <code>null</code> if no page is currently active
     */
    virtual SmartPointer<IWorkbenchPage> GetActivePage() const = 0;

    /**
     * Removes the given page's perspective listener.
     * Has no affect if an identical listener is not registered.
     *
     * @param listener a perspective listener
     */
    virtual void RemovePerspectiveListener(IPerspectiveListener* listener) = 0;

    virtual IPerspectiveListener::Events& GetPerspectiveEvents() = 0;
};

}

Q_DECLARE_INTERFACE(berry::IPageService, "org.blueberry.ui.IPageService")

#endif /* BERRYIPAGESERVICE_H_ */
