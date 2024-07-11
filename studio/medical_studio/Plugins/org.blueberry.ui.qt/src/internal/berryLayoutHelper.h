/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYLAYOUTHELPER_H_
#define BERRYLAYOUTHELPER_H_

#include "berryViewFactory.h"
#include "berryPartPane.h"

namespace berry {

class PageLayout;

/**
 * Helper methods that the internal layout classes (<code>PageLayout</code> and
 * <code>FolderLayout</code>) utilize for activities support and view creation.
 *
 * @since 3.0
 */
class LayoutHelper {


private:

  /**
   * Not intended to be instantiated.
   */
  LayoutHelper();

public:

    /**
     * Creates a series of listeners that will activate the provided view on the
     * provided page layout when <code>IIdentifier</code> enablement changes. The
     * rules for this activation are as follows: <p>
     * <ul>
     * <li> if the identifier becomes enabled and the perspective of the page
     * layout is the currently active perspective in its window, then activate
     * the views immediately.
     * <li> if the identifier becomes enabled and the perspective of the page
     * layout is not the currently active perspective in its window, then add an
     * <code>IPerspectiveListener</code> to the window and activate the views
     * when the perspective becomes active.
     *
     * @param pageLayout <code>PageLayout</code>.
     * @param viewId the view id to activate upon <code>IIdentifier</code> enablement.
     */
    static void AddViewActivator(SmartPointer<PageLayout> pageLayout,
            const QString& viewId);

    /**
     * Create the view.  If it's already been been created in the provided
     * factory, return the shared instance.
     *
     * @param factory the <code>ViewFactory</code> to use.
     * @param viewID the view id to use.
     * @return the new <code>ViewPane</code>.
     * @throws PartInitException thrown if there is a problem creating the view.
     */
    static PartPane::Pointer CreateView(ViewFactory* factory, const QString& viewId);

};

}

#endif /* BERRYLAYOUTHELPER_H_ */
