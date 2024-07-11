/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYIFOLDERLAYOUT_H_
#define BERRYIFOLDERLAYOUT_H_

#include "berryIPlaceholderFolderLayout.h"

namespace berry {

/**
 * \ingroup org_blueberry_ui_qt
 *
 * An <code>IFolderLayout</code> is used to define the initial views within a folder.
 * The folder itself is contained within an <code>IPageLayout</code>.
 * <p>
 * This interface is not intended to be implemented by clients.
 * </p>
 *
 * @see IPageLayout#createFolder
 */
struct BERRY_UI_QT IFolderLayout : public IPlaceholderFolderLayout
{

  berryObjectMacro(berry::IFolderLayout);

  ~IFolderLayout() override;

    /**
     * Adds a view with the given compound id to this folder.
     * See the {@link IPageLayout} type documentation for more details about compound ids.
     * The primary id must name a view contributed to the workbench's view extension point
     * (named <code>"org.blueberry.ui.views"</code>).
     *
     * @param viewId the view id
     */
    virtual void AddView(const QString& viewId) = 0;
};

}

#endif /*BERRYIFOLDERLAYOUT_H_*/
