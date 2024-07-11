/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef BERRYIPREFERENCEPAGE_H_
#define BERRYIPREFERENCEPAGE_H_

#include "berryObject.h"
#include "berryIWorkbench.h"

#include <QObject>

namespace berry
{

/**
 * \ingroup org_blueberry_ui_qt
 *
 * Interface for workbench preference pages.
 * <p>
 * Clients should implement this interface and include the name of their class
 * in an extension contributed to the workbench's preference extension point
 * (named <code>"org.blueberry.ui.preferencePages"</code>).
 * For example, the plug-in's XML markup might contain:
 * \code{.unparsed}
 * <extension point="org.blueberry.ui.preferencePages>
 *   <page id="com.example.myplugin.prefs"
 *         name="Knobs"
 *         class="ns::MyPreferencePage" />
 * </extension>
 * \endcode
 * </p>
 */
struct BERRY_UI_QT IPreferencePage: virtual public Object
{

  berryObjectMacro(berry::IPreferencePage);

  ~IPreferencePage() override;

  /**
     * Initializes this preference page for the given workbench.
     * <p>
     * This method is called automatically as the preference page is being created
     * and initialized. Clients must not call this method.
     * </p>
     *
     * @param workbench the workbench
     */
   virtual void Init(IWorkbench::Pointer workbench) = 0;

  /**
   * Creates the top level control for this preference
   * page under the given parent widget.
   * <p>
   * Implementors are responsible for ensuring that
   * the created control can be accessed via <code>GetControl</code>
   * </p>
   *
   * @param parent the parent widget
   */
  virtual void CreateControl(void* parent) = 0;

  /**
   * Returns the top level control for this dialog page.
   * <p>
   * May return <code>null</code> if the control
   * has not been created yet.
   * </p>
   *
   * @return the top level control or <code>null</code>
   */
  virtual void* GetControl() const = 0;

  ///
  /// Invoked when the OK button was clicked in the preferences dialog
  ///
  virtual bool PerformOk() = 0;

  ///
  /// Invoked when the Cancel button was clicked in the preferences dialog
  ///
  virtual void PerformCancel() = 0;

  ///
  /// Invoked when the user performed an import. As the values of the preferences may have changed
  /// you should read all values again from the preferences service.
  ///
  virtual void Update() = 0;
};

}

Q_DECLARE_INTERFACE(berry::IPreferencePage, "org.blueberry.ui.IPreferencePage")

#endif /*BERRYIPREFERENCEPAGE_H_*/
