/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef IWORKBENCHPARTSITE_H_
#define IWORKBENCHPARTSITE_H_


#include "berryIWorkbenchSite.h"

namespace berry {

struct IWorkbenchPart;

/**
 * \ingroup org_blueberry_ui_qt
 *
 * The primary interface between a workbench part and the workbench.
 * <p>
 * This interface is not intended to be implemented or extended by clients.
 * </p>
 */
struct BERRY_UI_QT IWorkbenchPartSite : public IWorkbenchSite
{

  berryObjectMacro(berry::IWorkbenchPartSite, IWorkbenchSite);

  ~IWorkbenchPartSite() override;

  /**
   * Returns the part registry extension id for this workbench site's part.
   * <p>
   * The name comes from the <code>id</code> attribute in the configuration
   * element.
   * </p>
   *
   * @return the registry extension id
   */
  virtual QString GetId() const = 0;

  /**
   * Returns the part associated with this site
   *
   * @return the part associated with this site
   */
  virtual SmartPointer<IWorkbenchPart> GetPart() = 0;

  /**
   * Returns the unique identifier of the plug-in that defines this workbench
   * site's part.
   *
   * @return the unique identifier of the declaring plug-in
   */
  virtual QString GetPluginId() const = 0;

  /**
   * Returns the registered name for this workbench site's part.
   * <p>
   * The name comes from the <code>name</code> attribute in the configuration
   * element.
   * </p>
   *
   * @return the part name
   */
  virtual QString GetRegisteredName() const = 0;

};

} // namespace berry


#endif /*IWORKBENCHPARTSITE_H_*/
