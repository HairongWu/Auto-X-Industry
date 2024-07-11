/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef org_mitk_example_gui_selectionservicemitk_Activator_H
#define org_mitk_example_gui_selectionservicemitk_Activator_H

#include <ctkPluginActivator.h>

class org_mitk_example_gui_selectionservicemitk_Activator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "org_mitk_example_gui_selectionservicemitk")
  Q_INTERFACES(ctkPluginActivator)

public:
  void start(ctkPluginContext *context) override;
  void stop(ctkPluginContext *context) override;
};

#endif // org_mitk_example_gui_selectionservicemitk_Activator_H
