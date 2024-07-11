/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkMitkWorkbenchIntroPlugin.h"
#include "QmitkMitkWorkbenchIntroPart.h"
#include "QmitkMitkWorkbenchIntroPreferencePage.h"

#include <mitkVersion.h>
#include <mitkLog.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <QFileInfo>
#include <QDateTime>

#include <usModuleInitialization.h>

US_INITIALIZE_MODULE

QmitkMitkWorkbenchIntroPlugin* QmitkMitkWorkbenchIntroPlugin::inst = nullptr;

QmitkMitkWorkbenchIntroPlugin::QmitkMitkWorkbenchIntroPlugin()
{
  inst = this;
}

QmitkMitkWorkbenchIntroPlugin::~QmitkMitkWorkbenchIntroPlugin()
{
}

QmitkMitkWorkbenchIntroPlugin* QmitkMitkWorkbenchIntroPlugin::GetDefault()
{
  return inst;
}

void QmitkMitkWorkbenchIntroPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);

  this->context = context;

  BERRY_REGISTER_EXTENSION_CLASS(QmitkMitkWorkbenchIntroPart, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkMitkWorkbenchIntroPreferencePage, context)
}

ctkPluginContext* QmitkMitkWorkbenchIntroPlugin::GetPluginContext() const
{
  return context;
}
