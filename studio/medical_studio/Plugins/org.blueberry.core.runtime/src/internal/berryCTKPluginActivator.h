/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef BERRYCTKPLUGINACTIVATOR_H
#define BERRYCTKPLUGINACTIVATOR_H

#include <ctkPluginActivator.h>

#include <QObject>

#include <berrySmartPointer.h>


namespace berry {

class ApplicationContainer;

struct IConfigurationElement;
struct IContributor;
struct IExtensionRegistry;

class org_blueberry_core_runtime_Activator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "org_blueberry_core_runtime")
  Q_INTERFACES(ctkPluginActivator)

public:

  static const bool DEBUG;

  org_blueberry_core_runtime_Activator();
  ~org_blueberry_core_runtime_Activator() override;

  void start(ctkPluginContext* context) override;
  void stop(ctkPluginContext* context) override;

  static ctkPluginContext* getPluginContext();

  static ApplicationContainer* GetContainer();

  /**
   * Returns the plug-in id of the plug-in that contains the provided symbol, or
   * a null QString if the plug-in could not be determined.
   */
  static QString getPluginId(void* symbol);

  // helper method to get a bundle from a contributor.
  static QSharedPointer<ctkPlugin> GetPlugin(const SmartPointer<IContributor>& contributor);


private:

  void startRegistry();
  void stopRegistry();

  void startAppContainer();
  void stopAppContainer();

  static ctkPluginContext* context;

  //QScopedPointer<CTKPluginListener> pluginListener;

  QScopedPointer<IExtensionRegistry> defaultRegistry;
  ctkServiceRegistration registryServiceReg;

  static QScopedPointer<ApplicationContainer> appContainer;

  QScopedPointer<QObject> userRegistryKey;
  QScopedPointer<QObject> masterRegistryKey;

};

typedef org_blueberry_core_runtime_Activator CTKPluginActivator;

}

#endif // BERRYCTKPLUGINACTIVATOR_H
