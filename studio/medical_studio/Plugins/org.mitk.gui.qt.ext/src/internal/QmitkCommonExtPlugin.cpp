/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkCommonExtPlugin.h"

#include <QtWidgetsExtRegisterClasses.h>

#include "QmitkAboutHandler.h"
#include "QmitkAppInstancesPreferencePage.h"
#include "QmitkStartupPreferencePage.h"

#include "QmitkModuleView.h"

#include <mitkIDataStorageService.h>
#include <mitkSceneIO.h>
#include <mitkProgressBar.h>
#include <mitkRenderingManager.h>
#include <mitkIOUtil.h>
#include <mitkCoreServices.h>
#include <mitkIPreferencesService.h>
#include <mitkIPreferences.h>

#include <mitkBaseApplication.h>

#include <berryPlatformUI.h>
#include <berryPlatform.h>

#include <Poco/Util/OptionProcessor.h>

#include <QApplication>
#include <QProcess>
#include <QMainWindow>

#include <usModuleInitialization.h>

US_INITIALIZE_MODULE

ctkPluginContext* QmitkCommonExtPlugin::_context = nullptr;

void QmitkCommonExtPlugin::start(ctkPluginContext* context)
{
  this->_context = context;

  QtWidgetsExtRegisterClasses();

  BERRY_REGISTER_EXTENSION_CLASS(QmitkAboutHandler, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkAppInstancesPreferencePage, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkStartupPreferencePage, context)

  BERRY_REGISTER_EXTENSION_CLASS(QmitkModuleView, context)

  if (qApp->metaObject()->indexOfSignal("messageReceived(QByteArray)") > -1)
  {
    connect(qApp, SIGNAL(messageReceived(QByteArray)), this, SLOT(handleIPCMessage(QByteArray)));
  }

  // This is a potentially long running operation.
  loadDataFromDisk(berry::Platform::GetApplicationArgs(), true);
}

void QmitkCommonExtPlugin::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)

  this->_context = nullptr;
}

ctkPluginContext* QmitkCommonExtPlugin::getContext()
{
  return _context;
}

void QmitkCommonExtPlugin::loadDataFromDisk(const QStringList &arguments, bool globalReinit)
{
  if (!arguments.empty())
  {
    ctkServiceReference serviceRef = _context->getServiceReference<mitk::IDataStorageService>();
    if (serviceRef)
    {
       mitk::IDataStorageService* dataStorageService = _context->getService<mitk::IDataStorageService>(serviceRef);
       mitk::DataStorage::Pointer dataStorage = dataStorageService->GetDefaultDataStorage()->GetDataStorage();

       int argumentsAdded = 0;
       for (int i = 0; i < arguments.size(); ++i)
       {
         if (arguments[i].right(5) == ".mitk")
         {
           mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();

           bool clearDataStorageFirst(false);
           mitk::ProgressBar::GetInstance()->AddStepsToDo(2);
           dataStorage = sceneIO->LoadScene( arguments[i].toLocal8Bit().constData(), dataStorage, clearDataStorageFirst );
           mitk::ProgressBar::GetInstance()->Progress(2);
           argumentsAdded++;
         }
         else if (arguments[i].right(15) == ".mitksceneindex")
         {
           mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();

           bool clearDataStorageFirst(false);
           mitk::ProgressBar::GetInstance()->AddStepsToDo(2);
           dataStorage = sceneIO->LoadSceneUnzipped(arguments[i].toLocal8Bit().constData(), dataStorage, clearDataStorageFirst);
           mitk::ProgressBar::GetInstance()->Progress(2);
           argumentsAdded++;
         }
         else
         {
           try
           {
             const std::string path(arguments[i].toStdString());
             auto addedNodes = mitk::IOUtil::Load(path, *dataStorage);

             for (const auto& node : *addedNodes )
             {
               node->SetIntProperty("layer", argumentsAdded);
             }

             argumentsAdded++;
           }
           catch(...)
           {
             MITK_WARN << "Failed to load command line argument: " << arguments[i].toStdString();
           }
         }
       } // end for each command line argument

       if (argumentsAdded > 0 && globalReinit)
       {
         // calculate bounding geometry
         mitk::RenderingManager::GetInstance()->InitializeViews(dataStorage->ComputeBoundingGeometry3D());
       }
    }
    else
    {
      MITK_ERROR << "A service reference for mitk::IDataStorageService does not exist";
    }
  }
}

void QmitkCommonExtPlugin::startNewInstance(const QStringList &args, const QStringList& files)
{
  QStringList newArgs(args);
#ifdef Q_OS_UNIX
  newArgs << QString("--") + mitk::BaseApplication::ARG_NEWINSTANCE;
#else
  newArgs << QString("/") + mitk::BaseApplication::ARG_NEWINSTANCE;
#endif
  newArgs << files;
  QProcess::startDetached(qApp->applicationFilePath(), newArgs);
}

void QmitkCommonExtPlugin::handleIPCMessage(const QByteArray& msg)
{
  QDataStream ds(msg);
  QString msgType;
  ds >> msgType;

  // we only handle messages containing command line arguments
  if (msgType != "$cmdLineArgs") return;

  // activate the current workbench window
  berry::IWorkbenchWindow::Pointer window =
      berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow();

  QMainWindow* mainWindow =
   static_cast<QMainWindow*> (window->GetShell()->GetControl());

  mainWindow->setWindowState(mainWindow->windowState() & ~Qt::WindowMinimized);
  mainWindow->raise();
  mainWindow->activateWindow();

  // Get the preferences for the instantiation behavior
  auto* prefService = mitk::CoreServices::GetPreferencesService();
  auto* prefs = prefService->GetSystemPreferences()->Node("/General");
  bool newInstanceAlways = prefs->GetBool("newInstance.always", false);
  bool newInstanceScene = prefs->GetBool("newInstance.scene", true);

  QStringList args;
  ds >> args;

  QStringList fileArgs;
  QStringList sceneArgs;

  foreach (QString arg, args)
  {
    if (arg.endsWith(".mitk"))
    {
      sceneArgs << arg;
    }
    else
    {
      fileArgs << arg;
    }
  }

  if (newInstanceAlways)
  {
    if (newInstanceScene)
    {
      startNewInstance(args, fileArgs);

      foreach(QString sceneFile, sceneArgs)
      {
        startNewInstance(args, QStringList(sceneFile));
      }
    }
    else
    {
      fileArgs.append(sceneArgs);
      startNewInstance(args, fileArgs);
    }
  }
  else
  {
    loadDataFromDisk(fileArgs, false);
    if (newInstanceScene)
    {
      foreach(QString sceneFile, sceneArgs)
      {
        startNewInstance(args, QStringList(sceneFile));
      }
    }
    else
    {
      loadDataFromDisk(sceneArgs, false);
    }
  }

}
