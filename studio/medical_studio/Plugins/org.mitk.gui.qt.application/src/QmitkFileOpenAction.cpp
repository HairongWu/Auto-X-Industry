/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkFileOpenAction.h"

#include "internal/org_mitk_gui_qt_application_Activator.h"

#include <mitkIDataStorageService.h>
#include <mitkNodePredicateProperty.h>
#include <mitkWorkbenchUtil.h>
#include <mitkCoreServices.h>
#include <mitkIPreferencesService.h>
#include <mitkIPreferences.h>

#include <QmitkIOUtil.h>

#include <QFileDialog>

namespace
{
  mitk::DataStorage::Pointer GetDataStorage()
  {
    auto context = mitk::org_mitk_gui_qt_application_Activator::GetContext();

    if (nullptr == context)
      return nullptr;

    auto dataStorageServiceReference = context->getServiceReference<mitk::IDataStorageService>();

    if (!dataStorageServiceReference)
      return nullptr;

    auto dataStorageService = context->getService<mitk::IDataStorageService>(dataStorageServiceReference);

    if (nullptr == dataStorageService)
      return nullptr;

    auto dataStorageReference = dataStorageService->GetDataStorage();

    if (dataStorageReference.IsNull())
      return nullptr;

    return dataStorageReference->GetDataStorage();
  }

  mitk::DataNode::Pointer GetFirstSelectedNode()
  {
    auto dataStorage = GetDataStorage();

    if (dataStorage.IsNull())
      return nullptr;

    auto selectedNodes = dataStorage->GetSubset(mitk::NodePredicateProperty::New("selected", mitk::BoolProperty::New(true)));

    if (selectedNodes->empty())
      return nullptr;

    return selectedNodes->front();
  }

  QString GetPathOfFirstSelectedNode()
  {
    auto firstSelectedNode = GetFirstSelectedNode();

    if (firstSelectedNode.IsNull())
      return "";

    auto data = firstSelectedNode->GetData();

    if (nullptr == data)
      return "";

    auto pathProperty = data->GetConstProperty("path");

    if (pathProperty.IsNull())
      return "";

    return QFileInfo(QString::fromStdString(pathProperty->GetValueAsString())).canonicalPath();
  }
}

class QmitkFileOpenActionPrivate
{
public:

  void Init(berry::IWorkbenchWindow* window, QmitkFileOpenAction* action)
  {
    m_Window = window;
    action->setText("&Open File...");
    action->setToolTip("Open data files (images, surfaces,...)");

    QObject::connect(action, SIGNAL(triggered(bool)), action, SLOT(Run()));
  }

  mitk::IPreferences* GetPreferences() const
  {
    auto* prefService = mitk::CoreServices::GetPreferencesService();

    return prefService != nullptr
      ? prefService->GetSystemPreferences()->Node("/General")
      : nullptr;
  }

  QString GetLastFileOpenPath() const
  {
    auto* prefs = GetPreferences();
    
    return prefs != nullptr
      ? QString::fromStdString(prefs->Get("LastFileOpenPath", ""))
      : QString();
  }

  void SetLastFileOpenPath(const QString& path) const
  {
    auto* prefs = GetPreferences();
    if (prefs != nullptr)
    {
      prefs->Put("LastFileOpenPath", path.toStdString());
      prefs->Flush();
    }
  }

  bool GetOpenEditor() const
  {
    auto* prefs = GetPreferences();

    return prefs != nullptr
      ? prefs->GetBool("OpenEditor", true)
      : true;
  }

  berry::IWorkbenchWindow* m_Window;
};

QmitkFileOpenAction::QmitkFileOpenAction(berry::IWorkbenchWindow::Pointer window)
  : QAction(nullptr)
  , d(new QmitkFileOpenActionPrivate)
{
  d->Init(window.GetPointer(), this);
}

QmitkFileOpenAction::QmitkFileOpenAction(const QIcon& icon, berry::IWorkbenchWindow::Pointer window)
  : QAction(nullptr)
  , d(new QmitkFileOpenActionPrivate)
{
  d->Init(window.GetPointer(), this);
  setIcon(icon);
}

QmitkFileOpenAction::QmitkFileOpenAction(const QIcon& icon, berry::IWorkbenchWindow* window)
  : QAction(nullptr), d(new QmitkFileOpenActionPrivate)
{
  d->Init(window, this);
  setIcon(icon);
}

QmitkFileOpenAction::~QmitkFileOpenAction()
{
}

void QmitkFileOpenAction::Run()
{
  auto path = GetPathOfFirstSelectedNode();

  if (path.isEmpty())
    path = d->GetLastFileOpenPath();

  // Ask the user for a list of files to open
  QStringList fileNames = QFileDialog::getOpenFileNames(nullptr, "Open",
                                                        path,
                                                        QmitkIOUtil::GetFileOpenFilterString());

  if (fileNames.empty())
  {
    return;
  }

  d->SetLastFileOpenPath(fileNames.front());
  mitk::WorkbenchUtil::LoadFiles(fileNames, berry::IWorkbenchWindow::Pointer(d->m_Window), d->GetOpenEditor());
}
