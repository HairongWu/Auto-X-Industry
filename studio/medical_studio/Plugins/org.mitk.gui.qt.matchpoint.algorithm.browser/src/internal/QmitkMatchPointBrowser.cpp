/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "org_mitk_gui_qt_matchpoint_algorithm_browser_Activator.h"

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "QmitkMatchPointBrowser.h"

// Qt
#include <QMessageBox>
#include <QErrorMessage>
#include <QDir>
#include <QStringList>

//MITK
#include <mitkStatusBar.h>
#include "MatchPointBrowserConstants.h"
#include "mitkAlgorithmInfoSelectionProvider.h"
#include <mitkCoreServices.h>
#include <mitkIPreferencesService.h>
#include <mitkIPreferences.h>

// MatchPoint
#include "mapRegistrationAlgorithmInterface.h"
#include "mapAlgorithmEvents.h"
#include "mapAlgorithmWrapperEvent.h"
#include "mapExceptionObjectMacros.h"
#include "mapDeploymentDLLDirectoryBrowser.h"
#include "mapDeploymentEvents.h"

const std::string QmitkMatchPointBrowser::VIEW_ID = "org.mitk.views.matchpoint.algorithm.browser";

QmitkMatchPointBrowser::QmitkMatchPointBrowser()
    : m_Parent(nullptr), m_LoadedDLLHandle(nullptr), m_LoadedAlgorithm(nullptr)
{
}

QmitkMatchPointBrowser::~QmitkMatchPointBrowser()
{
}

void QmitkMatchPointBrowser::OnPreferencesChanged(const mitk::IPreferences* /*prefs*/)
{
    this->OnSearchFolderButtonPushed();
}

void QmitkMatchPointBrowser::CreateConnections()
{
    connect(m_Controls.m_pbSearchFolder, SIGNAL(clicked()), this, SLOT(OnSearchFolderButtonPushed()));
    connect(m_Controls.m_algoTreeView, SIGNAL(clicked(const QModelIndex&)), this,
        SLOT(OnAlgoListSelectionChanged(const QModelIndex&)));
    connect(m_Controls.pbClearSearch, SIGNAL(clicked()), m_Controls.lineSearch, SLOT(clear()));
    connect(m_Controls.lineSearch, SIGNAL(textChanged(const QString&)), this,
        SLOT(OnSearchChanged(const QString&)));
}

void QmitkMatchPointBrowser::OnSearchFolderButtonPushed()
{
    RetrieveAndStorePreferenceValues();
    // test if some folder list non-empty
    int folderCount = m_currentSearchPaths.count();

    if (!folderCount)
    {
        Error(QString("No search folder selected for MatchPoint algorithm browser! Please set search paths in the MatchPoint preference page."));
        m_DLLInfoList.clear();
    }
    else
    {
        map::deployment::DLLDirectoryBrowser::Pointer browser = map::deployment::DLLDirectoryBrowser::New();
        auto validCommand = ::itk::MemberCommand<QmitkMatchPointBrowser>::New();
        validCommand->SetCallbackFunction(this, &QmitkMatchPointBrowser::OnValidDeploymentEvent);
        browser->AddObserver(::map::events::ValidDLLEvent(), validCommand);
        auto invalidCommand = ::itk::MemberCommand<QmitkMatchPointBrowser>::New();
        invalidCommand->SetCallbackFunction(this, &QmitkMatchPointBrowser::OnInvalidDeploymentEvent);
        browser->AddObserver(::map::events::InvalidDLLEvent(), invalidCommand);

        foreach(QString path, m_currentSearchPaths)
        {
            browser->addDLLSearchLocation(path.toStdString());
        }

        browser->update();
        m_DLLInfoList = browser->getLibraryInfos();
    }

    m_Controls.groupWarning->setVisible(m_DLLInfoList.empty());
    m_Controls.groupList->setVisible(!m_DLLInfoList.empty());

    m_algModel->SetAlgorithms(m_DLLInfoList);
    m_Controls.lineSearch->clear();
}

void QmitkMatchPointBrowser::OnAlgoListSelectionChanged(const QModelIndex& index)
{
    QVariant vIndex = index.data(Qt::UserRole).toInt();
    map::deployment::DLLInfo::ConstPointer currentItemInfo = nullptr;

    if (vIndex.isValid())
    {
        std::size_t algListIndex = vIndex.toInt();

        if (algListIndex < m_DLLInfoList.size())
        {
            currentItemInfo = m_DLLInfoList[algListIndex];
        }
    }

    m_Controls.m_teAlgorithmDetails->updateInfo(currentItemInfo);

    if (currentItemInfo)
    {
        //update selection provider
        mitk::MAPAlgorithmInfoSelection::Pointer infoSelection = mitk::MAPAlgorithmInfoSelection::Pointer(
            new mitk::MAPAlgorithmInfoSelection(currentItemInfo));
        this->m_SelectionProvider->SetInfoSelection(infoSelection);
    }
}


void QmitkMatchPointBrowser::OnSearchChanged(const QString& text)
{
    m_filterProxy->setFilterRegularExpression(text);
    m_filterProxy->setFilterCaseSensitivity(Qt::CaseInsensitive);
};

void QmitkMatchPointBrowser::OnInvalidDeploymentEvent(const ::itk::Object *, const itk::EventObject &event)
{
  auto deployEvent = dynamic_cast<const ::map::events::InvalidDLLEvent*>(&event);

  this->Error(QString("Error when try to inspect deployed registration algorithm. Details: ")+QString::fromStdString(deployEvent->getComment()));
}

void QmitkMatchPointBrowser::OnValidDeploymentEvent(const ::itk::Object *, const itk::EventObject &event)
{
  auto deployEvent = dynamic_cast<const ::map::events::ValidDLLEvent*>(&event);

  auto info = static_cast<const ::map::deployment::DLLInfo*>(deployEvent->getData());

  MITK_INFO << "Successfully inspected deployed registration algorithm. UID: " << info->getAlgorithmUID().toStr() << ". Path: " << info->getLibraryFilePath();

}


void QmitkMatchPointBrowser::CreateQtPartControl(QWidget* parent)
{
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls.setupUi(parent);
    m_Parent = parent;

    m_algModel = new QmitkAlgorithmListModel(parent);
    m_filterProxy = new QSortFilterProxyModel(parent);

    //! [Qt Selection Provider registration]
    // create new qt selection provider
    m_SelectionProvider = new mitk::AlgorithmInfoSelectionProvider();

    m_filterProxy->setSourceModel(m_algModel);
    m_filterProxy->setDynamicSortFilter(true);
    m_filterProxy->setFilterKeyColumn(-1);
    m_Controls.m_algoTreeView->setModel(m_filterProxy);
    m_Controls.m_algoTreeView->setSelectionMode(QAbstractItemView::SingleSelection);

    m_Controls.m_algoTreeView->header()->setStretchLastSection(false);
    m_Controls.m_algoTreeView->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_Controls.m_algoTreeView->setColumnHidden(3, true);

    this->CreateConnections();
}

void QmitkMatchPointBrowser::SetSelectionProvider()
{
    this->GetSite()->SetSelectionProvider(m_SelectionProvider);
}

void QmitkMatchPointBrowser::SetFocus()
{
}

void QmitkMatchPointBrowser::Error(QString msg)
{
    mitk::StatusBar::GetInstance()->DisplayErrorText(msg.toLatin1());
    MITK_ERROR << msg.toStdString().c_str();
}

void QmitkMatchPointBrowser::RetrieveAndStorePreferenceValues()
{
    auto* prefs = this->RetrievePreferences();

    bool loadApplicationDir = prefs->GetBool(MatchPointBrowserConstants::LOAD_FROM_APPLICATION_DIR, true);
    bool loadHomeDir = prefs->GetBool(MatchPointBrowserConstants::LOAD_FROM_HOME_DIR, false);
    bool loadCurrentDir = prefs->GetBool(MatchPointBrowserConstants::LOAD_FROM_CURRENT_DIR, false);
    bool loadAutoLoadDir = prefs->GetBool(MatchPointBrowserConstants::LOAD_FROM_AUTO_LOAD_DIR, false);

    // Get some default application paths.
    QStringList newPaths;

    // Here we can use the preferences to set up the builder,
    if (loadApplicationDir)
    {
        newPaths << QCoreApplication::applicationDirPath();
    }

    if (loadHomeDir)
    {
        newPaths << QDir::homePath();
    }

    if (loadCurrentDir)
    {
        newPaths << QDir::currentPath();
    }

    if (loadAutoLoadDir)
    {
        char* deployedAlgorithmLoadPath = getenv("MAP_MDRA_LOAD_PATH");

        if (deployedAlgorithmLoadPath != nullptr)
        {
            // The load path may in fact be a semi-colon or colon separated list of directories, not just one.
            QString paths(deployedAlgorithmLoadPath);

#ifdef Q_OS_WIN32
            QString pathSeparator(";");
#else
            QString pathSeparator(":");
#endif

            QStringList splitPath = paths.split(pathSeparator, Qt::SkipEmptyParts);

            foreach(QString path, splitPath)
            {
                QDir dir = QDir(path);
                newPaths << dir.absolutePath();
            }
        }

    }

    // We get additional directory paths from preferences.
    const auto pathString = QString::fromStdString(prefs->Get(MatchPointBrowserConstants::MDAR_DIRECTORIES_NODE_NAME, ""));
    QStringList additionalPaths = pathString.split(";", Qt::SkipEmptyParts);
    newPaths << additionalPaths;

    const auto additionalAlgorirthmsString = QString::fromStdString(prefs->Get(MatchPointBrowserConstants::MDAR_FILES_NODE_NAME, ""));
    additionalPaths = additionalAlgorirthmsString.split(";", Qt::SkipEmptyParts);
    newPaths << additionalPaths;

    m_currentSearchPaths = newPaths;
}

mitk::IPreferences* QmitkMatchPointBrowser::RetrievePreferences()
{
  const auto id = "/" + MatchPointBrowserConstants::VIEW_ID;
  return mitk::CoreServices::GetPreferencesService()->GetSystemPreferences()->Node(id);
}
