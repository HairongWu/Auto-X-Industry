/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#include "QmitkFlowApplicationWorkbenchWindowAdvisor.h"

#include <QMenu>
#include <QMenuBar>
#include <QMainWindow>
#include <QStatusBar>
#include <QString>
#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <QSettings>

#include <ctkPluginException.h>
#include <service/event/ctkEventAdmin.h>

#include <berryPlatform.h>
#include <berryPlatformUI.h>
#include <berryIActionBarConfigurer.h>
#include <berryIWorkbenchWindow.h>
#include <berryIWorkbenchPage.h>
#include <berryIPerspectiveRegistry.h>
#include <berryIPerspectiveDescriptor.h>
#include <berryIProduct.h>
#include <berryIWorkbenchPartConstants.h>
#include <berryQtPreferences.h>
#include <berryQtStyleManager.h>
#include <berryWorkbenchPlugin.h>

#include <internal/berryQtShowViewAction.h>
#include <internal/berryQtOpenPerspectiveAction.h>

#include <QmitkFileExitAction.h>
#include <QmitkCloseProjectAction.h>
#include <QmitkUndoAction.h>
#include <QmitkRedoAction.h>
#include <QmitkDefaultDropTargetListener.h>
#include <QmitkStatusBar.h>
#include <QmitkProgressBar.h>
#include <QmitkMemoryUsageIndicatorView.h>
#include <QmitkPreferencesDialog.h>
#include <QmitkApplicationConstants.h>
#include "QmitkExtFileSaveProjectAction.h"

#include <itkConfigure.h>
#include <mitkVersion.h>
#include <mitkIDataStorageService.h>
#include <mitkIDataStorageReference.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkWorkbenchUtil.h>
#include <vtkVersionMacros.h>
#include <mitkCoreServices.h>
#include <mitkIPreferencesService.h>
#include <mitkIPreferences.h>

// UGLYYY
#include "QmitkFlowApplicationWorkbenchWindowAdvisorHack.h"
#include "QmitkFlowApplicationPlugin.h"
#include "mitkUndoController.h"
#include "mitkVerboseLimitedLinearUndo.h"
#include <QToolBar>
#include <QToolButton>
#include <QMessageBox>
#include <QMouseEvent>
#include <QLabel>
#include <QmitkAboutDialog.h>

QmitkFlowApplicationWorkbenchWindowAdvisorHack* QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack =
  new QmitkFlowApplicationWorkbenchWindowAdvisorHack();

QString QmitkFlowApplicationWorkbenchWindowAdvisor::QT_SETTINGS_FILENAME = "QtSettings.ini";

class PartListenerForTitle: public berry::IPartListener
{
public:

  PartListenerForTitle(QmitkFlowApplicationWorkbenchWindowAdvisor* wa)
    : windowAdvisor(wa)
  {
  }

  Events::Types GetPartEventTypes() const override
  {
    return Events::ACTIVATED | Events::BROUGHT_TO_TOP | Events::CLOSED
      | Events::HIDDEN | Events::VISIBLE;
  }

  void PartActivated(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref.Cast<berry::IEditorReference> ())
    {
      windowAdvisor->UpdateTitle(false);
    }
  }

  void PartBroughtToTop(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref.Cast<berry::IEditorReference> ())
    {
      windowAdvisor->UpdateTitle(false);
    }
  }

  void PartClosed(const berry::IWorkbenchPartReference::Pointer& /*ref*/) override
  {
    windowAdvisor->UpdateTitle(false);
  }

  void PartHidden(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    auto lockedLastActiveEditor = windowAdvisor->lastActiveEditor.Lock();

    if (lockedLastActiveEditor.IsNotNull() && ref->GetPart(false) == lockedLastActiveEditor)
    {
      windowAdvisor->UpdateTitle(true);
    }
  }

  void PartVisible(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    auto lockedLastActiveEditor = windowAdvisor->lastActiveEditor.Lock();

    if (lockedLastActiveEditor.IsNotNull() && ref->GetPart(false) == lockedLastActiveEditor)
    {
      windowAdvisor->UpdateTitle(false);
    }
  }

private:
  QmitkFlowApplicationWorkbenchWindowAdvisor* windowAdvisor;
};

class PartListenerForImageNavigator: public berry::IPartListener
{
public:

  PartListenerForImageNavigator(QAction* act)
    : imageNavigatorAction(act)
  {
  }

  Events::Types GetPartEventTypes() const override
  {
    return Events::OPENED | Events::CLOSED | Events::HIDDEN |
      Events::VISIBLE;
  }

  void PartOpened(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref->GetId()=="org.mitk.views.imagenavigator")
    {
      imageNavigatorAction->setChecked(true);
    }
  }

  void PartClosed(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref->GetId()=="org.mitk.views.imagenavigator")
    {
      imageNavigatorAction->setChecked(false);
    }
  }

  void PartVisible(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref->GetId()=="org.mitk.views.imagenavigator")
    {
      imageNavigatorAction->setChecked(true);
    }
  }

  void PartHidden(const berry::IWorkbenchPartReference::Pointer& ref) override
  {
    if (ref->GetId()=="org.mitk.views.imagenavigator")
    {
      imageNavigatorAction->setChecked(false);
    }
  }

private:
  QAction* imageNavigatorAction;
};

class PerspectiveListenerForTitle: public berry::IPerspectiveListener
{
public:

  PerspectiveListenerForTitle(QmitkFlowApplicationWorkbenchWindowAdvisor* wa)
    : windowAdvisor(wa)
    , perspectivesClosed(false)
  {
  }

  Events::Types GetPerspectiveEventTypes() const override
  {
    return Events::ACTIVATED | Events::SAVED_AS | Events::DEACTIVATED
      | Events::CLOSED | Events::OPENED;
  }

  void PerspectiveActivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& /*perspective*/) override
  {
    windowAdvisor->UpdateTitle(false);
  }

  void PerspectiveSavedAs(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& /*oldPerspective*/,
    const berry::IPerspectiveDescriptor::Pointer& /*newPerspective*/) override
  {
    windowAdvisor->UpdateTitle(false);
  }

  void PerspectiveDeactivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& /*perspective*/) override
  {
    windowAdvisor->UpdateTitle(false);
  }

  void PerspectiveOpened(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& /*perspective*/) override
  {
    if (perspectivesClosed)
    {
      QListIterator<QAction*> i(windowAdvisor->viewActions);
      while (i.hasNext())
      {
        i.next()->setEnabled(true);
      }

      windowAdvisor->fileSaveProjectAction->setEnabled(true);
      windowAdvisor->undoAction->setEnabled(true);
      windowAdvisor->redoAction->setEnabled(true);
      windowAdvisor->imageNavigatorAction->setEnabled(true);
      windowAdvisor->resetPerspAction->setEnabled(true);
    }

    perspectivesClosed = false;
  }

  void PerspectiveClosed(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& /*perspective*/) override
  {
    berry::IWorkbenchWindow::Pointer wnd = windowAdvisor->GetWindowConfigurer()->GetWindow();
    bool allClosed = true;
    if (wnd->GetActivePage())
    {
      QList<berry::IPerspectiveDescriptor::Pointer> perspectives(wnd->GetActivePage()->GetOpenPerspectives());
      allClosed = perspectives.empty();
    }

    if (allClosed)
    {
      perspectivesClosed = true;

      QListIterator<QAction*> i(windowAdvisor->viewActions);
      while (i.hasNext())
      {
        i.next()->setEnabled(false);
      }

      windowAdvisor->fileSaveProjectAction->setEnabled(false);
      windowAdvisor->undoAction->setEnabled(false);
      windowAdvisor->redoAction->setEnabled(false);
      windowAdvisor->imageNavigatorAction->setEnabled(false);
      windowAdvisor->resetPerspAction->setEnabled(false);
    }
  }

private:
  QmitkFlowApplicationWorkbenchWindowAdvisor* windowAdvisor;
  bool perspectivesClosed;
};

class PerspectiveListenerForMenu: public berry::IPerspectiveListener
{
public:

  PerspectiveListenerForMenu(QmitkFlowApplicationWorkbenchWindowAdvisor* wa)
    : windowAdvisor(wa)
  {
  }

  Events::Types GetPerspectiveEventTypes() const override
  {
    return Events::ACTIVATED | Events::DEACTIVATED;
  }

  void PerspectiveActivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& perspective) override
  {
    QAction* action = windowAdvisor->mapPerspIdToAction[perspective->GetId()];
    if (action)
    {
      action->setChecked(true);
    }
  }

  void PerspectiveDeactivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& perspective) override
  {
    QAction* action = windowAdvisor->mapPerspIdToAction[perspective->GetId()];
    if (action)
    {
      action->setChecked(false);
    }
  }

private:
  QmitkFlowApplicationWorkbenchWindowAdvisor* windowAdvisor;
};

QmitkFlowApplicationWorkbenchWindowAdvisor::QmitkFlowApplicationWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
                                                               berry::IWorkbenchWindowConfigurer::Pointer configurer)
  : berry::WorkbenchWindowAdvisor(configurer)
  , lastInput(nullptr)
  , wbAdvisor(wbAdvisor)
  , showViewToolbar(true)
  , showVersionInfo(true)
  , showMitkVersionInfo(true)
  , showMemoryIndicator(true)
  , dropTargetListener(new QmitkDefaultDropTargetListener)
{
  productName = QCoreApplication::applicationName();
  viewExcludeList.push_back("org.mitk.views.viewnavigator");
}

QmitkFlowApplicationWorkbenchWindowAdvisor::~QmitkFlowApplicationWorkbenchWindowAdvisor()
{
}

QWidget* QmitkFlowApplicationWorkbenchWindowAdvisor::CreateEmptyWindowContents(QWidget* parent)
{
  QWidget* parentWidget = static_cast<QWidget*>(parent);
  auto   label = new QLabel(parentWidget);
  label->setText("<b>No perspectives are open. Open a perspective in the <i>Window->Open Perspective</i> menu.</b>");
  label->setContentsMargins(10,10,10,10);
  label->setAlignment(Qt::AlignTop);
  label->setEnabled(false);
  parentWidget->layout()->addWidget(label);
  return label;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::ShowMemoryIndicator(bool show)
{
  showMemoryIndicator = show;
}

bool QmitkFlowApplicationWorkbenchWindowAdvisor::GetShowMemoryIndicator()
{
  return showMemoryIndicator;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::ShowViewToolbar(bool show)
{
  showViewToolbar = show;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::ShowVersionInfo(bool show)
{
  showVersionInfo = show;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::ShowMitkVersionInfo(bool show)
{
  showMitkVersionInfo = show;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::SetProductName(const QString& product)
{
  productName = product;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::SetWindowIcon(const QString& wndIcon)
{
  windowIcon = wndIcon;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::PostWindowCreate()
{
  // very bad hack...
  berry::IWorkbenchWindow::Pointer window = this->GetWindowConfigurer()->GetWindow();
  QMainWindow* mainWindow = qobject_cast<QMainWindow*> (window->GetShell()->GetControl());

  if (!windowIcon.isEmpty())
  {
    mainWindow->setWindowIcon(QIcon(windowIcon));
  }
  mainWindow->setContextMenuPolicy(Qt::PreventContextMenu);

  // Load icon theme
  QIcon::setThemeSearchPaths(QStringList() << QStringLiteral(":/org_mitk_icons/icons/"));
  QIcon::setThemeName(QStringLiteral("awesome"));

  // ==== Application menu ============================

  QMenuBar* menuBar = mainWindow->menuBar();
  menuBar->setContextMenuPolicy(Qt::PreventContextMenu);

#ifdef __APPLE__
  menuBar->setNativeMenuBar(true);
#else
  menuBar->setNativeMenuBar(false);
#endif

  auto basePath = QStringLiteral(":/org_mitk_icons/icons/awesome/scalable/actions/");

  fileSaveProjectAction = new QmitkExtFileSaveProjectAction(window);
  fileSaveProjectAction->setIcon(berry::QtStyleManager::ThemeIcon(basePath + "document-save.svg"));

  auto   perspGroup = new QActionGroup(menuBar);
  std::map<QString, berry::IViewDescriptor::Pointer> VDMap;

  // sort elements (converting vector to map...)
  QList<berry::IViewDescriptor::Pointer>::const_iterator iter;

  berry::IViewRegistry* viewRegistry =
    berry::PlatformUI::GetWorkbench()->GetViewRegistry();
  const QList<berry::IViewDescriptor::Pointer> viewDescriptors = viewRegistry->GetViews();

  bool skip = false;
  for (iter = viewDescriptors.begin(); iter != viewDescriptors.end(); ++iter)
  {
    // if viewExcludeList is set, it contains the id-strings of view, which
    // should not appear as an menu-entry in the menu
    if (viewExcludeList.size() > 0)
    {
      for (int i=0; i<viewExcludeList.size(); i++)
      {
        if (viewExcludeList.at(i) == (*iter)->GetId())
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        skip = false;
        continue;
      }
    }

    if ((*iter)->GetId() == "org.blueberry.ui.internal.introview")
      continue;
    if ((*iter)->GetId() == "org.mitk.views.imagenavigator")
      continue;
    if ((*iter)->GetId() == "org.mitk.views.viewnavigator")
      continue;

    std::pair<QString, berry::IViewDescriptor::Pointer> p((*iter)->GetLabel(), (*iter));
    VDMap.insert(p);
  }

  std::map<QString, berry::IViewDescriptor::Pointer>::const_iterator MapIter;
  for (MapIter = VDMap.begin(); MapIter != VDMap.end(); ++MapIter)
  {
    berry::QtShowViewAction* viewAction = new berry::QtShowViewAction(window, (*MapIter).second);
    viewActions.push_back(viewAction);
  }

  QMenu* fileMenu = menuBar->addMenu("&File");
  fileMenu->setObjectName("FileMenu");
  fileMenu->addAction(fileSaveProjectAction);
  fileMenu->addSeparator();

  QAction* fileExitAction = new QmitkFileExitAction(window);
  fileExitAction->setIcon(berry::QtStyleManager::ThemeIcon(basePath + "system-log-out.svg"));
  fileExitAction->setShortcut(QKeySequence::Quit);
  fileExitAction->setObjectName("QmitkFileExitAction");
  fileMenu->addAction(fileExitAction);

  // another bad hack to get an edit/undo menu...
  QMenu* editMenu = menuBar->addMenu("&Edit");
  undoAction = editMenu->addAction(berry::QtStyleManager::ThemeIcon(basePath + "edit-undo.svg"),
    "&Undo",
    QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack, SLOT(onUndo()),
    QKeySequence("CTRL+Z"));
  undoAction->setToolTip("Undo the last action (not supported by all modules)");
  redoAction = editMenu->addAction(berry::QtStyleManager::ThemeIcon(basePath + "edit-redo.svg"),
    "&Redo",
    QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack, SLOT(onRedo()),
    QKeySequence("CTRL+Y"));
  redoAction->setToolTip("execute the last action that was undone again (not supported by all modules)");

  // ==== Window Menu ==========================
  QMenu* windowMenu = menuBar->addMenu("Window");

  QMenu* perspMenu = windowMenu->addMenu("&Open Perspective");

  windowMenu->addSeparator();
  resetPerspAction = windowMenu->addAction("&Reset Perspective",
    QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack, SLOT(onResetPerspective()));

  windowMenu->addSeparator();
  windowMenu->addAction("&Preferences...",
    QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack, SLOT(onEditPreferences()),
    QKeySequence("CTRL+P"));

  // fill perspective menu
  berry::IPerspectiveRegistry* perspRegistry =
    window->GetWorkbench()->GetPerspectiveRegistry();

  QList<berry::IPerspectiveDescriptor::Pointer> perspectives(
    perspRegistry->GetPerspectives());

  skip = false;
  for (QList<berry::IPerspectiveDescriptor::Pointer>::iterator perspIt =
    perspectives.begin(); perspIt != perspectives.end(); ++perspIt)
  {
    // if perspectiveExcludeList is set, it contains the id-strings of perspectives, which
    // should not appear as an menu-entry in the perspective menu
    if (perspectiveExcludeList.size() > 0)
    {
      for (int i=0; i<perspectiveExcludeList.size(); i++)
      {
        if (perspectiveExcludeList.at(i) == (*perspIt)->GetId())
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        skip = false;
        continue;
      }
    }

    QAction* perspAction = new berry::QtOpenPerspectiveAction(window, *perspIt, perspGroup);
    mapPerspIdToAction.insert((*perspIt)->GetId(), perspAction);
  }
  perspMenu->addActions(perspGroup->actions());

  // ===== Help menu ====================================
  QMenu* helpMenu = menuBar->addMenu("&Help");
  helpMenu->addAction("&Welcome",this, SLOT(onIntro()));
  helpMenu->addAction("&Open Help Perspective", this, SLOT(onHelpOpenHelpPerspective()));
  helpMenu->addAction("&Context Help",this, SLOT(onHelp()),  QKeySequence("F1"));
  helpMenu->addAction("&About",this, SLOT(onAbout()));
  // =====================================================


  // toolbar for showing file open, undo, redo and other main actions
  auto   mainActionsToolBar = new QToolBar;
  mainActionsToolBar->setObjectName("mainActionsToolBar");
  mainActionsToolBar->setContextMenuPolicy(Qt::PreventContextMenu);
#ifdef __APPLE__
  mainActionsToolBar->setToolButtonStyle ( Qt::ToolButtonTextUnderIcon );
#else
  mainActionsToolBar->setToolButtonStyle ( Qt::ToolButtonTextBesideIcon );
#endif

  basePath = QStringLiteral(":/org.mitk.gui.qt.ext/");
  imageNavigatorAction = new QAction(berry::QtStyleManager::ThemeIcon(basePath + "image_navigator.svg"), "&Image Navigator", nullptr);
  bool imageNavigatorViewFound = window->GetWorkbench()->GetViewRegistry()->Find("org.mitk.views.imagenavigator");

  if (imageNavigatorViewFound)
  {
    QObject::connect(imageNavigatorAction, SIGNAL(triggered(bool)), QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack, SLOT(onImageNavigator()));
    imageNavigatorAction->setCheckable(true);

    // add part listener for image navigator
    imageNavigatorPartListener.reset(new PartListenerForImageNavigator(imageNavigatorAction));
    window->GetPartService()->AddPartListener(imageNavigatorPartListener.data());
    berry::IViewPart::Pointer imageNavigatorView = window->GetActivePage()->FindView("org.mitk.views.imagenavigator");
    imageNavigatorAction->setChecked(false);
    if (imageNavigatorView)
    {
      bool isImageNavigatorVisible = window->GetActivePage()->IsPartVisible(imageNavigatorView);
      if (isImageNavigatorVisible)
        imageNavigatorAction->setChecked(true);
    }
    imageNavigatorAction->setToolTip("Toggle image navigator for navigating through image");
  }

  mainActionsToolBar->addAction(undoAction);
  mainActionsToolBar->addAction(redoAction);

  if (imageNavigatorViewFound)
  {
    mainActionsToolBar->addAction(imageNavigatorAction);
  }

  mainWindow->addToolBar(mainActionsToolBar);

  // ==== View Toolbar ==================================

  if (showViewToolbar)
  {
    auto* prefService = mitk::CoreServices::GetPreferencesService();
    auto* toolBarsPrefs = prefService->GetSystemPreferences()->Node(QmitkApplicationConstants::TOOL_BARS_PREFERENCES);
    bool showCategories = toolBarsPrefs->GetBool(QmitkApplicationConstants::TOOL_BARS_SHOW_CATEGORIES, true);

    // Order view descriptors by category

    QMultiMap<QString, berry::IViewDescriptor::Pointer> categoryViewDescriptorMap;

    for (auto labelViewDescriptorPair : VDMap)
    {
      auto viewDescriptor = labelViewDescriptorPair.second;
      auto category = !viewDescriptor->GetCategoryPath().isEmpty()
        ? viewDescriptor->GetCategoryPath().back()
        : QString();

      categoryViewDescriptorMap.insert(category, viewDescriptor);
    }

    // Create a separate toolbar for each category

    for (auto category : categoryViewDescriptorMap.uniqueKeys())
    {
      auto viewDescriptorsInCurrentCategory = categoryViewDescriptorMap.values(category);
      QList<berry::SmartPointer<berry::IViewDescriptor> > relevantViewDescriptors;

      for (auto viewDescriptor : viewDescriptorsInCurrentCategory)
      {
        if (viewDescriptor->GetId() != "org.mitk.views.flow.control" &&
            viewDescriptor->GetId() != "org.mitk.views.segmentationtasklist")
        {
          relevantViewDescriptors.push_back(viewDescriptor);
        }
      }

      if (!relevantViewDescriptors.isEmpty())
      {
        auto toolbar = new QToolBar;
        toolbar->setObjectName(category);
        mainWindow->addToolBar(toolbar);

        toolbar->setVisible(toolBarsPrefs->GetBool(category.toStdString(), true));

        if (!category.isEmpty())
        {
          auto categoryButton = new QToolButton;
          categoryButton->setToolButtonStyle(Qt::ToolButtonTextOnly);
          categoryButton->setText(category);
          categoryButton->setStyleSheet("background: transparent; margin: 0; padding: 0;");

          auto action = toolbar->addWidget(categoryButton);
          action->setObjectName("category");
          action->setVisible(showCategories);

          connect(categoryButton, &QToolButton::clicked, [toolbar]()
          {
            for (QWidget* widget : toolbar->findChildren<QWidget*>())
            {
              if (QStringLiteral("qt_toolbar_ext_button") == widget->objectName() && widget->isVisible())
              {
                QMouseEvent pressEvent(QEvent::MouseButtonPress, QPointF(0.0f, 0.0f), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
                QMouseEvent releaseEvent(QEvent::MouseButtonRelease, QPointF(0.0f, 0.0f), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
                QApplication::sendEvent(widget, &pressEvent);
                QApplication::sendEvent(widget, &releaseEvent);
              }
            }
          });
        }

        for (auto viewDescriptor : relevantViewDescriptors)
        {
          auto viewAction = new berry::QtShowViewAction(window, viewDescriptor);
          toolbar->addAction(viewAction);
        }
      }
    }
  }

  QSettings settings(GetQSettingsFile(), QSettings::IniFormat);
  mainWindow->restoreState(settings.value("ToolbarPosition").toByteArray());

  auto   qStatusBar = new QStatusBar();

  //creating a QmitkStatusBar for Output on the QStatusBar and connecting it with the MainStatusBar
  auto  statusBar = new QmitkStatusBar(qStatusBar);
  //disabling the SizeGrip in the lower right corner
  statusBar->SetSizeGripEnabled(false);

  auto  progBar = new QmitkProgressBar();

  qStatusBar->addPermanentWidget(progBar, 0);
  progBar->hide();

  mainWindow->setStatusBar(qStatusBar);

  if (showMemoryIndicator)
  {
    auto   memoryIndicator = new QmitkMemoryUsageIndicatorView();
    qStatusBar->addPermanentWidget(memoryIndicator, 0);
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::PreWindowOpen()
{
  berry::IWorkbenchWindowConfigurer::Pointer configurer = GetWindowConfigurer();

  this->HookTitleUpdateListeners(configurer);

  menuPerspectiveListener.reset(new PerspectiveListenerForMenu(this));
  configurer->GetWindow()->AddPerspectiveListener(menuPerspectiveListener.data());

  configurer->AddEditorAreaTransfer(QStringList("text/uri-list"));
  configurer->ConfigureEditorAreaDropListener(dropTargetListener.data());
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::PostWindowOpen()
{
  berry::WorkbenchWindowAdvisor::PostWindowOpen();
  // Force Rendering Window Creation on startup.
  berry::IWorkbenchWindowConfigurer::Pointer configurer = GetWindowConfigurer();

  ctkPluginContext* context = QmitkFlowApplicationPlugin::GetDefault()->GetPluginContext();
  ctkServiceReference serviceRef = context->getServiceReference<mitk::IDataStorageService>();
  if (serviceRef)
  {
    mitk::IDataStorageService *dsService = context->getService<mitk::IDataStorageService>(serviceRef);
    if (dsService)
    {
      mitk::IDataStorageReference::Pointer dsRef = dsService->GetDataStorage();
      mitk::DataStorageEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(dsRef));
      mitk::WorkbenchUtil::OpenEditor(configurer->GetWindow()->GetActivePage(),dsInput);
    }
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::onIntro()
{
  QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack->onIntro();
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::onHelp()
{
  QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack->onHelp();
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::onHelpOpenHelpPerspective()
{
  QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack->onHelpOpenHelpPerspective();
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::onAbout()
{
  QmitkFlowApplicationWorkbenchWindowAdvisorHack::undohack->onAbout();
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::HookTitleUpdateListeners(berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  // hook up the listeners to update the window title
  titlePartListener.reset(new PartListenerForTitle(this));
  titlePerspectiveListener.reset(new PerspectiveListenerForTitle(this));
  editorPropertyListener.reset(new berry::PropertyChangeIntAdapter<
    QmitkFlowApplicationWorkbenchWindowAdvisor>(this,
    &QmitkFlowApplicationWorkbenchWindowAdvisor::PropertyChange));

  configurer->GetWindow()->AddPerspectiveListener(titlePerspectiveListener.data());
  configurer->GetWindow()->GetPartService()->AddPartListener(titlePartListener.data());
}

QString QmitkFlowApplicationWorkbenchWindowAdvisor::ComputeTitle()
{
  berry::IWorkbenchWindowConfigurer::Pointer configurer = GetWindowConfigurer();
  berry::IWorkbenchPage::Pointer currentPage = configurer->GetWindow()->GetActivePage();
  berry::IEditorPart::Pointer activeEditor;
  if (currentPage)
  {
    activeEditor = lastActiveEditor.Lock();
  }

  QString title;
  berry::IProduct::Pointer product = berry::Platform::GetProduct();
  if (product.IsNotNull())
  {
    title = product->GetName();
  }
  if (title.isEmpty())
  {
    // instead of the product name, we use a custom variable for now
    title = productName;
  }

  if(showMitkVersionInfo)
  {
    QString mitkVersionInfo = MITK_REVISION_DESC;

    if(mitkVersionInfo.isEmpty())
      mitkVersionInfo = MITK_VERSION_STRING;

    title += " " + mitkVersionInfo;
  }

  if (showVersionInfo)
  {
    // add version informatioin
    QString versions = QString(" (ITK %1.%2.%3 | VTK %4.%5.%6 | Qt %7)")
      .arg(ITK_VERSION_MAJOR).arg(ITK_VERSION_MINOR).arg(ITK_VERSION_PATCH)
      .arg(VTK_MAJOR_VERSION).arg(VTK_MINOR_VERSION).arg(VTK_BUILD_VERSION)
      .arg(QT_VERSION_STR);

    title += versions;
  }

  if (currentPage)
  {
    if (activeEditor)
    {
      lastEditorTitle = activeEditor->GetTitleToolTip();
      if (!lastEditorTitle.isEmpty())
        title = lastEditorTitle + " - " + title;
    }
    berry::IPerspectiveDescriptor::Pointer persp = currentPage->GetPerspective();
    QString label = "";
    if (persp)
    {
      label = persp->GetLabel();
    }
    berry::IAdaptable* input = currentPage->GetInput();
    if (input && input != wbAdvisor->GetDefaultPageInput())
    {
      label = currentPage->GetLabel();
    }
    if (!label.isEmpty())
    {
      title = label + " - " + title;
    }
  }

  title += " (Not for use in diagnosis or treatment of patients)";

  return title;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::RecomputeTitle()
{
  berry::IWorkbenchWindowConfigurer::Pointer configurer = GetWindowConfigurer();
  QString oldTitle = configurer->GetTitle();
  QString newTitle = ComputeTitle();
  if (newTitle != oldTitle)
  {
    configurer->SetTitle(newTitle);
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::UpdateTitle(bool editorHidden)
{
  berry::IWorkbenchWindowConfigurer::Pointer configurer = GetWindowConfigurer();
  berry::IWorkbenchWindow::Pointer window = configurer->GetWindow();
  berry::IEditorPart::Pointer activeEditor;
  berry::IWorkbenchPage::Pointer currentPage = window->GetActivePage();
  berry::IPerspectiveDescriptor::Pointer persp;
  berry::IAdaptable* input = nullptr;

  if (currentPage)
  {
    activeEditor = currentPage->GetActiveEditor();
    persp = currentPage->GetPerspective();
    input = currentPage->GetInput();
  }

  if (editorHidden)
  {
    activeEditor = nullptr;
  }

  // Nothing to do if the editor hasn't changed
  if (activeEditor == lastActiveEditor.Lock() && currentPage == lastActivePage.Lock()
    && persp == lastPerspective.Lock() && input == lastInput)
  {
    return;
  }

  auto lockedLastActiveEditor = lastActiveEditor.Lock();

  if (lockedLastActiveEditor.IsNotNull())
  {
    lockedLastActiveEditor->RemovePropertyListener(editorPropertyListener.data());
  }

  lastActiveEditor = activeEditor;
  lastActivePage = currentPage;
  lastPerspective = persp;
  lastInput = input;

  if (activeEditor)
  {
    activeEditor->AddPropertyListener(editorPropertyListener.data());
  }

  RecomputeTitle();
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::PropertyChange(const berry::Object::Pointer& /*source*/, int propId)
{
  if (propId == berry::IWorkbenchPartConstants::PROP_TITLE)
  {
    auto lockedLastActiveEditor = lastActiveEditor.Lock();

    if (lockedLastActiveEditor.IsNotNull())
    {
      QString newTitle = lockedLastActiveEditor->GetPartName();
      if (lastEditorTitle != newTitle)
      {
        RecomputeTitle();
      }
    }
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::SetPerspectiveExcludeList(const QList<QString>& v)
{
  this->perspectiveExcludeList = v;
}

QList<QString> QmitkFlowApplicationWorkbenchWindowAdvisor::GetPerspectiveExcludeList()
{
  return this->perspectiveExcludeList;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::SetViewExcludeList(const QList<QString>& v)
{
  this->viewExcludeList = v;
}

QList<QString> QmitkFlowApplicationWorkbenchWindowAdvisor::GetViewExcludeList()
{
  return this->viewExcludeList;
}

void QmitkFlowApplicationWorkbenchWindowAdvisor::PostWindowClose()
{
  berry::IWorkbenchWindow::Pointer window = this->GetWindowConfigurer()->GetWindow();
  QMainWindow* mainWindow = static_cast<QMainWindow*> (window->GetShell()->GetControl());

  auto fileName = this->GetQSettingsFile();

  if (!fileName.isEmpty())
  {
    QSettings settings(fileName, QSettings::IniFormat);
    settings.setValue("ToolbarPosition", mainWindow->saveState());
  }
}

QString QmitkFlowApplicationWorkbenchWindowAdvisor::GetQSettingsFile() const
{
  QFileInfo settingsInfo = QmitkFlowApplicationPlugin::GetDefault()->GetPluginContext()->getDataFile(QT_SETTINGS_FILENAME);
  return settingsInfo.canonicalFilePath();
}

//--------------------------------------------------------------------------------
// Ugly hack from here on. Feel free to delete when command framework
// and undo buttons are done.
//--------------------------------------------------------------------------------

QmitkFlowApplicationWorkbenchWindowAdvisorHack::QmitkFlowApplicationWorkbenchWindowAdvisorHack()
  : QObject()
{
}

QmitkFlowApplicationWorkbenchWindowAdvisorHack::~QmitkFlowApplicationWorkbenchWindowAdvisorHack()
{
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onUndo()
{
  mitk::UndoModel* model = mitk::UndoController::GetCurrentUndoModel();
  if (model)
  {
    if (mitk::VerboseLimitedLinearUndo* verboseundo = dynamic_cast<mitk::VerboseLimitedLinearUndo*>(model))
    {
      mitk::VerboseLimitedLinearUndo::StackDescription descriptions = verboseundo->GetUndoDescriptions();
      if (descriptions.size() >= 1)
      {
        MITK_INFO << "Undo " << descriptions.front().second;
      }
    }
    model->Undo();
  }
  else
  {
    MITK_ERROR << "No undo model instantiated";
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onRedo()
{
  mitk::UndoModel* model = mitk::UndoController::GetCurrentUndoModel();
  if (model)
  {
    if (mitk::VerboseLimitedLinearUndo* verboseundo = dynamic_cast<mitk::VerboseLimitedLinearUndo*>(model))
    {
      mitk::VerboseLimitedLinearUndo::StackDescription descriptions = verboseundo->GetRedoDescriptions();
      if (descriptions.size() >= 1)
      {
        MITK_INFO << "Redo " << descriptions.front().second;
      }
    }
    model->Redo();
  }
  else
  {
    MITK_ERROR << "No undo model instantiated";
  }
}

// safe calls to the complete chain
// berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow()->GetActivePage()->FindView("org.mitk.views.imagenavigator");
// to cover for all possible cases of closed pages etc.
static void SafeHandleNavigatorView(QString view_query_name)
{
  berry::IWorkbench* wbench = berry::PlatformUI::GetWorkbench();
  if (wbench == nullptr)
    return;

  berry::IWorkbenchWindow::Pointer wbench_window = wbench->GetActiveWorkbenchWindow();
  if (wbench_window.IsNull())
    return;

  berry::IWorkbenchPage::Pointer wbench_page = wbench_window->GetActivePage();
  if (wbench_page.IsNull())
    return;

  auto wbench_view = wbench_page->FindView(view_query_name);

  if (wbench_view.IsNotNull())
  {
    bool isViewVisible = wbench_page->IsPartVisible(wbench_view);
    if (isViewVisible)
    {
      wbench_page->HideView(wbench_view);
      return;
    }

  }

  wbench_page->ShowView(view_query_name);
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onImageNavigator()
{
  // show/hide ImageNavigatorView
  SafeHandleNavigatorView("org.mitk.views.imagenavigator");
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onEditPreferences()
{
  QmitkPreferencesDialog _PreferencesDialog(QApplication::activeWindow());
  _PreferencesDialog.exec();
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onQuit()
{
  berry::PlatformUI::GetWorkbench()->Close();
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onResetPerspective()
{
  berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow()->GetActivePage()->ResetPerspective();
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onClosePerspective()
{
  berry::IWorkbenchPage::Pointer page =
    berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow()->GetActivePage();
  page->ClosePerspective(page->GetPerspective(), true, true);
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onIntro()
{
  bool hasIntro =
    berry::PlatformUI::GetWorkbench()->GetIntroManager()->HasIntro();
  if (!hasIntro)
  {
    QRegularExpression reg("(.*)<title>(\\n)*");
    QRegularExpression reg2("(\\n)*</title>(.*)");
    QFile file(":/org.mitk.gui.qt.ext/index.html");
    file.open(QIODevice::ReadOnly | QIODevice::Text); //text file only for reading

    QString text = QString(file.readAll());

    file.close();

    QString title = text;
    title.replace(reg, "");
    title.replace(reg2, "");

    std::cout << title.toStdString() << std::endl;

    QMessageBox::information(nullptr, title,
      text, "Close");
  }
  else
  {
    berry::PlatformUI::GetWorkbench()->GetIntroManager()->ShowIntro(
      berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow(), false);
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onHelp()
{
  ctkPluginContext* context = QmitkFlowApplicationPlugin::GetDefault()->GetPluginContext();
  if (context == nullptr)
  {
    MITK_WARN << "Plugin context not set, unable to open context help";
    return;
  }

  // Check if the org.blueberry.ui.qt.help plug-in is installed and started
  QList<QSharedPointer<ctkPlugin> > plugins = context->getPlugins();
  foreach(QSharedPointer<ctkPlugin> p, plugins)
  {
    if (p->getSymbolicName() == "org.blueberry.ui.qt.help")
    {
      if (p->getState() != ctkPlugin::ACTIVE)
      {
        // try to activate the plug-in explicitly
        try
        {
          p->start(ctkPlugin::START_TRANSIENT);
        }
        catch (const ctkPluginException& pe)
        {
          MITK_ERROR << "Activating org.blueberry.ui.qt.help failed: " << pe.what();
          return;
        }
      }
    }
  }

  ctkServiceReference eventAdminRef = context->getServiceReference<ctkEventAdmin>();
  ctkEventAdmin* eventAdmin = nullptr;
  if (eventAdminRef)
  {
    eventAdmin = context->getService<ctkEventAdmin>(eventAdminRef);
  }
  if (eventAdmin == nullptr)
  {
    MITK_WARN << "ctkEventAdmin service not found. Unable to open context help";
  }
  else
  {
    ctkEvent ev("org/blueberry/ui/help/CONTEXTHELP_REQUESTED");
    eventAdmin->postEvent(ev);
  }
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onHelpOpenHelpPerspective()
{
  berry::PlatformUI::GetWorkbench()->ShowPerspective("org.blueberry.perspectives.help",
    berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow());
}

void QmitkFlowApplicationWorkbenchWindowAdvisorHack::onAbout()
{
  auto aboutDialog = new QmitkAboutDialog(QApplication::activeWindow());
  aboutDialog->open();
}
