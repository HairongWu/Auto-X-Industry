/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkFlowApplicationWorkbenchWindowAdvisorHack_h
#define QmitkFlowApplicationWorkbenchWindowAdvisorHack_h

#include <QObject>

class ctkPluginContext;
class QmitkPreferencesDialog;

/** This class is a "hack" due to the currently missing command framework. It is a direct clone of QmitkExtWorkbenchWindowAdvisorHack.*/
class QmitkFlowApplicationWorkbenchWindowAdvisorHack : public QObject
{
  Q_OBJECT

  public slots:

    void onUndo();
    void onRedo();
    void onImageNavigator();
    void onEditPreferences();
    void onQuit();

    void onResetPerspective();
    void onClosePerspective();
    void onIntro();

    /**
     * @brief This slot is called if the user clicks the menu item "help->context help" or presses F1.
     * The help page is shown in a workbench editor.
     */
    void onHelp();

    void onHelpOpenHelpPerspective();

    /**
     * @brief This slot is called if the user clicks in help menu the about button
     */
    void onAbout();

  public:

    QmitkFlowApplicationWorkbenchWindowAdvisorHack();
    ~QmitkFlowApplicationWorkbenchWindowAdvisorHack() override;

    static QmitkFlowApplicationWorkbenchWindowAdvisorHack* undohack;
};

#endif
