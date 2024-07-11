/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkToolSelectionBox_h
#define QmitkToolSelectionBox_h

#include "QmitkToolGUIArea.h"
#include <MitkSegmentationUIExports.h>

#include "mitkToolManager.h"

#include <QButtonGroup>
#include <QGridLayout>
#include <QWidget>

#include <map>

class QmitkToolGUI;

/**
  \brief Display the tool selection state of a mitk::ToolManager

  \sa mitk::ToolManager

  \ingroup org_mitk_gui_qt_interactivesegmentation
  \ingroup ToolManagerEtAl

  This widget graphically displays the active tool of a mitk::ToolManager as a set
  of toggle buttons. Each button show the identification of a Tool (icon and name).
  When a button's toggle state is "down", the tool is activated. When a different button
  is clicked, the active tool is switched. When you click an already active button, the
  associated tool is deactivated with no replacement, which means that no tool is active
  then.

  When this widget is enabled/disabled it (normally) also enables/disables the tools. There
  could be cases where two QmitkToolSelectionBox widgets are associated to the same ToolManager,
  but if this happens, please look deeply into the code.

  Last contributor: $Author: maleike $
*/
class MITKSEGMENTATIONUI_EXPORT QmitkToolSelectionBox : public QWidget
//!
{
  Q_OBJECT

public:
  QmitkToolSelectionBox(QWidget *parent = nullptr, mitk::DataStorage *storage = nullptr);
  ~QmitkToolSelectionBox() override;

  mitk::ToolManager *GetToolManager();
  void SetToolManager(mitk::ToolManager &); // no nullptr pointer allowed here, a manager is required

  /**
    You may specify a list of tool "groups" that should be displayed in this widget. Every Tool can report its group
    as a string. This method will try to find the tool's group inside the supplied string \c toolGroups . If there is
    a match,
    the tool is displayed. Effectively, you can provide a human readable list like "default, lymphnodevolumetry,
    oldERISstuff".

    @param toolGroups
  */
  void SetDisplayedToolGroups(const std::string &toolGroups = nullptr);

  void OnToolManagerToolModified();
  void OnToolManagerReferenceDataModified();
  void OnToolManagerWorkingDataModified();

  void OnToolGUIProcessEventsMessage();
  void OnToolErrorMessage(std::string s);
  void OnGeneralToolMessage(std::string s);

  void RecreateButtons();

signals:

  /// Whenever a tool is activated. id is the index of the active tool. Counting starts at 0, -1 indicates "no tool
  /// selected"
  /// This signal is also emitted, when the whole QmitkToolSelectionBox get disabled. Then it will claim
  /// ToolSelected(-1)
  /// When it is enabled again, there will be another ToolSelected event with the tool that is currently selected
  void ToolSelected(int id);

public slots:

  virtual void setEnabled(bool);

  virtual void SetLayoutColumns(int);
  virtual void SetShowNames(bool);
  virtual void SetGenerateAccelerators(bool);

  virtual void SetToolGUIArea(QWidget *parentWidget);

protected slots:

  void toolButtonClicked(int id);
  void UpdateButtonsEnabledState();

protected:

  void SetOrUnsetButtonForActiveTool();

  mitk::ToolManager::Pointer m_ToolManager;

  bool m_SelfCall;

  std::string m_DisplayedGroups;

  /// stores relationship between button IDs of the Qt widget and tool IDs of ToolManager
  std::map<int, int> m_ButtonIDForToolID;
  /// stores relationship between button IDs of the Qt widget and tool IDs of ToolManager
  std::map<int, int> m_ToolIDForButtonID;

  int m_LayoutColumns;
  bool m_ShowNames;
  bool m_GenerateAccelerators;

  QWidget *m_ToolGUIWidget;
  QmitkToolGUI *m_LastToolGUI;

  // store buttons in this group
  QButtonGroup *m_ToolButtonGroup;
  QGridLayout *m_ButtonLayout;
};

#endif
