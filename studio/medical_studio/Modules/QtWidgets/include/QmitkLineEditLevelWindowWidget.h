/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkLineEditLevelWindowWidget_h
#define QmitkLineEditLevelWindowWidget_h

#include <MitkQtWidgetsExports.h>

// mitk core
#include <mitkLevelWindowManager.h>

// qt
#include <QWidget>

class QmitkLevelWindowWidgetContextMenu;
class QLineEdit;

/**
 * \ingroup QmitkModule
 * \brief Provides a widget with two lineedit fields, one to change the
 * window value of the current image and one to change the level value of
 * the current image.
 */
class MITKQTWIDGETS_EXPORT QmitkLineEditLevelWindowWidget : public QWidget
{
  Q_OBJECT

public:
  /// constructor
  QmitkLineEditLevelWindowWidget(QWidget *parent = nullptr, Qt::WindowFlags f = {});

  /// destructor
  ~QmitkLineEditLevelWindowWidget() override;

  /// inputfield for level value
  QLineEdit *m_LevelInput;

  /// inputfield for window value
  QLineEdit *m_WindowInput;

  /*!
  *  data structure which stores the values manipulated
  *  by a QmitkLineEditLevelWindowWidget
  */
  mitk::LevelWindow m_LevelWindow;

  /// manager who is responsible to collect and deliver changes on Level/Window
  mitk::LevelWindowManager::Pointer m_Manager;

  /// sets the manager who is responsible to collect and deliver changes on Level/Window
  void SetLevelWindowManager(mitk::LevelWindowManager *levelWindowManager);

  /// sets the DataStorage which holds all image-nodes
  void SetDataStorage(mitk::DataStorage *ds);

  /// returns the manager who is responsible to collect and deliver changes on Level/Window
  mitk::LevelWindowManager *GetManager();

private:
  /// creates the contextmenu for this widget from class QmitkLevelWindowWidgetContextMenu
  void contextMenuEvent(QContextMenuEvent *) override;

  /// change notifications from the mitkLevelWindowManager
  void OnPropertyModified(const itk::EventObject &e);

public Q_SLOTS:

  /** @brief Read the levelInput and change level and slider when the button "ENTER" was pressed
  *          in the windowInput-LineEdit.
  */
  void SetLevelValue();
  /** @brief Read the windowInput and change window and slider when the button "ENTER" was pressed
  *          in the windowInput-LineEdit.
  */
  void SetWindowValue();

protected:
  unsigned long m_ObserverTag;
  bool m_IsObserverTagSet;

  QmitkLevelWindowWidgetContextMenu *m_Contextmenu;
};

#endif
