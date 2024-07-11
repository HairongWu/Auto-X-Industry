/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef QmitkAbstractRenderEditor_h
#define QmitkAbstractRenderEditor_h

#include <berryQtEditorPart.h>

#include "mitkIRenderWindowPart.h"

#include <mitkIDataStorageReference.h>
#include <mitkDataStorage.h>

#include <org_mitk_gui_qt_common_Export.h>

class QmitkAbstractRenderEditorPrivate;

namespace mitk
{
  class IPreferences;
}

/**
 * \ingroup org_mitk_gui_qt_common
 *
 * \brief A convenient base class for MITK render window BlueBerry Editors.
 *
 * QmitkAbstractRenderEditor provides several convenience methods that ease the introduction of
 * a new editor for rendering a MITK DataStorage:
 *
 * <ol>
 *   <li> Access to the DataStorage (~ the shared data repository)
 *   <li> Access to and update notification for the editor's preferences
 *   <li> Default implementation of some mitk::IRenderWindowPart methods
 * </ol>
 *
 * When inheriting from QmitkAbstractRenderEditor, you must implement the following methods:
 * <ul>
 * <li>void CreateQtPartControl(QWidget* parent)
 * <li>void SetFocus()
 * </ul>
 *
 * You may reimplement the following private virtual methods to be notified about certain changes:
 * <ul>
 * <li>void OnPreferencesChanged(const mitk::IPreferences*)
 * </ul>
 *
 * \see IRenderWindowPart
 * \see ILinkedRenderWindowPart
 */
class MITK_QT_COMMON QmitkAbstractRenderEditor : public berry::QtEditorPart,
    public virtual mitk::IRenderWindowPart
{
  Q_OBJECT
  Q_INTERFACES(mitk::IRenderWindowPart)

public:

  berryObjectMacro(QmitkAbstractRenderEditor, QtEditorPart, mitk::IRenderWindowPart);

  /**
  * \see mitk::IRenderWindowPart::GetSelectedTimePoint()
  This default implementation assumes that all renderer use the same TimeNavigationControl
  provided by this class (GetTimeNavigationControl()).
  */
  mitk::TimePointType GetSelectedTimePoint(const QString& id = QString()) const override;

  QmitkAbstractRenderEditor();
  ~QmitkAbstractRenderEditor() override;

protected:

  /**
   * Initializes this editor by checking for a valid mitk::DataStorageEditorInput as <code>input</code>.
   *
   * \see berry::IEditorPart::Init
   */
  void Init(berry::IEditorSite::Pointer site, berry::IEditorInput::Pointer input) override;

  /**
   * Get a reference to the DataStorage set by the editor input.
   */
  virtual mitk::IDataStorageReference::Pointer GetDataStorageReference() const;

  /**
   * Get the DataStorage set by the editor input.
   */
  virtual mitk::DataStorage::Pointer GetDataStorage() const;

  /**
   * Get the preferences for this editor.
   */
  virtual mitk::IPreferences* GetPreferences() const;

  /**
   * Get the RenderingManager used by this editor. This default implementation uses the
   * global MITK RenderingManager provided by mitk::RenderingManager::GetInstance().
   *
   * \see mitk::IRenderWindowPart::GetRenderingManager
   */
  mitk::IRenderingManager* GetRenderingManager() const override;

  /**
   * Request an update of this editor's render windows.
   * This implementation calls mitk::IRenderingManager::RequestUpdate on the rendering
   * manager interface returned by GetRenderingManager();
   *
   * \param requestType The type of render windows for which an update is requested.
   *
   * \see mitk::IRenderWindowPart::RequestUpdate
   */
  void RequestUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) override;

  /**
   * Force an immediate update of this editor's render windows.
   * This implementation calls mitk::IRenderingManager::ForceImmediateUpdate() on the rendering
   * manager interface returned by GetRenderingManager();
   *
   * \param requestType The type of render windows for which an immediate update is forced.
   *
   * \see mitk::IRenderWindowPart::ForceImmediateUpdate
   */
  void ForceImmediateUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) override;

  /**
   * Get the time navigation controller for this editor.
   * This implementation returns the TimeNavigationController returned by the mitk::IRenderingManager::GetTimeNavigationController()
   * method of the interface implementation returned by GetRenderingManager().
   *
   * \see mitk::IRenderingManager::GetTimeNavigationController
   */
  mitk::TimeNavigationController* GetTimeNavigationController() const override;

  /** \see berry::IEditorPart::DoSave */
  void DoSave() override;

  /** \see berry::IEditorPart::DoSaveAs */
  void DoSaveAs() override;

  /** \see berry::IEditorPart::IsDirty */
  bool IsDirty() const override;

  /** \see berry::IEditorPart::IsSaveAsAllowed */
  bool IsSaveAsAllowed() const override;

private:

  virtual void OnPreferencesChanged(const mitk::IPreferences*);

private:

  QScopedPointer<QmitkAbstractRenderEditorPrivate> d;

};

#endif
